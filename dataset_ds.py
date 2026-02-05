"""
Dataset implementation for OpenNeuro ds003825 (EEG-BIDS, THINGS RSVP EEG).

Goal: Provide a paper-faithful preprocessing + epoching pipeline with caching,
without modifying the legacy `dataset.py` (used by other training stages).

Paper rules (as requested by user):
1) Source: EEG-BIDS (OpenNeuro ds003825), per-subject continuous EEG + events.tsv.
   Prefer mne-bids read_raw_bids/read_events when available; otherwise fallback
   to mne.io.read_raw_brainvision + parsing events.tsv.
2) Epoch only stimulus onset events (trigger E1). Ignore E2(offset) and E3(sequence start).
   In practice, the BIDS events.tsv already describes stimulus onsets; we treat each
   row as an E1 onset and slice epochs relative to that onset.
3) Preprocess:
   - FIR (Hamming) bandpass: 0.1–100 Hz
   - Average reference
   - Resample to 250 Hz
4) Epoch: tmin=-0.1s, tmax=1.0s relative to stimulus onset
5) Output epoch: float32, shape [C, T], C=64, T≈276 at 250Hz for 1.1s window.
   Baseline correction optional (default False to match paper's technical validation).
6) Parse stimulus identifier (concept id) from events.tsv (objectnumber 0..1853),
   map to concept embeddings (npy: [1854, 512]). __getitem__ returns (eeg, image_vec, text_vec).
7) Allow exclude_targets=True to remove target (bullseye/red) trials when events include it.
   Train/val/test split configurable by subject or trial.

Caching:
- First time a subject is processed, we save epochs to a `.pt` under `cache_dir`.
- Dataset loads per-subject cached tensors lazily with a small LRU to avoid huge RAM usage.
"""

from __future__ import annotations

import csv
import hashlib
import os
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


def _zscore_channelwise(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # x: [C, T]
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / (std + float(eps))


def _is_rank0() -> bool:
    try:
        return int(os.environ.get("RANK", "0")) == 0
    except Exception:
        return True


def _is_bids_root(path: str) -> bool:
    return bool(path) and os.path.isdir(path) and os.path.isfile(os.path.join(path, "dataset_description.json"))


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _safe_get(cfg: Any, key: str, default=None):
    try:
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)
    except Exception:
        return default


def _load_concept_vectors(text_vec_path: str, image_vec_path: Optional[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    text_np = np.load(text_vec_path)
    if image_vec_path and os.path.isfile(image_vec_path):
        img_np = np.load(image_vec_path)
    else:
        # Text-only fallback: image_vec == text_vec (no need to regenerate vectors).
        img_np = text_np
    text = torch.from_numpy(text_np).float()
    img = torch.from_numpy(img_np).float()
    n = min(int(text.shape[0]), int(img.shape[0]))
    return img[:n], text[:n]


def _read_events_table(events_path: str) -> List[Dict[str, str]]:
    """
    Parse BIDS events table.
    We avoid pandas to keep dependencies minimal on servers.
    Supports TSV/CSV (auto delimiter).
    """
    if not os.path.isfile(events_path):
        raise FileNotFoundError(events_path)

    # Auto delimiter sniff (tab vs comma).
    with open(events_path, "r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,")
        reader = csv.DictReader(f, dialect=dialect)
        rows = []
        for r in reader:
            rows.append({k.strip(): (v.strip() if isinstance(v, str) else str(v)) for k, v in r.items()})
        return rows


def _float_or_none(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() == "n/a":
            return None
        return float(s)
    except Exception:
        return None


def _int_or_none(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() == "n/a":
            return None
        return int(float(s))
    except Exception:
        return None


@dataclass(frozen=True)
class _TrialRef:
    subject: str
    epoch_index: int
    concept_id: int


class _SubjectLRU:
    def __init__(self, max_subjects: int = 2):
        self.max_subjects = max(1, int(max_subjects))
        self._cache: "OrderedDict[str, Dict[str, torch.Tensor]]" = OrderedDict()

    def get(self, key: str) -> Optional[Dict[str, torch.Tensor]]:
        v = self._cache.get(key)
        if v is None:
            return None
        self._cache.move_to_end(key)
        return v

    def put(self, key: str, value: Dict[str, torch.Tensor]) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_subjects:
            self._cache.popitem(last=False)


def _discover_subjects(bids_root: str) -> List[str]:
    subs = sorted([d for d in os.listdir(bids_root) if d.startswith("sub-") and os.path.isdir(os.path.join(bids_root, d))])
    if not subs:
        raise ValueError(f"No subjects found under bids_root={bids_root}")
    return subs


def _resolve_subject_files(bids_root: str, subject: str) -> Tuple[str, str]:
    """
    Return (vhdr_path, events_path) for a subject.
    """
    eeg_dir = os.path.join(bids_root, subject, "eeg")
    if not os.path.isdir(eeg_dir):
        raise FileNotFoundError(eeg_dir)

    # Prefer task-rsvp naming but tolerate variants.
    def _first(paths: Sequence[str]) -> str:
        if not paths:
            return ""
        for p in paths:
            if "task-rsvp" in os.path.basename(p).lower():
                return p
        return paths[0]

    vhdr_candidates = sorted([os.path.join(eeg_dir, f) for f in os.listdir(eeg_dir) if f.lower().endswith(".vhdr")])
    if not vhdr_candidates:
        raise FileNotFoundError(f"No .vhdr under {eeg_dir}")
    vhdr_path = _first(vhdr_candidates)

    events_candidates = sorted([os.path.join(eeg_dir, f) for f in os.listdir(eeg_dir) if f.lower().endswith(("_events.tsv", "_events.csv"))])
    if not events_candidates:
        raise FileNotFoundError(f"No events.tsv/csv under {eeg_dir}")
    # Prefer TSV.
    events_candidates = sorted(events_candidates, key=lambda p: (0 if p.lower().endswith(".tsv") else 1, p))
    events_path = _first(events_candidates)
    return vhdr_path, events_path


def _preproc_and_epoch_subject(
    bids_root: str,
    subject: str,
    *,
    baseline_correction: bool,
    exclude_targets: bool,
    l_freq: float,
    h_freq: float,
    resample_sfreq: float,
    tmin: float,
    tmax: float,
    n_channels_epoch: int,
    n_channels_out: int,
    channel_expand_mode: str,
    n_samples_out: int,
    interp_chunk: int,
    verbose: bool,
) -> Dict[str, torch.Tensor]:
    """
    Returns dict with:
      - eeg: float32 [N, C, T]
      - concept_id: int16 [N]
    """
    import mne

    vhdr_path, events_path = _resolve_subject_files(bids_root, subject)

    # (Rule 1) Continuous EEG.
    # Prefer mne-bids when available; fallback to BrainVision.
    raw = None
    try:
        import mne_bids  # type: ignore

        # Build a best-effort BIDSPath; if it fails, fallback below.
        try:
            bids_path = mne_bids.BIDSPath(root=bids_root, subject=subject.replace("sub-", ""), datatype="eeg", task="rsvp")
            raw = mne_bids.read_raw_bids(bids_path=bids_path, verbose="ERROR")
        except Exception:
            raw = None
    except Exception:
        raw = None

    if raw is None:
        raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose="ERROR")
    else:
        raw.load_data()

    orig_sfreq = float(raw.info["sfreq"])

    # (Rule 3) FIR(Hamming) bandpass 0.1–100Hz.
    raw.filter(l_freq=l_freq, h_freq=h_freq, method="fir", fir_window="hamming", phase="zero", verbose="ERROR")
    # (Rule 3) Average reference.
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    # (Rule 3) Resample to 250Hz.
    raw.resample(resample_sfreq, npad="auto", verbose="ERROR")

    # (Rule 2 & 6) Build stimulus-onset events from events table.
    rows = _read_events_table(events_path)
    concept_ids: List[int] = []
    onset_vals: List[float] = []
    sample_vals: List[float] = []
    time_stimon_vals: List[float] = []
    time_stimoff_vals: List[float] = []

    sfreq = float(raw.info["sfreq"])
    first_samp = int(getattr(raw, "first_samp", 0))
    n_samp_total = first_samp + int(raw.n_times)
    tmin_samp = int(round(tmin * sfreq))
    tmax_samp = int(round(tmax * sfreq))

    dropped_parse = 0
    for r in rows:
        concept = _int_or_none(r.get("objectnumber"))
        if concept is None or concept < 0:
            continue
        if exclude_targets:
            is_target = _int_or_none(r.get("istarget"))
            if is_target is not None and is_target != 0:
                continue
        onset_raw = _float_or_none(r.get("onset"))
        sample_raw = _float_or_none(r.get("sample"))
        # ds003825 events.tsv often includes these timing columns, which tend to be in seconds.
        # In some exports, the standard BIDS `onset` column may be in samples/ms rather than seconds,
        # while `time_stimon` matches the real experiment clock in seconds.
        # We collect both and choose the best interpretation below.
        time_stimon_raw = _float_or_none(r.get("time_stimon"))
        time_stimoff_raw = _float_or_none(r.get("time_stimoff"))
        if onset_raw is None and sample_raw is None:
            dropped_parse += 1
            continue

        concept_ids.append(int(concept))
        onset_vals.append(float(onset_raw) if onset_raw is not None else float("nan"))
        sample_vals.append(float(sample_raw) if sample_raw is not None else float("nan"))
        time_stimon_vals.append(float(time_stimon_raw) if time_stimon_raw is not None else float("nan"))
        time_stimoff_vals.append(float(time_stimoff_raw) if time_stimoff_raw is not None else float("nan"))

    if len(concept_ids) == 0:
        raise ValueError(
            f"No usable stimulus onset events for {subject} from {events_path} "
            f"(rows={len(rows)}, kept=0, dropped_parse={dropped_parse}, dropped_oob=0, "
            f"orig_sfreq={orig_sfreq}, sfreq={sfreq}, first_samp={first_samp}, n_times={int(raw.n_times)})."
        )

    # Choose the best timing interpretation per subject.
    onset_arr = np.asarray(onset_vals, dtype=np.float64)
    sample_arr = np.asarray(sample_vals, dtype=np.float64)
    time_stimon_arr = np.asarray(time_stimon_vals, dtype=np.float64) if time_stimon_vals else np.full_like(onset_arr, np.nan)
    time_stimoff_arr = np.asarray(time_stimoff_vals, dtype=np.float64) if time_stimoff_vals else np.full_like(onset_arr, np.nan)

    def _sec_from(arr: np.ndarray, mode: str) -> np.ndarray:
        if mode == "as_seconds":
            return arr
        if mode == "ms_to_seconds":
            return arr / 1000.0
        if mode == "samples_at_orig":
            return arr / float(orig_sfreq)
        if mode == "samples_at_orig_ms":
            # sometimes stored as ms@1000Hz samples but already integer; interpret as ms
            return (arr / float(orig_sfreq)) / 1000.0
        raise ValueError(mode)

    timing_candidates: List[Tuple[str, np.ndarray]] = []
    # ds003825 (THINGS RSVP EEG) exports often store BIDS `onset` not in seconds, but in samples/ms.
    # Our alignment checks against BrainVision E1 markers show `onset_ms_to_seconds` (or equivalently
    # samples_at_orig at 1000Hz) matches E1 timestamps best, while `time_stimon` can be a separate
    # experiment clock with a non-constant offset/drift.
    #
    # Therefore, we prefer `onset`-derived timing first, and keep `time_stimon` only as a fallback.
    # onset column candidates
    timing_candidates.append(("onset_ms_to_seconds", _sec_from(onset_arr, "ms_to_seconds")))
    timing_candidates.append(("onset_samples_at_orig", _sec_from(onset_arr, "samples_at_orig")))
    timing_candidates.append(("onset_as_seconds", _sec_from(onset_arr, "as_seconds")))
    # sample column candidates
    timing_candidates.append(("sample_ms_to_seconds", _sec_from(sample_arr, "ms_to_seconds")))
    timing_candidates.append(("sample_samples_at_orig", _sec_from(sample_arr, "samples_at_orig")))
    timing_candidates.append(("sample_as_seconds", _sec_from(sample_arr, "as_seconds")))
    # Explicit experiment clock columns (often in seconds) as last resort.
    timing_candidates.append(("time_stimon_as_seconds", _sec_from(time_stimon_arr, "as_seconds")))
    timing_candidates.append(("time_stimon_ms_to_seconds", _sec_from(time_stimon_arr, "ms_to_seconds")))
    timing_candidates.append(("time_stimoff_as_seconds", _sec_from(time_stimoff_arr, "as_seconds")))

    # Tie-break preference order: earlier is better when keep counts match.
    pref = {name: i for i, (name, _) in enumerate(timing_candidates)}

    best_name = ""
    best_keep = 0
    best_samples_abs: Optional[np.ndarray] = None
    dropped_oob = 0
    for name, sec in timing_candidates:
        if not np.isfinite(sec).any():
            continue
        sec2 = np.where(np.isfinite(sec), sec, -1.0)
        try:
            idx0 = raw.time_as_index(sec2, use_rounding=True)
        except Exception:
            continue
        idx0 = np.asarray(idx0, dtype=np.int64)
        cand_abs = first_samp + idx0
        ok = (sec2 >= 0.0) & ((cand_abs + tmin_samp) >= first_samp) & ((cand_abs + tmax_samp) < n_samp_total)
        keep = int(ok.sum())
        if keep > best_keep or (keep == best_keep and keep > 0 and (pref.get(name, 1_000_000) < pref.get(best_name, 1_000_000))):
            best_keep = keep
            best_name = name
            best_samples_abs = cand_abs[ok]
            dropped_oob = int((~ok).sum())

    if best_samples_abs is None or best_keep <= 0:
        raise ValueError(
            f"No usable stimulus onset events for {subject} from {events_path} "
            f"(rows={len(rows)}, kept=0, dropped_parse={dropped_parse}, dropped_oob={len(concept_ids)}, "
            f"orig_sfreq={orig_sfreq}, sfreq={sfreq}, first_samp={first_samp}, n_times={int(raw.n_times)})."
        )

    # Align concept ids to kept indices by recomputing ok mask for best.
    sec_best = dict(timing_candidates)[best_name]
    sec_best = np.where(np.isfinite(sec_best), sec_best, -1.0)
    idx0_best = np.asarray(raw.time_as_index(sec_best, use_rounding=True), dtype=np.int64)
    cand_abs_best = first_samp + idx0_best
    ok_best = (sec_best >= 0.0) & ((cand_abs_best + tmin_samp) >= first_samp) & ((cand_abs_best + tmax_samp) < n_samp_total)
    concept_kept = np.asarray(concept_ids, dtype=np.int32)[ok_best]
    event_samples = cand_abs_best[ok_best].astype(np.int64, copy=False).tolist()
    concept_ids = concept_kept.tolist()

    if verbose and _is_rank0():
        print(f"[ds003825] {subject}: timing={best_name} kept={best_keep}/{len(ok_best)} dropped_oob={dropped_oob}")

    events = np.zeros((len(event_samples), 3), dtype=np.int64)
    events[:, 0] = np.asarray(event_samples, dtype=np.int64)
    events[:, 2] = 1  # E1
    event_id = {"E1": 1}

    # (Rule 4,5) Epoching.
    baseline = (tmin, 0.0) if baseline_correction else None
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        picks=picks,
        preload=True,
        reject=None,
        verbose="ERROR",
    )

    if len(epochs) == 0:
        # Avoid caching empty subjects; surface a helpful error.
        raise ValueError(
            f"Empty epochs for {subject}. "
            f"events_in={len(events)} kept=0 (tmin={tmin}, tmax={tmax}, sfreq={sfreq}, first_samp={first_samp}, n_samp_total={n_samp_total}). "
            f"Check event timing alignment between events.tsv and raw."
        )

    data = epochs.get_data().astype(np.float32, copy=False)  # [N, C, T]
    # Paper typically uses 64 channels, while the DreamDiffusion backbone expects 128.
    # First enforce "epoch channels" (e.g. 64), then pad/truncate to "output channels" for the model (e.g. 128).
    n, c, t = data.shape
    n_channels_epoch = int(n_channels_epoch)
    n_channels_out = int(n_channels_out)

    if c < n_channels_epoch:
        pad = np.zeros((n, n_channels_epoch - c, t), dtype=np.float32)
        data = np.concatenate([data, pad], axis=1)
    elif c > n_channels_epoch:
        data = data[:, :n_channels_epoch, :]

    c2 = int(data.shape[1])
    if c2 < n_channels_out:
        mode = str(channel_expand_mode or "zero").strip().lower()
        if mode in {"repeat", "tile"} and c2 > 0:
            reps = int((n_channels_out + c2 - 1) // c2)
            data = np.tile(data, (1, reps, 1))[:, :n_channels_out, :].astype(np.float32, copy=False)
        else:
            pad = np.zeros((n, n_channels_out - c2, t), dtype=np.float32)
            data = np.concatenate([data, pad], axis=1)
    elif c2 > n_channels_out:
        data = data[:, :n_channels_out, :]

    # Model compatibility: DreamDiffusion backbone expects a fixed n_samples (typically 512 -> 128 tokens).
    # Paper epoch at 250Hz yields ~276 samples; interpolate to n_samples_out for training.
    n_samples_out = int(n_samples_out)
    if n_samples_out > 0 and int(data.shape[-1]) != n_samples_out:
        src_len = int(data.shape[-1])
        chunk = int(interp_chunk) if int(interp_chunk) > 0 else 256
        out = np.empty((int(data.shape[0]), int(data.shape[1]), n_samples_out), dtype=np.float32)
        for s in range(0, int(data.shape[0]), chunk):
            e = min(int(data.shape[0]), s + chunk)
            x = torch.from_numpy(data[s:e]).float()  # [B,C,L]
            y = F.interpolate(x, size=n_samples_out, mode="linear", align_corners=False)
            out[s:e] = y.cpu().numpy().astype(np.float32, copy=False)
        data = out
        t = n_samples_out

    if verbose and _is_rank0():
        extra = f" (interp {src_len}->{t})" if 'src_len' in locals() else ""
        print(f"[ds003825] {subject}: epochs={data.shape[0]} shape={tuple(data.shape)} sfreq={epochs.info['sfreq']}{extra}")

    return {
        "eeg": torch.from_numpy(data),  # float32 [N,C,T]
        "concept_id": torch.tensor(concept_ids[: data.shape[0]], dtype=torch.int16),
    }


class Ds003825TripletDataset(Dataset):
    """
    Triplet dataset for ds003825.
    Returns (eeg_epoch [C,T], image_vec [D], text_vec [D]) or with concept_id if enabled.
    """

    def __init__(self, cfg_data: Any, mode: str = "train", split_index: int = 0):
        self.mode = str(mode)
        # Enable unique-concept batch sampling in main_ds.py (compatible with utils/batch_samplers.py).
        # We expose the same `backend` string the sampler checks for.
        self.backend = "ds003825"
        self.bids_root = str(_safe_get(cfg_data, "eeg_path", ""))
        if not _is_bids_root(self.bids_root):
            raise ValueError(f"eeg_path must be a BIDS root (missing dataset_description.json): {self.bids_root}")

        self.exclude_targets = bool(_safe_get(cfg_data, "exclude_targets", True))
        self.baseline_correction = bool(_safe_get(cfg_data, "baseline_correction", False))
        self.return_concept_id = bool(_safe_get(cfg_data, "return_concept_id", False))
        self.zscore = bool(_safe_get(cfg_data, "zscore", False))
        self.zscore_eps = float(_safe_get(cfg_data, "zscore_eps", 1e-6))
        self.trial_stride = int(_safe_get(cfg_data, "trial_stride", 1))

        # Vectors
        text_vec_path = str(_safe_get(cfg_data, "text_vec_path", ""))
        image_vec_path = str(_safe_get(cfg_data, "image_vec_path", "")) or None
        if not text_vec_path or not os.path.isfile(text_vec_path):
            raise FileNotFoundError(f"text_vec_path not found: {text_vec_path}")
        self.all_image_vectors, self.all_text_vectors = _load_concept_vectors(text_vec_path, image_vec_path)
        self.num_concepts = int(_safe_get(cfg_data, "num_concepts", int(self.all_text_vectors.shape[0])))

        # Preproc params (paper defaults)
        self.l_freq = float(_safe_get(cfg_data, "l_freq", 0.1))
        self.h_freq = float(_safe_get(cfg_data, "h_freq", 100.0))
        self.resample_sfreq = float(_safe_get(cfg_data, "resample_sfreq", 250.0))
        self.tmin = float(_safe_get(cfg_data, "tmin", -0.1))
        self.tmax = float(_safe_get(cfg_data, "tmax", 1.0))
        self.n_channels_epoch = int(_safe_get(cfg_data, "n_channels_epoch", 64))
        self.n_channels_out = int(_safe_get(cfg_data, "n_channels_out", 128))
        self.channel_expand_mode = str(_safe_get(cfg_data, "channel_expand_mode", "zero")).strip().lower()
        self.n_samples_out = int(_safe_get(cfg_data, "n_samples_out", 512))
        self.interp_chunk = int(_safe_get(cfg_data, "interp_chunk", 256))

        # Cache
        cache_dir = _safe_get(cfg_data, "cache_dir", None)
        if not cache_dir:
            # Deterministic cache folder per preprocessing setup.
            tag = f"hp{self.l_freq}_lp{self.h_freq}_refavg_rs{self.resample_sfreq}_t{self.tmin}_{self.tmax}_bl{int(self.baseline_correction)}_exT{int(self.exclude_targets)}"
            cache_dir = os.path.join(os.path.dirname(__file__), "data", "cache", f"ds003825_{_sha1(tag)[:10]}")
        self.cache_dir = os.path.abspath(str(cache_dir))
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_format = str(_safe_get(cfg_data, "cache_format", "npy")).lower()  # npy | pt

        self.subjects = _discover_subjects(self.bids_root)
        # Optional: restrict to a subset of subjects for quick runs.
        subjects_csv = _safe_get(cfg_data, "subjects", "")
        if subjects_csv:
            allowed = {s.strip() for s in str(subjects_csv).split(",") if s.strip()}
            self.subjects = [s for s in self.subjects if s in allowed]
        exclude_csv = _safe_get(cfg_data, "exclude_subjects", "")
        if exclude_csv:
            excluded = {s.strip() for s in str(exclude_csv).split(",") if s.strip()}
            self.subjects = [s for s in self.subjects if s not in excluded]
        if not self.subjects:
            raise ValueError("No subjects left after applying subjects/exclude_subjects filters.")

        # Split configuration
        self.split_by = str(_safe_get(cfg_data, "split_by", "subject")).lower()  # subject|trial
        self.seed = int(_safe_get(cfg_data, "seed", 2026))
        self.split_index = int(split_index)
        self.subject_split = tuple(_safe_get(cfg_data, "subject_split", (0.8, 0.1, 0.1)))
        self.trial_split = tuple(_safe_get(cfg_data, "trial_split", (0.8, 0.1, 0.1)))

        # LRU cache for subject tensors (note: each DataLoader worker has its own Dataset copy)
        self._lru = _SubjectLRU(max_subjects=int(_safe_get(cfg_data, "lru_subjects", 1)))

        # Build index of trials (lazy load subject caches when needed)
        self._trials: List[_TrialRef] = []
        self._build_index()

        # Expose concept ids for unique-concept sampler compatibility.
        self.indices = np.arange(len(self._trials), dtype=np.int64)
        self.ds_concept_ids = np.asarray([t.concept_id for t in self._trials], dtype=np.int32)

        if _is_rank0():
            print(f"[ds003825] mode={self.mode} trials={len(self._trials)} cache_dir={self.cache_dir}")

    def _cache_path(self, subject: str) -> str:
        ext = "pt" if self.cache_format == "pt" else "npy"
        ch_tag = f"ch{int(self.n_channels_epoch)}to{int(self.n_channels_out)}"
        ce_tag = f"ce{self.channel_expand_mode or 'zero'}"
        samp_tag = f"s{int(self.n_samples_out)}"
        fname = (
            f"{subject}_{ch_tag}_{ce_tag}_{samp_tag}_hp{self.l_freq}_lp{self.h_freq}_rs{self.resample_sfreq}"
            f"_t{self.tmin}_{self.tmax}_bl{int(self.baseline_correction)}_exT{int(self.exclude_targets)}.{ext}"
        )
        return os.path.join(self.cache_dir, fname)

    def _cache_paths(self, subject: str) -> Dict[str, str]:
        ch_tag = f"ch{int(self.n_channels_epoch)}to{int(self.n_channels_out)}"
        ce_tag = f"ce{self.channel_expand_mode or 'zero'}"
        samp_tag = f"s{int(self.n_samples_out)}"
        base = (
            f"{subject}_{ch_tag}_{ce_tag}_{samp_tag}_hp{self.l_freq}_lp{self.h_freq}_rs{self.resample_sfreq}"
            f"_t{self.tmin}_{self.tmax}_bl{int(self.baseline_correction)}_exT{int(self.exclude_targets)}"
        )
        if self.cache_format == "pt":
            return {"pt": os.path.join(self.cache_dir, f"{base}.pt")}
        return {
            "eeg": os.path.join(self.cache_dir, f"{base}_eeg.npy"),
            "concept": os.path.join(self.cache_dir, f"{base}_concept.npy"),
        }

    def _lock_path(self, subject: str) -> str:
        return self._cache_path(subject) + ".lock"

    def _acquire_lock(self, lock_path: str, timeout_s: float = 900.0, poll_s: float = 0.2) -> bool:
        """
        Cross-process lock using atomic create. Works for DDP ranks on the same filesystem.
        """
        start = time.time()
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.close(fd)
                return True
            except FileExistsError:
                if (time.time() - start) > timeout_s:
                    return False
                time.sleep(poll_s)

    def _release_lock(self, lock_path: str) -> None:
        try:
            os.remove(lock_path)
        except Exception:
            pass

    def _safe_load_cache(self, path: str) -> Dict[str, torch.Tensor]:
        """
        Load cache dict. If the cache is corrupted/partial (common under concurrent writes),
        raise so caller can rebuild it.
        """
        if self.cache_format == "pt":
            return torch.load(path, map_location="cpu", weights_only=False)
        raise RuntimeError("Use _safe_load_cache_npy for npy cache format")

    def _safe_load_cache_npy(self, paths: Dict[str, str]) -> Dict[str, Any]:
        eeg_path = paths["eeg"]
        concept_path = paths["concept"]
        if not (os.path.isfile(eeg_path) and os.path.isfile(concept_path)):
            raise FileNotFoundError(f"Missing cache files: {eeg_path} / {concept_path}")
        eeg_mm = np.load(eeg_path, mmap_mode="r")  # [N,C,T] float32 memmap
        concept_np = np.load(concept_path)  # small, load fully
        return {"eeg_mm": eeg_mm, "concept_id": torch.from_numpy(concept_np).to(torch.int16)}

    def _ensure_subject_cached(self, subject: str) -> None:
        paths = self._cache_paths(subject)
        # Fast path
        if self.cache_format == "pt":
            if os.path.isfile(paths["pt"]):
                return
        else:
            if os.path.isfile(paths["eeg"]) and os.path.isfile(paths["concept"]):
                return

        lock_path = self._lock_path(subject)
        got = self._acquire_lock(lock_path)
        if not got:
            raise TimeoutError(f"Timed out waiting for cache lock: {lock_path}")

        try:
            # Another rank may have created it while we waited.
            if self.cache_format == "pt":
                if os.path.isfile(paths["pt"]):
                    return
            else:
                if os.path.isfile(paths["eeg"]) and os.path.isfile(paths["concept"]):
                    return

            if _is_rank0():
                target = paths["pt"] if self.cache_format == "pt" else paths["eeg"]
                print(f"[ds003825] caching {subject} -> {target}")

            tensors = _preproc_and_epoch_subject(
                self.bids_root,
                subject,
                baseline_correction=self.baseline_correction,
                exclude_targets=self.exclude_targets,
                l_freq=self.l_freq,
                h_freq=self.h_freq,
                resample_sfreq=self.resample_sfreq,
                tmin=self.tmin,
                tmax=self.tmax,
                n_channels_epoch=self.n_channels_epoch,
                n_channels_out=self.n_channels_out,
                channel_expand_mode=self.channel_expand_mode,
                n_samples_out=self.n_samples_out,
                interp_chunk=self.interp_chunk,
                verbose=False,
            )

            if self.cache_format == "pt":
                path = paths["pt"]
                tmp = f"{path}.tmp.{os.getpid()}"
                torch.save(tensors, tmp, _use_new_zipfile_serialization=True)
                os.replace(tmp, path)  # atomic on POSIX
            else:
                eeg = tensors["eeg"].numpy().astype(np.float32, copy=False)
                concept = tensors["concept_id"].numpy().astype(np.int16, copy=False)
                eeg_path = paths["eeg"]
                concept_path = paths["concept"]
                # NOTE: numpy.save appends ".npy" if the filename doesn't end with it.
                # Ensure our tmp file names end with ".npy" so os.replace can find them.
                eeg_tmp = f"{eeg_path}.tmp.{os.getpid()}.npy"
                concept_tmp = f"{concept_path}.tmp.{os.getpid()}.npy"
                np.save(eeg_tmp, eeg)
                np.save(concept_tmp, concept)
                os.replace(eeg_tmp, eeg_path)
                os.replace(concept_tmp, concept_path)
        finally:
            self._release_lock(lock_path)

    def _load_subject(self, subject: str) -> Dict[str, torch.Tensor]:
        cached = self._lru.get(subject)
        if cached is not None:
            return cached
        self._ensure_subject_cached(subject)
        paths = self._cache_paths(subject)

        if self.cache_format == "pt":
            path = paths["pt"]
            obj = self._safe_load_cache(path)
            if not isinstance(obj, dict) or "eeg" not in obj or "concept_id" not in obj:
                raise ValueError(f"Bad cache file: {path}")
            self._lru.put(subject, obj)
            return obj

        obj2 = self._safe_load_cache_npy(paths)
        # Store memmap + tensor dict; __getitem__ handles conversion to torch.
        self._lru.put(subject, obj2)  # type: ignore[arg-type]
        return obj2  # type: ignore[return-value]

    def _split_subjects(self) -> Tuple[Sequence[str], Sequence[str], Sequence[str]]:
        subs = list(self.subjects)
        # Deterministic rotation by split_index to allow cross-val.
        rot = int(self.split_index) % len(subs)
        subs = subs[rot:] + subs[:rot]

        p_train, p_val, p_test = self.subject_split
        n = len(subs)
        # Allow zero-sized splits when user sets p_val/p_test to 0.0 (useful for quick sanity runs).
        n_test = int(round(n * float(p_test))) if float(p_test) > 0.0 else 0
        n_val = int(round(n * float(p_val))) if float(p_val) > 0.0 else 0
        # Ensure at least 1 train subject if possible by trimming val/test first.
        if (n - n_test - n_val) <= 0 and n > 0:
            if n_val > 0:
                n_val = max(0, n_val - 1)
            if (n - n_test - n_val) <= 0 and n_test > 0:
                n_test = max(0, n_test - 1)
        test = subs[:n_test]
        val = subs[n_test : n_test + n_val]
        train = subs[n_test + n_val :]
        return train, val, test

    def _build_index(self) -> None:
        stride = max(1, int(self.trial_stride))
        if self.split_by == "subject":
            train_subs, val_subs, test_subs = self._split_subjects()
            if self.mode == "train":
                use = list(train_subs)
            elif self.mode == "val":
                use = list(val_subs)
            elif self.mode == "test":
                use = list(test_subs)
            else:
                use = list(train_subs)

            for sub in use:
                self._ensure_subject_cached(sub)
                if self.cache_format == "pt":
                    obj = self._safe_load_cache(self._cache_paths(sub)["pt"])
                    n_epochs = int(obj["eeg"].shape[0])
                    concept = obj["concept_id"].numpy().astype(np.int32, copy=False)
                else:
                    obj = self._safe_load_cache_npy(self._cache_paths(sub))
                    n_epochs = int(obj["eeg_mm"].shape[0])
                    concept = obj["concept_id"].numpy().astype(np.int32, copy=False)
                for i in range(n_epochs):
                    if stride > 1 and (i % stride) != 0:
                        continue
                    cid = int(concept[i])
                    if 0 <= cid < int(self.all_text_vectors.shape[0]):
                        self._trials.append(_TrialRef(subject=sub, epoch_index=i, concept_id=cid))
            return

        # trial-level split (across all subjects)
        all_trials: List[_TrialRef] = []
        for sub in self.subjects:
            self._ensure_subject_cached(sub)
            if self.cache_format == "pt":
                obj = self._safe_load_cache(self._cache_paths(sub)["pt"])
                n_epochs = int(obj["eeg"].shape[0])
                concept = obj["concept_id"].numpy().astype(np.int32, copy=False)
            else:
                obj = self._safe_load_cache_npy(self._cache_paths(sub))
                n_epochs = int(obj["eeg_mm"].shape[0])
                concept = obj["concept_id"].numpy().astype(np.int32, copy=False)
            for i in range(n_epochs):
                if stride > 1 and (i % stride) != 0:
                    continue
                cid = int(concept[i])
                if 0 <= cid < int(self.all_text_vectors.shape[0]):
                    all_trials.append(_TrialRef(subject=sub, epoch_index=i, concept_id=cid))

        rng = random.Random(self.seed + self.split_index)
        rng.shuffle(all_trials)
        p_train, p_val, p_test = self.trial_split
        n = len(all_trials)
        n_test = int(round(n * float(p_test))) if float(p_test) > 0.0 else 0
        n_val = int(round(n * float(p_val))) if float(p_val) > 0.0 else 0
        if (n - n_test - n_val) <= 0 and n > 0:
            if n_val > 0:
                n_val = max(0, n_val - 1)
            if (n - n_test - n_val) <= 0 and n_test > 0:
                n_test = max(0, n_test - 1)
        test = all_trials[:n_test]
        val = all_trials[n_test : n_test + n_val]
        train = all_trials[n_test + n_val :]

        if self.mode == "train":
            self._trials = train
        elif self.mode == "val":
            self._trials = val
        elif self.mode == "test":
            self._trials = test
        else:
            self._trials = train

    def __len__(self) -> int:
        return len(self._trials)

    def __getitem__(self, idx: int):
        tr = self._trials[int(idx)]
        subj = tr.subject
        obj = self._load_subject(subj)
        if self.cache_format == "pt":
            eeg = obj["eeg"][int(tr.epoch_index)]  # [C,T] float32
        else:
            eeg_np = obj["eeg_mm"][int(tr.epoch_index)]  # numpy memmap slice [C,T]
            eeg = torch.from_numpy(np.asarray(eeg_np, dtype=np.float32))

        if self.zscore:
            eeg = _zscore_channelwise(eeg, eps=float(self.zscore_eps))
        concept = int(tr.concept_id)
        image_vec = self.all_image_vectors[concept]
        text_vec = self.all_text_vectors[concept]
        if self.return_concept_id:
            return eeg, image_vec, text_vec, concept
        return eeg, image_vec, text_vec
