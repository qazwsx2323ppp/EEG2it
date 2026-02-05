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
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


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
    n_channels_out: int,
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

    # (Rule 3) FIR(Hamming) bandpass 0.1–100Hz.
    raw.filter(l_freq=l_freq, h_freq=h_freq, method="fir", fir_window="hamming", phase="zero", verbose="ERROR")
    # (Rule 3) Average reference.
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    # (Rule 3) Resample to 250Hz.
    raw.resample(resample_sfreq, npad="auto", verbose="ERROR")

    # (Rule 2 & 6) Build stimulus-onset events from events table.
    rows = _read_events_table(events_path)
    concept_ids: List[int] = []
    event_samples: List[int] = []

    for r in rows:
        concept = _int_or_none(r.get("objectnumber"))
        if concept is None or concept < 0:
            continue
        if exclude_targets:
            is_target = _int_or_none(r.get("istarget"))
            if is_target is not None and is_target != 0:
                continue
        onset_sec = _float_or_none(r.get("onset"))
        if onset_sec is None:
            continue
        # Convert onset seconds -> sample index in the *resampled* raw.
        # This corresponds to trigger E1 (stimulus onset). E2/E3 are ignored by design.
        try:
            samp = int(raw.time_as_index(onset_sec)[0])
        except Exception:
            continue
        concept_ids.append(int(concept))
        event_samples.append(int(samp))

    if len(event_samples) == 0:
        raise ValueError(f"No usable stimulus onset events for {subject} from {events_path}")

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

    data = epochs.get_data().astype(np.float32, copy=False)  # [N, C, T]
    # Enforce C=64 (pad or truncate).
    n, c, t = data.shape
    if c < n_channels_out:
        pad = np.zeros((n, n_channels_out - c, t), dtype=np.float32)
        data = np.concatenate([data, pad], axis=1)
    elif c > n_channels_out:
        data = data[:, :n_channels_out, :]

    if verbose and _is_rank0():
        print(f"[ds003825] {subject}: epochs={data.shape[0]} shape={tuple(data.shape)} sfreq={epochs.info['sfreq']}")

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
        self.bids_root = str(_safe_get(cfg_data, "eeg_path", ""))
        if not _is_bids_root(self.bids_root):
            raise ValueError(f"eeg_path must be a BIDS root (missing dataset_description.json): {self.bids_root}")

        self.exclude_targets = bool(_safe_get(cfg_data, "exclude_targets", True))
        self.baseline_correction = bool(_safe_get(cfg_data, "baseline_correction", False))
        self.return_concept_id = bool(_safe_get(cfg_data, "return_concept_id", False))

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
        self.n_channels_out = int(_safe_get(cfg_data, "n_channels_out", 64))

        # Cache
        cache_dir = _safe_get(cfg_data, "cache_dir", None)
        if not cache_dir:
            # Deterministic cache folder per preprocessing setup.
            tag = f"hp{self.l_freq}_lp{self.h_freq}_refavg_rs{self.resample_sfreq}_t{self.tmin}_{self.tmax}_bl{int(self.baseline_correction)}_exT{int(self.exclude_targets)}"
            cache_dir = os.path.join(os.path.dirname(__file__), "data", "cache", f"ds003825_{_sha1(tag)[:10]}")
        self.cache_dir = os.path.abspath(str(cache_dir))
        os.makedirs(self.cache_dir, exist_ok=True)

        self.subjects = _discover_subjects(self.bids_root)

        # Split configuration
        self.split_by = str(_safe_get(cfg_data, "split_by", "subject")).lower()  # subject|trial
        self.seed = int(_safe_get(cfg_data, "seed", 2026))
        self.split_index = int(split_index)
        self.subject_split = tuple(_safe_get(cfg_data, "subject_split", (0.8, 0.1, 0.1)))
        self.trial_split = tuple(_safe_get(cfg_data, "trial_split", (0.8, 0.1, 0.1)))

        # LRU cache for subject tensors
        self._lru = _SubjectLRU(max_subjects=int(_safe_get(cfg_data, "lru_subjects", 2)))

        # Build index of trials (lazy load subject caches when needed)
        self._trials: List[_TrialRef] = []
        self._build_index()

        if _is_rank0():
            print(f"[ds003825] mode={self.mode} trials={len(self._trials)} cache_dir={self.cache_dir}")

    def _cache_path(self, subject: str) -> str:
        fname = f"{subject}_hp{self.l_freq}_lp{self.h_freq}_rs{self.resample_sfreq}_t{self.tmin}_{self.tmax}_bl{int(self.baseline_correction)}_exT{int(self.exclude_targets)}.pt"
        return os.path.join(self.cache_dir, fname)

    def _ensure_subject_cached(self, subject: str) -> None:
        path = self._cache_path(subject)
        if os.path.isfile(path):
            return
        if _is_rank0():
            print(f"[ds003825] caching {subject} -> {path}")
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
            n_channels_out=self.n_channels_out,
            verbose=False,
        )
        torch.save(tensors, path)

    def _load_subject(self, subject: str) -> Dict[str, torch.Tensor]:
        cached = self._lru.get(subject)
        if cached is not None:
            return cached
        self._ensure_subject_cached(subject)
        path = self._cache_path(subject)
        obj = torch.load(path, map_location="cpu")
        if not isinstance(obj, dict) or "eeg" not in obj or "concept_id" not in obj:
            raise ValueError(f"Bad cache file: {path}")
        self._lru.put(subject, obj)
        return obj

    def _split_subjects(self) -> Tuple[Sequence[str], Sequence[str], Sequence[str]]:
        subs = list(self.subjects)
        # Deterministic rotation by split_index to allow cross-val.
        rot = int(self.split_index) % len(subs)
        subs = subs[rot:] + subs[:rot]

        p_train, p_val, p_test = self.subject_split
        n = len(subs)
        n_test = max(1, int(round(n * float(p_test))))
        n_val = max(1, int(round(n * float(p_val))))
        test = subs[:n_test]
        val = subs[n_test : n_test + n_val]
        train = subs[n_test + n_val :]
        return train, val, test

    def _build_index(self) -> None:
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
                obj = torch.load(self._cache_path(sub), map_location="cpu")
                n_epochs = int(obj["eeg"].shape[0])
                concept = obj["concept_id"].numpy().astype(np.int32, copy=False)
                for i in range(n_epochs):
                    cid = int(concept[i])
                    if 0 <= cid < int(self.all_text_vectors.shape[0]):
                        self._trials.append(_TrialRef(subject=sub, epoch_index=i, concept_id=cid))
            return

        # trial-level split (across all subjects)
        all_trials: List[_TrialRef] = []
        for sub in self.subjects:
            self._ensure_subject_cached(sub)
            obj = torch.load(self._cache_path(sub), map_location="cpu")
            n_epochs = int(obj["eeg"].shape[0])
            concept = obj["concept_id"].numpy().astype(np.int32, copy=False)
            for i in range(n_epochs):
                cid = int(concept[i])
                if 0 <= cid < int(self.all_text_vectors.shape[0]):
                    all_trials.append(_TrialRef(subject=sub, epoch_index=i, concept_id=cid))

        rng = random.Random(self.seed + self.split_index)
        rng.shuffle(all_trials)
        p_train, p_val, p_test = self.trial_split
        n = len(all_trials)
        n_test = max(1, int(round(n * float(p_test))))
        n_val = max(1, int(round(n * float(p_val))))
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
        eeg = obj["eeg"][int(tr.epoch_index)]  # [C,T] float32
        concept = int(tr.concept_id)
        image_vec = self.all_image_vectors[concept]
        text_vec = self.all_text_vectors[concept]
        if self.return_concept_id:
            return eeg, image_vec, text_vec, concept
        return eeg, image_vec, text_vec

