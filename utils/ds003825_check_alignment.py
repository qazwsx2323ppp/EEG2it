import argparse
import math
import os
from typing import List, Tuple

import numpy as np

from dataset_ds import _read_events_table, _resolve_subject_files


def _float(x: str) -> float:
    try:
        return float(str(x).strip())
    except Exception:
        return float("nan")


def _nearest_abs_diff(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """
    For each value in a, compute abs diff to nearest value in b.
    Returns (median, mean) of diffs.
    """
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan")
    b_sorted = np.sort(b)
    # searchsorted to find insertion points
    idx = np.searchsorted(b_sorted, a)
    idx0 = np.clip(idx - 1, 0, b_sorted.size - 1)
    idx1 = np.clip(idx, 0, b_sorted.size - 1)
    d0 = np.abs(a - b_sorted[idx0])
    d1 = np.abs(a - b_sorted[idx1])
    d = np.minimum(d0, d1)
    return float(np.median(d)), float(np.mean(d))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Check alignment between BrainVision marker events and BIDS events.tsv timing columns for ds003825."
    )
    ap.add_argument("--bids-root", required=True, help="BIDS root")
    ap.add_argument("--subject", required=True, help="Subject id (e.g. sub-01)")
    ap.add_argument("--n", type=int, default=8, help="How many example rows/events to print")
    args = ap.parse_args()

    bids_root = args.bids_root
    subject = args.subject
    vhdr_path, events_path = _resolve_subject_files(bids_root, subject)
    rows = _read_events_table(events_path)
    if not rows:
        print(f"[align] empty events table: {events_path}")
        return 0

    # Load raw with MNE and extract annotation-derived events (from .vmrk).
    import mne

    raw = mne.io.read_raw_brainvision(vhdr_path, preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    # Use annotations -> events. We don't rely on exact event ids here; we just want timing anchors.
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    anno_samp = events[:, 0].astype(np.int64)
    anno_sec = (anno_samp - int(getattr(raw, "first_samp", 0))) / sfreq
    # Prefer stimulus-onset markers (E  1) if present; this avoids "nearest event" matching to E2/E3.
    e1_key = None
    for k in event_id.keys():
        if str(k).strip() in {"Event/E  1", "Event/E1", "E1"}:
            e1_key = k
            break
    if e1_key is not None:
        e1_code = int(event_id[e1_key])
        e1_mask = events[:, 2] == e1_code
        e1_samp = events[e1_mask][:, 0].astype(np.int64)
        e1_sec = (e1_samp - int(getattr(raw, "first_samp", 0))) / sfreq
    else:
        e1_sec = anno_sec

    # Parse candidate timing columns from events.tsv.
    onset = np.asarray([_float(r.get("onset", "nan")) for r in rows], dtype=np.float64)
    time_stimon = np.asarray([_float(r.get("time_stimon", "nan")) for r in rows], dtype=np.float64)
    sample = np.asarray([_float(r.get("sample", "nan")) for r in rows], dtype=np.float64)

    def finite(x: np.ndarray) -> np.ndarray:
        return x[np.isfinite(x)]

    candidates: List[Tuple[str, np.ndarray]] = []
    candidates.append(("time_stimon_as_seconds", finite(time_stimon)))
    candidates.append(("onset_as_seconds", finite(onset)))
    candidates.append(("onset_ms_to_seconds", finite(onset / 1000.0)))
    candidates.append(("onset_samples_at_orig", finite(onset / sfreq)))
    if np.isfinite(sample).any():
        candidates.append(("sample_as_seconds", finite(sample)))
        candidates.append(("sample_ms_to_seconds", finite(sample / 1000.0)))
        candidates.append(("sample_samples_at_orig", finite(sample / sfreq)))

    print(f"[align] subject={subject} raw_sfreq={sfreq} annotations={len(anno_sec)} E1={len(e1_sec)} events_rows={len(rows)}")
    print(f"[align] event_id_keys(sample)={list(event_id)[:10]}")
    if e1_key is not None:
        print(f"[align] using E1 key: {e1_key} -> code={event_id[e1_key]}")
    else:
        print("[align] E1 key not found; using all annotation events.")

    # Score each candidate by median/mean nearest diff to E1 annotation times (seconds).
    for name, sec in candidates:
        if sec.size == 0:
            continue
        # Only compare first chunk to keep it fast.
        sec2 = sec[: min(sec.size, 5000)]
        med, mean = _nearest_abs_diff(sec2, e1_sec)
        print(f"[align] {name}: finite={sec.size} nearest_diff_sec_to_E1 median={med:.4f} mean={mean:.4f}")

        # Also check ordered alignment (i-th with i-th) for the compared prefix.
        # This is more stringent than nearest-event matching.
        if e1_sec.size >= sec2.size and sec2.size > 0:
            diff = sec2 - e1_sec[: sec2.size]
            med2 = float(np.median(np.abs(diff)))
            mean2 = float(np.mean(np.abs(diff)))
            print(f"[align]   ordered_abs_diff_sec median={med2:.4f} mean={mean2:.4f}")

    # Print a few example rows from events.tsv (onset vs time_stimon) and nearest annotation.
    print("[align] examples (events.tsv -> nearest annotation):")
    for i in range(min(args.n, len(rows))):
        r = rows[i]
        o = _float(r.get("onset", "nan"))
        ts = _float(r.get("time_stimon", "nan"))
        # choose whichever is finite for lookup
        look = ts if math.isfinite(ts) else (o / 1000.0 if o > 1000 else o)
        # nearest E1 annotation
        if e1_sec.size:
            j = int(np.argmin(np.abs(e1_sec - look)))
            nearest = float(e1_sec[j])
            delta = float(look - nearest)
        else:
            nearest, delta = float("nan"), float("nan")
        print(f"  [{i}] onset={o} time_stimon={ts} -> look={look:.4f}s nearest_anno={nearest:.4f}s delta={delta:.4f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
