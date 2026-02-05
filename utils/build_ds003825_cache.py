import argparse
import os

from dataset_ds import Ds003825TripletDataset


def main():
    ap = argparse.ArgumentParser(description="Pre-build ds003825 paper-style cache (npy/pt) to speed up training.")
    ap.add_argument("--bids-root", required=True, help="BIDS root (contains dataset_description.json)")
    ap.add_argument("--text-vec", required=True, help="Path to concept text vectors .npy (1854x512)")
    ap.add_argument("--image-vec", default="", help="Optional concept image vectors .npy (1854x512)")
    ap.add_argument("--cache-dir", required=True, help="Where to write cache files")
    ap.add_argument("--cache-format", choices=["npy", "pt"], default="npy")
    ap.add_argument("--subjects", default="", help="Comma-separated subject ids (e.g. sub-01,sub-02). Empty=all.")
    ap.add_argument("--exclude-subjects", default="", help="Comma-separated subjects to exclude.")
    ap.add_argument("--exclude-targets", action="store_true", default=False)
    ap.add_argument("--baseline-correction", action="store_true", default=False)
    ap.add_argument("--l-freq", type=float, default=0.1)
    ap.add_argument("--h-freq", type=float, default=100.0)
    ap.add_argument("--resample-sfreq", type=float, default=250.0)
    ap.add_argument("--tmin", type=float, default=-0.1)
    ap.add_argument("--tmax", type=float, default=1.0)
    ap.add_argument("--n-channels-out", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    # Construct a minimal cfg_data object the dataset understands.
    cfg_data = {
        "backend": "ds003825_bids",
        "dataset_impl": "ds003825_bids",
        "eeg_path": args.bids_root,
        "text_vec_path": args.text_vec,
        "image_vec_path": args.image_vec or args.text_vec,
        "cache_dir": args.cache_dir,
        "cache_format": args.cache_format,
        "subjects": args.subjects,
        "exclude_subjects": args.exclude_subjects,
        "exclude_targets": bool(args.exclude_targets),
        "baseline_correction": bool(args.baseline_correction),
        "l_freq": float(args.l_freq),
        "h_freq": float(args.h_freq),
        "resample_sfreq": float(args.resample_sfreq),
        "tmin": float(args.tmin),
        "tmax": float(args.tmax),
        "n_channels_out": int(args.n_channels_out),
        # Build cache only; splits don't matter here.
        "split_by": "subject",
        "subject_split": [1.0, 0.0, 0.0],
        "return_concept_id": True,
        "lru_subjects": 1,
    }

    ds = Ds003825TripletDataset(cfg_data, mode="train", split_index=0)
    # Force caching for every subject selected by cfg_data.
    for sub in ds.subjects:
        ds._ensure_subject_cached(sub)  # noqa: SLF001 (intentional internal call)
    print(f"Done. Cache written to: {args.cache_dir}")


if __name__ == "__main__":
    raise SystemExit(main())

