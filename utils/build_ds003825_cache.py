import argparse
import os
from pathlib import Path

from dataset_ds import Ds003825TripletDataset


def _load_cfg_from_hydra(config_name: str):
    try:
        from hydra import compose, initialize_config_dir
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Hydra is required for --config-name, but it could not be imported. "
            "Install project deps or run without --config-name."
        ) from e

    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "configs"
    if not config_dir.exists():
        raise FileNotFoundError(f"configs/ not found at: {config_dir}")

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name=config_name)


def main():
    ap = argparse.ArgumentParser(description="Pre-build ds003825 paper-style cache (npy/pt) to speed up training.")
    ap.add_argument(
        "--config-name",
        default="",
        help="Optional hydra config name under configs/ (e.g. ds003825_quick15m). When set, "
        "you can omit --bids-root/--text-vec/--cache-dir and they will be read from config.",
    )
    ap.add_argument("--bids-root", default="", help="BIDS root (contains dataset_description.json)")
    ap.add_argument("--text-vec", default="", help="Path to concept text vectors .npy (1854x512)")
    ap.add_argument("--image-vec", default="", help="Optional concept image vectors .npy (1854x512)")
    ap.add_argument("--cache-dir", default="", help="Where to write cache files")
    ap.add_argument("--cache-format", choices=["npy", "pt"], default="")
    ap.add_argument("--subjects", default="", help="Comma-separated subject ids (e.g. sub-01,sub-02). Empty=all.")
    ap.add_argument("--exclude-subjects", default="", help="Comma-separated subjects to exclude.")

    ex_group = ap.add_mutually_exclusive_group()
    ex_group.add_argument("--exclude-targets", action="store_true", default=None)
    ex_group.add_argument("--include-targets", action="store_true", default=None, help="Do not exclude targets.")

    bl_group = ap.add_mutually_exclusive_group()
    bl_group.add_argument("--baseline-correction", action="store_true", default=None)
    bl_group.add_argument("--no-baseline-correction", action="store_true", default=None, help="Disable baseline correction.")

    ap.add_argument("--l-freq", type=float, default=None)
    ap.add_argument("--h-freq", type=float, default=None)
    ap.add_argument("--resample-sfreq", type=float, default=None)
    ap.add_argument("--tmin", type=float, default=None)
    ap.add_argument("--tmax", type=float, default=None)
    ap.add_argument("--n-channels-epoch", type=int, default=None, help="Real EEG channels after picks (paper uses 64).")
    ap.add_argument("--n-channels-out", type=int, default=None, help="Channels fed to model (DreamDiffusion expects 128).")
    ap.add_argument("--n-samples-out", type=int, default=None, help="Samples fed to model (DreamDiffusion expects 512).")
    ap.add_argument("--interp-chunk", type=int, default=None, help="Chunk size (epochs) for time interpolation.")
    args = ap.parse_args()

    cfg = None
    if args.config_name:
        cfg = _load_cfg_from_hydra(args.config_name)

    def cfg_get(path: str, default):
        if cfg is None:
            return default
        cur = cfg
        for key in path.split("."):
            if key not in cur:
                return default
            cur = cur[key]
        return cur

    bids_root = args.bids_root or cfg_get("data.eeg_path", "")
    text_vec = args.text_vec or cfg_get("data.text_vec_path", "")
    image_vec = args.image_vec or cfg_get("data.image_vec_path", "") or text_vec
    cache_dir = args.cache_dir or cfg_get("data.cache_dir", "")
    cache_format = args.cache_format or cfg_get("data.cache_format", "npy")
    subjects = args.subjects or cfg_get("data.subjects", "")
    exclude_subjects = args.exclude_subjects or cfg_get("data.exclude_subjects", "")

    if args.exclude_targets is True:
        exclude_targets = True
    elif args.include_targets is True:
        exclude_targets = False
    else:
        exclude_targets = bool(cfg_get("data.exclude_targets", False))

    if args.baseline_correction is True:
        baseline_correction = True
    elif args.no_baseline_correction is True:
        baseline_correction = False
    else:
        baseline_correction = bool(cfg_get("data.baseline_correction", False))

    l_freq = float(args.l_freq if args.l_freq is not None else cfg_get("data.l_freq", 0.1))
    h_freq = float(args.h_freq if args.h_freq is not None else cfg_get("data.h_freq", 100.0))
    resample_sfreq = float(
        args.resample_sfreq if args.resample_sfreq is not None else cfg_get("data.resample_sfreq", 250.0)
    )
    tmin = float(args.tmin if args.tmin is not None else cfg_get("data.tmin", -0.1))
    tmax = float(args.tmax if args.tmax is not None else cfg_get("data.tmax", 1.0))
    n_channels_epoch = int(args.n_channels_epoch if args.n_channels_epoch is not None else cfg_get("data.n_channels_epoch", 64))
    n_channels_out = int(args.n_channels_out if args.n_channels_out is not None else cfg_get("data.n_channels_out", 128))
    n_samples_out = int(args.n_samples_out if args.n_samples_out is not None else cfg_get("data.n_samples_out", 512))
    interp_chunk = int(args.interp_chunk if args.interp_chunk is not None else cfg_get("data.interp_chunk", 256))

    missing = [k for k, v in [("bids_root", bids_root), ("text_vec", text_vec), ("cache_dir", cache_dir)] if not v]
    if missing:
        raise SystemExit(
            "Missing required args: "
            + ", ".join(missing)
            + ". Provide them explicitly or pass --config-name <yaml in configs/>."
        )

    os.makedirs(cache_dir, exist_ok=True)

    # Construct a minimal cfg_data object the dataset understands.
    cfg_data = {
        "backend": "ds003825_bids",
        "dataset_impl": "ds003825_bids",
        "eeg_path": bids_root,
        "text_vec_path": text_vec,
        "image_vec_path": image_vec,
        "cache_dir": cache_dir,
        "cache_format": cache_format,
        "subjects": subjects,
        "exclude_subjects": exclude_subjects,
        "exclude_targets": bool(exclude_targets),
        "baseline_correction": bool(baseline_correction),
        "l_freq": float(l_freq),
        "h_freq": float(h_freq),
        "resample_sfreq": float(resample_sfreq),
        "tmin": float(tmin),
        "tmax": float(tmax),
        "n_channels_epoch": int(n_channels_epoch),
        "n_channels_out": int(n_channels_out),
        "n_samples_out": int(n_samples_out),
        "interp_chunk": int(interp_chunk),
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
    print(f"Done. Cache written to: {cache_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
