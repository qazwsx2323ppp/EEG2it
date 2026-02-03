import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def build_concept_table(bids_root: Path) -> pd.DataFrame:
    # Prefer a subject that likely has full columns
    for sub in sorted([p.name for p in bids_root.iterdir() if p.is_dir() and p.name.startswith("sub-")]):
        events = bids_root / sub / "eeg" / f"{sub}_task-rsvp_events.tsv"
        if not events.exists():
            continue
        df = pd.read_csv(events, sep="\t")
        if "objectnumber" not in df.columns or "object" not in df.columns:
            continue
        df = df[df["objectnumber"].astype(int) >= 0][["objectnumber", "object"]]
        df["objectnumber"] = df["objectnumber"].astype(int)
        df["object"] = df["object"].astype(str)
        # dedupe
        df = df.drop_duplicates("objectnumber").sort_values("objectnumber")
        return df

    raise RuntimeError("Could not find events.tsv with columns 'objectnumber' and 'object'.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Embed ds003825 THINGS concept names into CLIP text space.")
    ap.add_argument("--bids-root", type=str, required=True, help="BIDS root (e.g. C:\\ppp\\CODE\\EEG\\data\\ds003825)")
    ap.add_argument("--out-text", type=str, required=True, help="Output .npy for text vectors (shape [1854, D])")
    ap.add_argument("--out-image", type=str, default="", help="Optional output .npy for image vectors (shape [1854, D])")
    ap.add_argument(
        "--image-same-as-text",
        action="store_true",
        help="If set, write image vectors identical to text vectors (useful when stimuli images are unavailable).",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="HuggingFace model id or local path.",
    )
    ap.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow transformers to download weights (set this if weights are not cached).",
    )
    ap.add_argument("--prompt", type=str, default="", help="(Deprecated) Same as --prompt-text.")
    ap.add_argument(
        "--prompt-text",
        type=str,
        default="a {}",
        help="Text prompt template for the *text* target; '{}' will be replaced by the concept string.",
    )
    ap.add_argument(
        "--prompt-image",
        type=str,
        default="a photo of a {}",
        help="Text prompt template for the *image-like* target; '{}' will be replaced by the concept string.",
    )
    args = ap.parse_args()

    bids_root = Path(args.bids_root)
    out_text = Path(args.out_text)
    out_image = Path(args.out_image) if args.out_image else None

    prompt_text = args.prompt_text
    if args.prompt and not args.prompt_text:
        prompt_text = args.prompt
    prompt_image = args.prompt_image

    concept_df = build_concept_table(bids_root)
    if concept_df["objectnumber"].min() != 0:
        raise RuntimeError("Concept ids do not start at 0; unexpected ds003825 format.")
    n_concepts = int(concept_df["objectnumber"].max()) + 1

    try:
        from transformers import CLIPModel, CLIPProcessor
    except Exception as e:
        raise RuntimeError("Missing transformers CLIP. Install transformers, or embed vectors elsewhere.") from e

    local_files_only = not args.allow_download
    try:
        model = CLIPModel.from_pretrained(args.model, local_files_only=local_files_only)
        proc = CLIPProcessor.from_pretrained(args.model, local_files_only=local_files_only)
    except Exception as e:
        msg = (
            f"Failed to load {args.model} (local_files_only={local_files_only}).\n"
            "If you don't have weights cached locally, rerun with --allow-download (requires internet)."
        )
        raise RuntimeError(msg) from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    def embed_texts(text_list: list[str]) -> np.ndarray:
        vecs = []
        batch = 128
        with torch.no_grad():
            for i in range(0, len(text_list), batch):
                chunk = text_list[i : i + batch]
                inputs = proc(text=chunk, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                feats = model.get_text_features(**inputs)
                feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
                vecs.append(feats.cpu().numpy().astype(np.float32))
        return np.concatenate(vecs, axis=0)

    names = [name.replace("_", " ") for name in concept_df["object"].tolist()]
    texts_text = [prompt_text.format(n) for n in names]
    text_vecs = embed_texts(texts_text)
    if text_vecs.shape[0] != n_concepts:
        raise RuntimeError(f"Unexpected concept count: got {text_vecs.shape[0]}, expected {n_concepts}")

    out_text.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_text, text_vecs)

    image_vecs = None
    if out_image is not None:
        if args.image_same_as_text:
            image_vecs = text_vecs
        else:
            texts_image = [prompt_image.format(n) for n in names]
            image_vecs = embed_texts(texts_image)
            if image_vecs.shape[0] != n_concepts:
                raise RuntimeError(f"Unexpected concept count for image-like vectors: got {image_vecs.shape[0]}, expected {n_concepts}")
        out_image.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_image, image_vecs)

    print(f"Wrote text vectors: {out_text} shape={text_vecs.shape}")
    if out_image is not None:
        same = " (same as text)" if args.image_same_as_text else ""
        print(f"Wrote image vectors{same}: {out_image} shape={image_vecs.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
