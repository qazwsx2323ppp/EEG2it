import argparse
import os
from typing import Any

import torch


def _summarize(obj: Any, max_items: int = 5) -> str:
    if isinstance(obj, dict):
        keys = list(obj.keys())
        return f"dict(keys={keys[:max_items]}{'...' if len(keys) > max_items else ''})"
    if isinstance(obj, (list, tuple)):
        return f"{type(obj).__name__}(len={len(obj)})"
    return f"{type(obj).__name__}"


def _safe_get_image_name(image_data: Any, image_id: int) -> str:
    if image_data is None:
        return ""
    # dict with list under common keys
    if isinstance(image_data, dict):
        for key in ("images", "data", "image_data", "items"):
            if key in image_data:
                return _safe_get_image_name(image_data[key], image_id)
        # dict mapping id->name
        if image_id in image_data:
            return str(image_data[image_id])
        if str(image_id) in image_data:
            return str(image_data[str(image_id)])
        return ""
    # list of dicts
    if isinstance(image_data, list):
        if image_id < 0 or image_id >= len(image_data):
            return ""
        item = image_data[image_id]
        if isinstance(item, dict):
            for k in ("file_name", "filename", "path", "image_path", "image"):
                if k in item:
                    return str(item[k])
            return str(item)
        if isinstance(item, str):
            return item
    return ""

def _build_image_list(image_root: str, exts: tuple[str, ...]):
    if not image_root or not os.path.isdir(image_root):
        return []
    image_root = os.path.abspath(image_root)
    rel_paths: list[str] = []
    for cls in sorted(os.listdir(image_root)):
        cls_dir = os.path.join(image_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = []
        for fn in os.listdir(cls_dir):
            if fn.lower().endswith(exts):
                files.append(fn)
        for fn in sorted(files):
            rel_paths.append(os.path.join(cls, fn))
    return rel_paths

def _resolve_eeg_image_name(eeg_name: str, image_root: str, exts: tuple[str, ...]) -> str:
    name = str(eeg_name).strip()
    if not name:
        return ""
    if "/" in name or "\\" in name or "." in os.path.basename(name):
        return name
    synset = name.split("_")[0] if "_" in name else ""
    if image_root and synset:
        base_dir = os.path.join(image_root, synset)
        if os.path.isdir(base_dir):
            for ext in exts:
                cand = os.path.join(synset, f"{name}{ext}")
                if os.path.isfile(os.path.join(image_root, cand)):
                    return cand
            for fn in os.listdir(base_dir):
                if fn.startswith(name + "."):
                    return os.path.join(synset, fn)
    return name


def main():
    ap = argparse.ArgumentParser(description="Inspect EEG/image mapping in .pth files.")
    ap.add_argument("--eeg_pth", required=True, help="Path to eeg_*.pth")
    ap.add_argument("--splits_pth", required=True, help="Path to block_splits_by_image_all.pth")
    ap.add_argument("--image_data_pth", default="", help="Path to image_data.pth (optional)")
    ap.add_argument("--image_root", default="", help="Root folder of images (class subfolders)")
    ap.add_argument("--image_exts", default=".jpg,.jpeg,.png,.bmp,.webp", help="Comma-separated extensions")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--split_index", type=int, default=0)
    ap.add_argument("--num", type=int, default=10, help="Number of samples to show")
    args = ap.parse_args()

    eeg = torch.load(args.eeg_pth, map_location="cpu")
    splits = torch.load(args.splits_pth, map_location="cpu")
    image_data = torch.load(args.image_data_pth, map_location="cpu") if args.image_data_pth else None
    exts = tuple(x.strip().lower() for x in args.image_exts.split(",") if x.strip())
    image_list = _build_image_list(args.image_root, exts)

    print("=== EEG .pth ===")
    print("type:", _summarize(eeg))
    if isinstance(eeg, dict):
        print("keys:", list(eeg.keys()))
    dataset = eeg["dataset"] if isinstance(eeg, dict) and "dataset" in eeg else None
    eeg_images = eeg.get("images") if isinstance(eeg, dict) else None
    if isinstance(eeg_images, list) and eeg_images:
        print("eeg_images_len:", len(eeg_images))
        print("eeg_images_head:", eeg_images[:5])
    if dataset is None:
        print("No 'dataset' key found in EEG pth.")
        return
    print("dataset_len:", len(dataset))
    print("dataset_item0_type:", _summarize(dataset[0]))
    if isinstance(dataset[0], dict):
        print("dataset_item0_keys:", list(dataset[0].keys()))

    print("\n=== Splits .pth ===")
    print("type:", _summarize(splits))
    if isinstance(splits, dict):
        print("keys:", list(splits.keys()))
    splits_list = splits.get("splits") if isinstance(splits, dict) else None
    if not splits_list:
        print("No 'splits' key found in splits pth.")
        return
    print("num_splits:", len(splits_list))
    split_obj = splits_list[args.split_index]
    print(f"split[{args.split_index}] keys:", list(split_obj.keys()))
    for k in ("train", "val", "test"):
        if k in split_obj:
            print(f"{k}_len:", len(split_obj[k]))

    print("\n=== Image data .pth ===")
    if image_data is None:
        print("No image_data provided.")
    else:
        print("type:", _summarize(image_data))
        if isinstance(image_data, dict):
            print("keys:", list(image_data.keys())[:20])
        if isinstance(image_data, list) and image_data:
            print("image_data_item0_type:", _summarize(image_data[0]))
            if isinstance(image_data[0], dict):
                print("image_data_item0_keys:", list(image_data[0].keys()))

    # Show mapping for first N items in split
    print("\n=== Sample mapping ===")
    split_indices = split_obj[args.split]
    for i, eeg_idx in enumerate(split_indices[: args.num]):
        eeg_item = dataset[int(eeg_idx)]
        image_id = None
        if isinstance(eeg_item, dict):
            image_id = eeg_item.get("image", None)
        name = _safe_get_image_name(image_data, int(image_id)) if image_id is not None else ""
        name2 = ""
        if image_id is not None and image_list:
            if 0 <= int(image_id) < len(image_list):
                name2 = image_list[int(image_id)]
        name3 = ""
        if image_id is not None and isinstance(eeg_images, list) and 0 <= int(image_id) < len(eeg_images):
            name3 = eeg_images[int(image_id)]
        name4 = _resolve_eeg_image_name(name3, args.image_root, exts) if name3 else ""
        print(
            f"[{i}] eeg_idx={int(eeg_idx)} image_id={image_id} "
            f"image_name={name} image_root_name={name2} eeg_images_name={name3} resolved_name={name4}"
        )


if __name__ == "__main__":
    main()
    if image_list:
        print(f"image_root: {args.image_root}")
        print("image_list_len:", len(image_list))
        print("image_list_head:", image_list[:5])
