import os
import sys
import torch
import numpy as np
from omegaconf import OmegaConf

def inspect_eeg_pth(eeg_path, max_items=3):
    print(f"\n=== Inspect EEG .pth ===\nFile: {eeg_path}")
    data = torch.load(eeg_path)
    if isinstance(data, dict):
        print(f"Top-level keys: {list(data.keys())}")
        if 'dataset' in data:
            items = data['dataset']
            print(f"'dataset' type: {type(items)} | length: {len(items)}")
            for i in range(min(max_items, len(items))):
                item = items[i]
                print(f"\n[Item {i}] keys: {list(item.keys())}")
                eeg = item.get('eeg', None)
                img_idx = item.get('image', None)
                caption = item.get('caption', None) or item.get('text', None)
                if torch.is_tensor(eeg):
                    print(f"  eeg: shape={tuple(eeg.shape)}, dtype={eeg.dtype}")
                    # 通常为 (Channels=128, Time≈500~600)，在 dataset.py 中会先裁剪到 440，再插值到 512
                    print(f"  eeg stats: mean={eeg.mean().item():.4f}, std={eeg.std().item():.4f}, min={eeg.min().item():.4f}, max={eeg.max().item():.4f}")
                else:
                    print(f"  eeg: {type(eeg)}")
                print(f"  image index: {img_idx}")
                print(f"  caption/text: {caption}")
        else:
            print("Warning: dict without 'dataset' key. Full dict printed below:")
            print(data)
    else:
        print(f"Unexpected type at top-level: {type(data)}")
        print(data)

def inspect_vectors(image_vec_path, text_vec_path, max_items=3):
    print(f"\n=== Inspect Aligned Vectors (.npy) ===")
    img_vecs = np.load(image_vec_path)
    txt_vecs = np.load(text_vec_path)
    print(f"image_vectors: shape={img_vecs.shape}, dtype={img_vecs.dtype}")
    print(f"text_vectors:  shape={txt_vecs.shape}, dtype={txt_vecs.dtype}")
    # 通常都应为 (N, 512)，与编码器输出维度一致
    for name, arr in [('image', img_vecs), ('text', txt_vecs)]:
        vec = torch.from_numpy(arr).float()
        print(f"  {name} sample stats: mean={vec.mean().item():.4f}, std={vec.std().item():.4f}, norm(avg)={vec.norm(dim=-1).mean().item():.4f}")
    n_avail = min(len(img_vecs), len(txt_vecs))
    print(f"usable vector count (min of two): {n_avail}")

def inspect_splits(splits_path, split_index=0):
    print(f"\n=== Inspect Splits .pth ===\nFile: {splits_path}")
    splits = torch.load(splits_path)
    if isinstance(splits, dict):
        print(f"Top-level keys: {list(splits.keys())}")
        if 'splits' in splits:
            arr = splits['splits']
            print(f"'splits' type: {type(arr)} | length: {len(arr)}")
            if split_index < len(arr):
                split = arr[split_index]
                print(f"\nSplit[{split_index}] modes: {list(split.keys())}")
                for mode in ['train', 'val', 'test']:
                    if mode in split:
                        idxs = split[mode]
                        print(f"  {mode}: count={len(idxs)} | sample head={idxs[:10]}")
                    else:
                        print(f"  {mode}: missing")
            else:
                print(f"Split index {split_index} out of range.")
        else:
            print("Warning: dict without 'splits' key. Full dict printed below:")
            print(splits)
    else:
        print(f"Unexpected splits type: {type(splits)}")
        print(splits)

def cross_check_indices(eeg_path, image_vec_path, text_vec_path, splits_path, split_index=0, max_checks=5):
    print("\n=== Cross-Check Index Mapping ===")
    data = torch.load(eeg_path)
    items = data['dataset'] if isinstance(data, dict) and 'dataset' in data else []
    img_vecs = np.load(image_vec_path)
    txt_vecs = np.load(text_vec_path)
    n_avail = min(len(img_vecs), len(txt_vecs))
    splits = torch.load(splits_path)
    split = splits['splits'][split_index] if 'splits' in splits else {}
    train_ids = split.get('train', [])
    print(f"Items: {len(items)} | usable vectors: {n_avail} | train indices: {len(train_ids)}")
    checked = 0
    for eeg_idx in train_ids:
        if eeg_idx >= len(items):
            print(f"  [Skip] EEG idx {eeg_idx} out of range ({len(items)})")
            continue
        image_idx = items[eeg_idx].get('image', None)
        if image_idx is None:
            print(f"  [Skip] EEG idx {eeg_idx} missing 'image' key")
            continue
        ok = image_idx < n_avail
        print(f"  EEG {eeg_idx} -> image {image_idx} | valid:{ok}")
        checked += 1
        if checked >= max_checks:
            break
    print("Note: dataset.py 会过滤掉 image_idx >= usable_vector_count 的条目，确保索引有效")

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg_path = os.path.join(project_root, "configs", "triplet_config.yaml")
    cfg = OmegaConf.load(cfg_path)
    # Hydra 未运行时，手动解析 root
    cfg.data.root = project_root

    eeg_path = os.path.abspath(cfg.data.eeg_path)
    image_vec_path = os.path.abspath(cfg.data.image_vec_path)
    text_vec_path = os.path.abspath(cfg.data.text_vec_path)
    splits_path = os.path.abspath(cfg.data.splits_path)
    split_index = cfg.data.get("split_index", 0)

    print("Config paths:")
    print("  eeg_path       :", eeg_path)
    print("  image_vec_path :", image_vec_path)
    print("  text_vec_path  :", text_vec_path)
    print("  splits_path    :", splits_path)
    print("  split_index    :", split_index)

    inspect_eeg_pth(eeg_path)
    inspect_vectors(image_vec_path, text_vec_path)
    inspect_splits(splits_path, split_index=split_index)
    cross_check_indices(eeg_path, image_vec_path, text_vec_path, splits_path, split_index=split_index)

if __name__ == "__main__":
    main()