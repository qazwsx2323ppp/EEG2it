import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import os
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from models.clip_models import SpatialMoEEncoder
from dataset import TripletDataset

# ====== 配置 ======
CONFIG_PATH = "configs/triplet_config.yaml"
MODEL_PATH = "temp/best_12.8_change.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 64
MAX_BATCHES = 30
OUT_DIR = "saliency_outputs_expert"
# =================


def try_load_ch_names(eeg_path: str):
    try:
        obj = torch.load(eeg_path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    for k in ["ch_names", "channels", "channel_names"]:
        if k in obj and isinstance(obj[k], (list, tuple)) and len(obj[k]) == 128:
            return list(obj[k])
    return None


def get_montage_and_names(eeg_path: str):
    try:
        import mne
    except Exception as e:
        raise RuntimeError("需要安装 mne 才能画 topomap：pip install mne") from e

    std = mne.channels.make_standard_montage("biosemi128")
    ch_names = try_load_ch_names(eeg_path)
    if ch_names is None:
        return std.ch_names, std, True

    common = [ch for ch in ch_names if ch in std.ch_names]
    if len(common) < 100:
        warnings.warn(
            f"读取到 ch_names 但与模板重合仅 {len(common)}/128，将使用模板顺序。"
            "建议从原始数据包找 electrodes/chanloc 文件以获得真实 montage。"
        )
        return std.ch_names, std, True

    return ch_names, std, False


def normalize_01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def plot_topomap(values_01, ch_names, montage, out_path, title):
    import mne

    info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types="eeg")
    info.set_montage(montage)

    fig, ax = plt.subplots(figsize=(6, 6))
    mne.viz.plot_topomap(values_01, info, axes=ax, contours=0, show=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def group_by_coords(info, values_01):
    pos = np.array([ch["loc"][:3] for ch in info["chs"]], dtype=np.float32)
    x = pos[:, 0]
    y = pos[:, 1]

    mid_mask = np.abs(x) < (0.15 * (np.max(np.abs(x)) + 1e-6))
    ant_mask = y > np.median(y)
    post_mask = y <= np.median(y)
    left_mask = x < 0
    right_mask = x >= 0

    def mean(mask):
        if mask.sum() == 0:
            return 0.0
        return float(values_01[mask].mean())

    return {
        "Posterior": mean(post_mask),
        "Anterior": mean(ant_mask),
        "Left": mean(left_mask),
        "Right": mean(right_mask),
        "Midline": mean(mid_mask),
    }


def plot_region_bar_from_coords(ch_names, montage, values_01, out_path, title):
    import mne

    info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types="eeg")
    info.set_montage(montage)

    stats = group_by_coords(info, values_01)
    keys = ["Posterior", "Anterior", "Left", "Right", "Midline"]
    vals = [stats[k] for k in keys]

    plt.figure(figsize=(7.5, 4))
    plt.bar(keys, vals)
    plt.ylabel("Mean normalized importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_model():
    model = SpatialMoEEncoder(n_channels=128, n_samples=512, embedding_dim=512).to(DEVICE)
    return model


def compute_expert_saliency_semantic(model, loader, device, max_batches=20):
    """
    目标：cos(emb_sem, text_vector)
    attribution：mean_{batch,t} | d target / d eeg |
    """
    model.eval()

    cache = {"emb_sem": None}

    def hook_sem(module, inp, out):
        cache["emb_sem"] = out  # [B, 512]

    h = model.expert_semantic_head.register_forward_hook(hook_sem)

    saliencies = []
    n_used = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break

        eeg, _, text_vec = batch
        eeg = eeg.to(device)
        text_vec = text_vec.to(device)

        eeg.requires_grad_(True)

        _img_emb, _txt_emb, _weights = model(eeg)

        emb_sem = cache["emb_sem"]
        if emb_sem is None:
            raise RuntimeError("没有捕获到 emb_sem，请确认模型中存在 expert_semantic_head。")

        emb_sem_n = F.normalize(emb_sem, p=2, dim=-1)
        text_vec_n = F.normalize(text_vec, p=2, dim=-1)

        score = (emb_sem_n * text_vec_n).sum(dim=-1)  # [B]
        loss = score.mean()

        model.zero_grad(set_to_none=True)
        if eeg.grad is not None:
            eeg.grad.zero_()
        loss.backward()

        sal = eeg.grad.abs().mean(dim=(0, 2))  # [128]
        saliencies.append(sal.detach().cpu().numpy())
        n_used += 1

    h.remove()

    if n_used == 0:
        raise RuntimeError("没有取到 batch，请检查 DataLoader。")

    saliency = np.mean(np.stack(saliencies, axis=0), axis=0)
    return saliency


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cfg = OmegaConf.load(CONFIG_PATH)
    cfg.data.root = os.getcwd()

    dataset = TripletDataset(cfg.data, mode="val", split_index=0)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = build_model()
    print(f"Loading model: {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()

    print("Computing SEMANTIC EXPERT saliency: cos(emb_sem, text_vec) ...")
    sal = compute_expert_saliency_semantic(model, loader, DEVICE, max_batches=MAX_BATCHES)
    sal_01 = normalize_01(sal)

    np.save(os.path.join(OUT_DIR, "saliency_expert_semantic.npy"), sal_01)

    ch_names, montage, is_template = get_montage_and_names(cfg.data.eeg_path)

    plot_topomap(
        sal_01, ch_names, montage,
        os.path.join(OUT_DIR, "expert_topomap_semantic.png"),
        title=f"Semantic Expert Attribution (cos(emb_sem, txt_vec), template={is_template})"
    )

    plot_region_bar_from_coords(
        ch_names, montage, sal_01,
        os.path.join(OUT_DIR, "expert_region_bar_semantic_coords.png"),
        title="Semantic Expert Attribution by Region (Coord-based Rough Matching)"
    )

    # 自动做差异图（Visual - Semantic），如果 visual npy 已存在
    vis_npy = os.path.join(OUT_DIR, "saliency_expert_visual.npy")
    if os.path.exists(vis_npy):
        vis = np.load(vis_npy)
        diff = vis - sal_01
        plot_topomap(
            diff, ch_names, montage,
            os.path.join(OUT_DIR, "expert_topomap_visual_minus_semantic.png"),
            title=f"Expert Attribution Difference (Visual - Semantic, template={is_template})"
        )

    print("Saved semantic expert maps to:", OUT_DIR)


if __name__ == "__main__":
    main()
