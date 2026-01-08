import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import os
import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from models.clip_models import SpatialMoEEncoder
from dataset import TripletDataset

# === 配置 ===
CONFIG_PATH = "configs/triplet_config.yaml"
MODEL_PATH = "temp/best_12.8_change.pth"  # 替换为你的模型路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 64
MAX_BATCHES_FOR_SALIENCY = 30  # 多 batch 平均，越大越稳定但越慢
OUT_DIR = "saliency_outputs"   # 输出目录
# ============


def try_load_ch_names(eeg_path: str):
    """
    尝试从 eeg .pt 中读取通道名（如果你在预处理时保存过）。
    常见 key: ch_names / channels / channel_names
    """
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
    """
    返回 (ch_names, montage, is_template)

    - 若能从数据里读到 ch_names，则尝试与标准 montage 对齐
    - 若读不到，则回退到 biosemi128 的模板 montage，并假设通道顺序一致
    """
    try:
        import mne
    except Exception as e:
        raise RuntimeError(
            "需要安装 mne 才能画 topomap，请先 pip install mne。"
        ) from e

    ch_names = try_load_ch_names(eeg_path)

    # 兜底：标准模板（你数据通道顺序未必一致，建议后续尽量找到真实 ch_names）
    std = mne.channels.make_standard_montage("biosemi128")

    if ch_names is None:
        return std.ch_names, std, True

    # 如果读到了 ch_names，尽量匹配
    # 若匹配率很低，仍用模板并提醒
    common = [ch for ch in ch_names if ch in std.ch_names]
    if len(common) < 100:
        warnings.warn(
            f"读取到了 ch_names，但与 biosemi128 模板的重合通道数仅 {len(common)}/128，"
            "将回退使用模板顺序。建议你从原始数据包中找 electrodes/chanloc 文件以获得真实 montage。"
        )
        return std.ch_names, std, True

    # 这里：我们仍返回原始 ch_names（用于与你数据通道顺序保持一致）
    # montage 用模板 std（至少能画出空间分布）
    return ch_names, std, False


def rough_region_group(ch_name: str) -> str:
    """
    Palazzo Fig.4 风格的粗分区：按通道名前缀粗略映射到脑区。
    注意：这是 rough matching，不是 source localization。
    """
    name = ch_name.upper()

    if name.startswith("FP") or name.startswith("AF") or (name.startswith("F") and not name.startswith("FT")):
        return "Frontal"
    if name.startswith("T") or name.startswith("FT") or name.startswith("TP"):
        return "Temporal"
    if name.startswith("C"):
        return "Central"
    if name.startswith("P") or name.startswith("CP"):
        return "Parietal"
    if name.startswith("O") or name.startswith("PO"):
        return "Occipital"
    return "Other"


@torch.no_grad()
def _forward_get_weights(model, eeg_batch):
    # 这里不使用 no_grad，因为要做梯度；此函数仅保留结构示意
    raise NotImplementedError


def compute_saliency_over_loader(model, loader, device, max_batches=20):
    """
    对多个 batch 求平均的 saliency（更稳定）。
    saliency 定义：mean_{batch,t} | d target / d eeg |
    输出 shape: [128]
    """
    model.eval()
    saliencies = []
    n_used = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break

        eeg, _, _ = batch
        eeg = eeg.to(device)
        eeg.requires_grad_(True)

        # forward
        _, _, weights = model(eeg)

        # Visual router weight
        target = weights["w_vis_img"]  # [B,1] or [B]
        loss = target.sum()

        # backward
        model.zero_grad(set_to_none=True)
        if eeg.grad is not None:
            eeg.grad.zero_()
        loss.backward()

        # grad: [B, 128, 512] -> importance per channel
        sal = eeg.grad.abs().mean(dim=(0, 2))  # [128]
        saliencies.append(sal.detach().cpu().numpy())
        n_used += 1

    if n_used == 0:
        raise RuntimeError("DataLoader 没有取到任何 batch，请检查数据集与 split。")

    saliency = np.mean(np.stack(saliencies, axis=0), axis=0)
    return saliency


def normalize_01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def plot_index_bar(importance_01, out_path, title, expected_span=None):
    plt.figure(figsize=(15, 5))
    plt.bar(range(len(importance_01)), importance_01, color="blue", alpha=0.7, label="Attribution (normalized)")
    if expected_span is not None:
        a, b, label, color = expected_span
        plt.axvspan(a, b, color=color, alpha=0.2, label=label)
    plt.title(title)
    plt.xlabel("Channel Index (0-127)")
    plt.ylabel("Importance (0-1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_topomap(importance_01, ch_names, montage, out_path, title):
    import mne

    info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types="eeg")
    info.set_montage(montage)

    fig, ax = plt.subplots(figsize=(6, 6))
    mne.viz.plot_topomap(importance_01, info, axes=ax, contours=0, show=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_region_bar(importance_01, ch_names, out_path, title):
    from collections import defaultdict

    groups = defaultdict(list)
    for s, name in zip(importance_01, ch_names):
        groups[rough_region_group(name)].append(float(s))

    keys = ["Occipital", "Parietal", "Temporal", "Central", "Frontal", "Other"]
    vals = [np.mean(groups[k]) if len(groups[k]) else 0.0 for k in keys]

    plt.figure(figsize=(8, 4))
    plt.bar(keys, vals)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Mean normalized importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_model(expected_visual_indices, expected_semantic_indices):
    """
    与你当前脚本兼容：如果 SpatialMoEEncoder 仍要求 visual_indices/semantic_indices，就传占位。
    """
    try:
        model = SpatialMoEEncoder(
            n_channels=128, n_samples=512,
            embedding_dim=512
        ).to(DEVICE)
    except TypeError:
        print(">>> 检测到模型仍需要索引参数，传入占位数据...")
        model = SpatialMoEEncoder(
            n_channels=128, n_samples=512,
            visual_indices=expected_visual_indices,
            semantic_indices=expected_semantic_indices,
            embedding_dim=512
        ).to(DEVICE)
    return model


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cfg = OmegaConf.load(CONFIG_PATH)
    cfg.data.root = os.getcwd()

    # 只为“参考阴影”保留，不再作为论证依据
    expected_visual_indices = list(range(64, 128))
    expected_semantic_indices = list(range(0, 64))

    # DataLoader（使用 val，避免训练增强）
    dataset = TripletDataset(cfg.data, mode="val", split_index=0)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Model
    model = build_model(expected_visual_indices, expected_semantic_indices)
    print(f"Loading model from {MODEL_PATH} ...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()

    # Compute saliency
    print("Computing visual-router saliency over multiple batches ...")
    saliency = compute_saliency_over_loader(model, loader, DEVICE, max_batches=MAX_BATCHES_FOR_SALIENCY)
    saliency_01 = normalize_01(saliency)

    # Save raw
    npy_path = os.path.join(OUT_DIR, "saliency_visual_router.npy")
    np.save(npy_path, saliency_01)
    print(f"Saved saliency vector: {npy_path}")

    # Plot index bar (保留你原来的图)
    bar_path = os.path.join(OUT_DIR, "router_saliency_visual_bar.png")
    plot_index_bar(
        saliency_01,
        bar_path,
        title="Visual Expert Router Attribution (Index View)",
        expected_span=(64, 128, "Expected Visual Region (Occipital) [index heuristic]", "yellow"),
    )
    print(f"Saved index bar: {bar_path}")

    # Montage + topomap
    ch_names, montage, is_template = get_montage_and_names(cfg.data.eeg_path)

    topo_path = os.path.join(OUT_DIR, "router_topomap_visual.png")
    plot_topomap(
        saliency_01,
        ch_names=ch_names,
        montage=montage,
        out_path=topo_path,
        title=f"Visual Expert Router Attribution (Scalp Topomap, template={is_template})",
    )
    print(f"Saved topomap: {topo_path}")

    # Region bar (rough matching)
    region_path = os.path.join(OUT_DIR, "router_region_bar_visual.png")
    plot_region_bar(
        saliency_01,
        ch_names=ch_names,
        out_path=region_path,
        title="Visual Expert Router Attribution by Region (Rough Matching)",
    )
    print(f"Saved region bar: {region_path}")

    top_10 = saliency_01.argsort()[-10:][::-1]
    print(f"Top-10 channel indices (visual router): {top_10}")


if __name__ == "__main__":
    main()
