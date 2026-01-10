import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import os
import json
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from models.clip_models import SpatialMoEEncoder
from dataset import TripletDataset


# =======================
# Config
# =======================
CONFIG_PATH = "configs/triplet_config.yaml"
MODEL_PATH = "temp/best_12.8_change.pth"  # <- 改成你的 ckpt
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = "saliency_outputs_expert_common"
BATCH_SIZE = 64
MAX_BATCHES = 100            # 建议 >= 100 才稳定
NUM_WORKERS = 0

# attribution method: "inputxgrad" | "absgrad" | "signedgrad"
ATTR_METHOD = "inputxgrad"

# common-scale plotting
CMAP_POSITIVE = "Reds"       # 用于 raw attribution（非负）
CMAP_DIFF = "RdBu_r"         # 用于差分（有正有负）

# =======================


def try_load_ch_names(eeg_path: str):
    """尝试从 eeg_path（你 cfg 里指定的文件）读取通道名列表。"""
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
    返回 (ch_names, montage, template_flag).
    若无法获取真实通道名，则使用 biosemi128 模板 montage。
    """
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
            f"读取到 ch_names，但与 biosemi128 模板重合仅 {len(common)}/128。"
            "将使用模板顺序绘制（template=True）。建议从原始数据包找 electrodes/chanloc 获取真实 montage。"
        )
        return std.ch_names, std, True

    return ch_names, std, False


def build_info(ch_names, montage):
    import mne
    info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types="eeg")
    info.set_montage(montage)
    return info


def plot_topomap(values, info, out_path, title, vmin=None, vmax=None, cmap="Reds"):
    import mne
    fig, ax = plt.subplots(figsize=(6, 6))
    mne.viz.plot_topomap(
        values, info,
        axes=ax, contours=0, show=False,
        vmin=vmin, vmax=vmax, cmap=cmap
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def group_by_coords(info, values):
    """
    坐标粗分区：Posterior/Anterior/Left/Right/Midline
    values: (N,) raw attribution
    """
    pos = np.array([ch["loc"][:3] for ch in info["chs"]], dtype=np.float32)  # (N,3)
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
        return float(np.mean(values[mask]))

    return {
        "Posterior": mean(post_mask),
        "Anterior": mean(ant_mask),
        "Left": mean(left_mask),
        "Right": mean(right_mask),
        "Midline": mean(mid_mask),
    }


def plot_region_bar(stats: dict, out_path: str, title: str):
    keys = ["Posterior", "Anterior", "Left", "Right", "Midline"]
    vals = [stats[k] for k in keys]
    plt.figure(figsize=(7.5, 4))
    plt.bar(keys, vals)
    plt.ylabel("Mean attribution (raw)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_model():
    model = SpatialMoEEncoder(n_channels=128, n_samples=512, embedding_dim=512).to(DEVICE)
    return model


def attribution_from_grad(eeg, grad, method="inputxgrad"):
    """
    eeg:  [B, C, T]
    grad: [B, C, T]
    return [C]
    """
    if method == "absgrad":
        sal = grad.abs().mean(dim=(0, 2))
        return sal
    if method == "signedgrad":
        sal = grad.mean(dim=(0, 2))
        return sal
    if method == "inputxgrad":
        sal = (eeg * grad).mean(dim=(0, 2))
        # Input×Grad 可保留符号；这里为了“重要性”默认取 abs
        sal = sal.abs()
        return sal
    raise ValueError(f"Unknown ATTR_METHOD={method}")


@torch.no_grad()
def compute_mean_cos(a, b):
    a = F.normalize(a, p=2, dim=-1)
    b = F.normalize(b, p=2, dim=-1)
    return float((a * b).sum(dim=-1).mean().item())


def compute_expert_saliency_both(model, loader, device, max_batches=50, attr_method="inputxgrad"):
    """
    同一批数据上同时计算：
    - Visual:  cos(emb_vis, img_vec)
    - Semantic:cos(emb_sem, txt_vec)

    并输出：
    - saliency_visual_raw (C,)
    - saliency_semantic_raw (C,)
    - metrics: cos(emb_vis, emb_sem), cos(img_vec, txt_vec), corr(sal_v, sal_s) etc.
    """
    model.eval()

    cache = {"emb_vis": None, "emb_sem": None}

    def hook_vis(_module, _inp, out):
        cache["emb_vis"] = out

    def hook_sem(_module, _inp, out):
        cache["emb_sem"] = out

    h1 = model.expert_visual_head.register_forward_hook(hook_vis)
    h2 = model.expert_semantic_head.register_forward_hook(hook_sem)

    sal_v_list = []
    sal_s_list = []

    cos_emb_vs_list = []
    cos_it_list = []

    n_used = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break

        eeg, image_vec, text_vec = batch
        eeg = eeg.to(device)
        image_vec = image_vec.to(device)
        text_vec = text_vec.to(device)

        # 统计目标相似度（不需要梯度）
        cos_it_list.append(compute_mean_cos(image_vec, text_vec))

        # 需要对 eeg 求梯度
        eeg.requires_grad_(True)

        # forward（触发 hook）
        _img_emb, _txt_emb, _weights = model(eeg)

        emb_vis = cache["emb_vis"]
        emb_sem = cache["emb_sem"]
        if emb_vis is None or emb_sem is None:
            raise RuntimeError("没有捕获到 emb_vis/emb_sem，请确认模型中存在 expert_visual_head/expert_semantic_head。")

        # 统计两个 expert 输出的相似度
        cos_emb_vs_list.append(compute_mean_cos(emb_vis, emb_sem))

        # -------- Visual target: cos(emb_vis, image_vec) --------
        v = F.normalize(emb_vis, p=2, dim=-1)
        img = F.normalize(image_vec, p=2, dim=-1)
        score_v = (v * img).sum(dim=-1).mean()

        model.zero_grad(set_to_none=True)
        if eeg.grad is not None:
            eeg.grad.zero_()
        score_v.backward(retain_graph=True)

        grad_v = eeg.grad.detach()
        sal_v = attribution_from_grad(eeg.detach(), grad_v, method=attr_method)  # [C]
        sal_v_list.append(sal_v.cpu().numpy())

        # -------- Semantic target: cos(emb_sem, text_vec) --------
        if eeg.grad is not None:
            eeg.grad.zero_()

        s = F.normalize(emb_sem, p=2, dim=-1)
        txt = F.normalize(text_vec, p=2, dim=-1)
        score_s = (s * txt).sum(dim=-1).mean()

        model.zero_grad(set_to_none=True)
        score_s.backward()

        grad_s = eeg.grad.detach()
        sal_s = attribution_from_grad(eeg.detach(), grad_s, method=attr_method)  # [C]
        sal_s_list.append(sal_s.cpu().numpy())

        n_used += 1

    h1.remove()
    h2.remove()

    if n_used == 0:
        raise RuntimeError("没有取到 batch，请检查 DataLoader。")

    sal_v_raw = np.mean(np.stack(sal_v_list, axis=0), axis=0).astype(np.float32)
    sal_s_raw = np.mean(np.stack(sal_s_list, axis=0), axis=0).astype(np.float32)

    metrics = {
        "n_batches": n_used,
        "attr_method": attr_method,
        "mean_cos_emb_vis_emb_sem": float(np.mean(cos_emb_vs_list)),
        "mean_cos_img_vec_txt_vec": float(np.mean(cos_it_list)),
        "corr_sal_v_sal_s": float(np.corrcoef(sal_v_raw, sal_s_raw)[0, 1]),
        "mean_abs_diff_sal": float(np.mean(np.abs(sal_v_raw - sal_s_raw))),
        "max_abs_diff_sal": float(np.max(np.abs(sal_v_raw - sal_s_raw))),
    }
    return sal_v_raw, sal_s_raw, metrics


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # load cfg + dataset
    cfg = OmegaConf.load(CONFIG_PATH)
    cfg.data.root = os.getcwd()

    dataset = TripletDataset(cfg.data, mode="val", split_index=0)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,          # 关键：固定顺序，减少噪声
        num_workers=NUM_WORKERS
    )

    # model
    model = build_model()
    print(f"Loading model: {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()

    # compute saliency
    print("Computing expert saliency (visual & semantic) ...")
    sal_v_raw, sal_s_raw, metrics = compute_expert_saliency_both(
        model, loader, DEVICE,
        max_batches=MAX_BATCHES,
        attr_method=ATTR_METHOD
    )

    # save raw arrays
    v_path = os.path.join(OUT_DIR, "saliency_expert_visual_raw.npy")
    s_path = os.path.join(OUT_DIR, "saliency_expert_semantic_raw.npy")
    np.save(v_path, sal_v_raw)
    np.save(s_path, sal_s_raw)

    # save metrics
    metrics_path = os.path.join(OUT_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # montage/info
    ch_names, montage, is_template = get_montage_and_names(cfg.data.eeg_path)
    info = build_info(ch_names, montage)

    # ---------- common scale for visual & semantic ----------
    vmin = float(min(sal_v_raw.min(), sal_s_raw.min()))
    vmax = float(max(sal_v_raw.max(), sal_s_raw.max()))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    plot_topomap(
        sal_v_raw, info,
        os.path.join(OUT_DIR, "expert_topomap_visual_common.png"),
        title=f"Visual Expert Attribution (cos(emb_vis,img_vec), common scale, template={is_template})",
        vmin=vmin, vmax=vmax, cmap=CMAP_POSITIVE
    )
    plot_topomap(
        sal_s_raw, info,
        os.path.join(OUT_DIR, "expert_topomap_semantic_common.png"),
        title=f"Semantic Expert Attribution (cos(emb_sem,txt_vec), common scale, template={is_template})",
        vmin=vmin, vmax=vmax, cmap=CMAP_POSITIVE
    )

    # region bars (raw)
    v_stats = group_by_coords(info, sal_v_raw)
    s_stats = group_by_coords(info, sal_s_raw)
    plot_region_bar(
        v_stats,
        os.path.join(OUT_DIR, "expert_region_bar_visual_coords_raw.png"),
        title="Visual Expert Attribution by Region (Coord-based, raw)"
    )
    plot_region_bar(
        s_stats,
        os.path.join(OUT_DIR, "expert_region_bar_semantic_coords_raw.png"),
        title="Semantic Expert Attribution by Region (Coord-based, raw)"
    )

    # ---------- difference map (visual - semantic), common symmetric scale ----------
    diff = (sal_v_raw - sal_s_raw).astype(np.float32)
    dmax = float(np.max(np.abs(diff)) + 1e-6)

    plot_topomap(
        diff, info,
        os.path.join(OUT_DIR, "expert_topomap_visual_minus_semantic_common.png"),
        title=f"Expert Attribution Difference (Visual - Semantic, common symmetric, template={is_template})",
        vmin=-dmax, vmax=dmax, cmap=CMAP_DIFF
    )

    print("Saved outputs to:", OUT_DIR)
    print("Key files:")
    print("  - expert_topomap_visual_common.png")
    print("  - expert_topomap_semantic_common.png")
    print("  - expert_topomap_visual_minus_semantic_common.png")
    print("  - metrics.json")
    print("  - saliency_expert_visual_raw.npy / saliency_expert_semantic_raw.npy")


if __name__ == "__main__":
    main()
