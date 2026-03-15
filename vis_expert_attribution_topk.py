import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
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
MODEL_PATH = "temp/best_fornow.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = "saliency_outputs_expert_topk"
BATCH_SIZE = 64
MAX_BATCHES = 100
NUM_WORKERS = 0

# attribution method: "inputxgrad" | "absgrad" | "signedgrad"
ATTR_METHOD = "inputxgrad"

# Top-K settings (set TOPK <= 0 to disable)
TOPK = 10
# "binary": keep 1 for top-k channels per sample
# "value": keep attribution values for top-k, zero elsewhere
TOPK_MODE = "binary"

# common-scale plotting
CMAP_POSITIVE = "Reds"
CMAP_DIFF = "RdBu_r"
# =======================


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
    """
    Returns (ch_names, montage, template_flag).
    If real channel names are not found, use biosemi128 montage.
    """
    try:
        import mne
    except Exception as e:
        raise RuntimeError("mne is required for topomap: pip install mne") from e

    std = mne.channels.make_standard_montage("biosemi128")

    ch_names = try_load_ch_names(eeg_path)
    if ch_names is None:
        return std.ch_names, std, True

    common = [ch for ch in ch_names if ch in std.ch_names]
    if len(common) < 100:
        warnings.warn(
            f"Loaded ch_names but overlap with biosemi128 is {len(common)}/128. "
            "Fallback to template montage (template=True)."
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
        return float(np.mean(values[mask]))

    return {
        "Posterior": mean(post_mask),
        "Anterior": mean(ant_mask),
        "Left": mean(left_mask),
        "Right": mean(right_mask),
        "Midline": mean(mid_mask),
    }


def plot_region_bar(stats: dict, out_path: str, title: str, ylabel: str):
    keys = ["Posterior", "Anterior", "Left", "Right", "Midline"]
    vals = [stats[k] for k in keys]
    plt.figure(figsize=(7.5, 4))
    plt.bar(keys, vals)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_model():
    model = SpatialMoEEncoder(n_channels=128, n_samples=512, embedding_dim=512).to(DEVICE)
    return model


def attribution_from_grad_per_sample(eeg, grad, method="inputxgrad"):
    """
    eeg:  [B, C, T]
    grad: [B, C, T]
    return [B, C]
    """
    if method == "absgrad":
        sal = grad.abs().mean(dim=2)
        return sal
    if method == "signedgrad":
        sal = grad.mean(dim=2)
        return sal
    if method == "inputxgrad":
        sal = (eeg * grad).mean(dim=2)
        sal = sal.abs()
        return sal
    raise ValueError(f"Unknown ATTR_METHOD={method}")


def apply_topk(sal_bxc: torch.Tensor, k: int, mode: str):
    """
    sal_bxc: [B, C]
    mode:
      - "binary": keep 1 for top-k channels per sample
      - "value": keep attribution values for top-k, zero elsewhere
    """
    if k is None or k <= 0:
        return sal_bxc
    k = min(k, sal_bxc.shape[1])
    vals, idx = torch.topk(sal_bxc, k=k, dim=1, largest=True, sorted=False)
    if mode == "binary":
        out = torch.zeros_like(sal_bxc)
        out.scatter_(1, idx, 1.0)
        return out
    if mode == "value":
        out = torch.zeros_like(sal_bxc)
        out.scatter_(1, idx, vals)
        return out
    raise ValueError(f"Unknown TOPK_MODE={mode}")


@torch.no_grad()
def compute_mean_cos(a, b):
    a = F.normalize(a, p=2, dim=-1)
    b = F.normalize(b, p=2, dim=-1)
    return float((a * b).sum(dim=-1).mean().item())


def compute_expert_saliency_both(model, loader, device, max_batches=50, attr_method="inputxgrad", topk=0, topk_mode="binary"):
    """
    Compute expert saliency with optional top-k selection per sample.
    Returns:
      - saliency_visual_raw (C,)
      - saliency_semantic_raw (C,)
      - metrics dict
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
    n_samples = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break

        eeg, image_vec, text_vec = batch
        eeg = eeg.to(device)
        image_vec = image_vec.to(device)
        text_vec = text_vec.to(device)

        cos_it_list.append(compute_mean_cos(image_vec, text_vec))

        eeg.requires_grad_(True)

        _img_emb, _txt_emb, _weights = model(eeg)

        emb_vis = cache["emb_vis"]
        emb_sem = cache["emb_sem"]
        if emb_vis is None or emb_sem is None:
            raise RuntimeError("Failed to capture emb_vis/emb_sem. Check expert heads.")

        cos_emb_vs_list.append(compute_mean_cos(emb_vis, emb_sem))

        # Visual target: cos(emb_vis, image_vec)
        v = F.normalize(emb_vis, p=2, dim=-1)
        img = F.normalize(image_vec, p=2, dim=-1)
        score_v = (v * img).sum(dim=-1).mean()

        model.zero_grad(set_to_none=True)
        if eeg.grad is not None:
            eeg.grad.zero_()
        score_v.backward(retain_graph=True)

        grad_v = eeg.grad.detach()
        sal_v = attribution_from_grad_per_sample(eeg.detach(), grad_v, method=attr_method)  # [B,C]
        sal_v = apply_topk(sal_v, topk, topk_mode)
        sal_v_list.append(sal_v.cpu().numpy())

        # Semantic target: cos(emb_sem, text_vec)
        if eeg.grad is not None:
            eeg.grad.zero_()

        s = F.normalize(emb_sem, p=2, dim=-1)
        txt = F.normalize(text_vec, p=2, dim=-1)
        score_s = (s * txt).sum(dim=-1).mean()

        model.zero_grad(set_to_none=True)
        score_s.backward()

        grad_s = eeg.grad.detach()
        sal_s = attribution_from_grad_per_sample(eeg.detach(), grad_s, method=attr_method)  # [B,C]
        sal_s = apply_topk(sal_s, topk, topk_mode)
        sal_s_list.append(sal_s.cpu().numpy())

        n_used += 1
        n_samples += sal_v.shape[0]

    h1.remove()
    h2.remove()

    if n_used == 0:
        raise RuntimeError("No batches processed. Check DataLoader.")

    sal_v_all = np.concatenate(sal_v_list, axis=0).astype(np.float32)  # [N,C]
    sal_s_all = np.concatenate(sal_s_list, axis=0).astype(np.float32)  # [N,C]
    sal_v_raw = np.mean(sal_v_all, axis=0)
    sal_s_raw = np.mean(sal_s_all, axis=0)

    metrics = {
        "n_batches": n_used,
        "n_samples": n_samples,
        "attr_method": attr_method,
        "topk": int(topk) if topk else 0,
        "topk_mode": str(topk_mode),
        "mean_cos_emb_vis_emb_sem": float(np.mean(cos_emb_vs_list)),
        "mean_cos_img_vec_txt_vec": float(np.mean(cos_it_list)),
        "corr_sal_v_sal_s": float(np.corrcoef(sal_v_raw, sal_s_raw)[0, 1]),
        "mean_abs_diff_sal": float(np.mean(np.abs(sal_v_raw - sal_s_raw))),
        "max_abs_diff_sal": float(np.max(np.abs(sal_v_raw - sal_s_raw))),
    }
    return sal_v_raw, sal_s_raw, metrics


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cfg = OmegaConf.load(CONFIG_PATH)
    cfg.data.root = os.getcwd()

    dataset = TripletDataset(cfg.data, mode="val", split_index=0)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    model = build_model()
    print(f"Loading model: {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()

    print("Computing expert saliency (visual & semantic) ...")
    sal_v_raw, sal_s_raw, metrics = compute_expert_saliency_both(
        model, loader, DEVICE,
        max_batches=MAX_BATCHES,
        attr_method=ATTR_METHOD,
        topk=TOPK,
        topk_mode=TOPK_MODE
    )

    v_path = os.path.join(OUT_DIR, "saliency_expert_visual_raw.npy")
    s_path = os.path.join(OUT_DIR, "saliency_expert_semantic_raw.npy")
    np.save(v_path, sal_v_raw)
    np.save(s_path, sal_s_raw)

    metrics_path = os.path.join(OUT_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    ch_names, montage, is_template = get_montage_and_names(cfg.data.eeg_path)
    info = build_info(ch_names, montage)

    vmin = float(min(sal_v_raw.min(), sal_s_raw.min()))
    vmax = float(max(sal_v_raw.max(), sal_s_raw.max()))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    title_suffix = f"topk={TOPK}, mode={TOPK_MODE}, template={is_template}"

    plot_topomap(
        sal_v_raw, info,
        os.path.join(OUT_DIR, "expert_topomap_visual_common.png"),
        title=f"Visual Expert Attribution ({title_suffix})",
        vmin=vmin, vmax=vmax, cmap=CMAP_POSITIVE
    )
    plot_topomap(
        sal_s_raw, info,
        os.path.join(OUT_DIR, "expert_topomap_semantic_common.png"),
        title=f"Semantic Expert Attribution ({title_suffix})",
        vmin=vmin, vmax=vmax, cmap=CMAP_POSITIVE
    )

    ylabel = "Top-k frequency" if TOPK_MODE == "binary" and TOPK > 0 else "Mean attribution"
    v_stats = group_by_coords(info, sal_v_raw)
    s_stats = group_by_coords(info, sal_s_raw)
    plot_region_bar(
        v_stats,
        os.path.join(OUT_DIR, "expert_region_bar_visual_coords.png"),
        title="Visual Expert Attribution by Region (Coord-based)",
        ylabel=ylabel
    )
    plot_region_bar(
        s_stats,
        os.path.join(OUT_DIR, "expert_region_bar_semantic_coords.png"),
        title="Semantic Expert Attribution by Region (Coord-based)",
        ylabel=ylabel
    )

    diff = (sal_v_raw - sal_s_raw).astype(np.float32)
    dmax = float(np.max(np.abs(diff)) + 1e-6)

    plot_topomap(
        diff, info,
        os.path.join(OUT_DIR, "expert_topomap_visual_minus_semantic_common.png"),
        title=f"Expert Attribution Difference (Visual - Semantic, {title_suffix})",
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
