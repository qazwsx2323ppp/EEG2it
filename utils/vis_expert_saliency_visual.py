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
MODEL_PATH = "temp/best_12.8_change.pth"  # 改成你的
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
    """
    返回 (ch_names, montage, template_flag).
    目前优先保证能画 topomap；若缺真实通道名，则使用 biosemi128 模板并标注 template=True。
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
    """
    当 template=True 或通道名不可靠时，用坐标做粗分区（更稳健）：
    - Anterior / Posterior：按 y
    - Left / Right：按 x
    - Midline：|x| 很小
    """
    pos = np.array([ch["loc"][:3] for ch in info["chs"]], dtype=np.float32)  # (N,3)
    x = pos[:, 0]
    y = pos[:, 1]

    # 阈值经验设定：midline 更严格一点
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
    # 你的 SpatialMoEEncoder 当前不需要 visual_indices / semantic_indices
    model = SpatialMoEEncoder(n_channels=128, n_samples=512, embedding_dim=512).to(DEVICE)
    return model


def compute_expert_saliency_visual(model, loader, device, max_batches=20):
    """
    目标：cos(emb_vis, image_vector)
    attribution：mean_{batch,t} | d target / d eeg |
    """
    model.eval()

    # 用 hook 抓取 emb_vis
    cache = {"emb_vis": None}

    def hook_vis(module, inp, out):
        cache["emb_vis"] = out  # [B, 512]

    h = model.expert_visual_head.register_forward_hook(hook_vis)

    saliencies = []
    n_used = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break

        eeg, image_vec, _ = batch
        eeg = eeg.to(device)
        image_vec = image_vec.to(device)

        eeg.requires_grad_(True)

        # forward（触发 hook）
        _img_emb, _txt_emb, _weights = model(eeg)

        emb_vis = cache["emb_vis"]
        if emb_vis is None:
            raise RuntimeError("没有捕获到 emb_vis，请确认模型中存在 expert_visual_head。")

        # 标量目标：cosine(emb_vis, image_vector)
        emb_vis_n = F.normalize(emb_vis, p=2, dim=-1)
        image_vec_n = F.normalize(image_vec, p=2, dim=-1)

        score = (emb_vis_n * image_vec_n).sum(dim=-1)  # [B]
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

    print("Computing VISUAL EXPERT saliency: cos(emb_vis, image_vector) ...")
    sal = compute_expert_saliency_visual(model, loader, DEVICE, max_batches=MAX_BATCHES)
    sal_01 = normalize_01(sal)

    np.save(os.path.join(OUT_DIR, "saliency_expert_visual.npy"), sal_01)

    ch_names, montage, is_template = get_montage_and_names(cfg.data.eeg_path)

    plot_topomap(
        sal_01, ch_names, montage,
        os.path.join(OUT_DIR, "expert_topomap_visual.png"),
        title=f"Visual Expert Attribution (cos(emb_vis, img_vec), template={is_template})"
    )

    plot_region_bar_from_coords(
        ch_names, montage, sal_01,
        os.path.join(OUT_DIR, "expert_region_bar_visual_coords.png"),
        title="Visual Expert Attribution by Region (Coord-based Rough Matching)"
    )

    print("Saved visual expert maps to:", OUT_DIR)


if __name__ == "__main__":
    main()
