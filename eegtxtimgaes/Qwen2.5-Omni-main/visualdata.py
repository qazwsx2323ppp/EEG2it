import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from omegaconf import OmegaConf

def load_dataset(project_root, mode='test', split_index=0):
    sys.path.append(project_root)
    from dataset import TripletDataset
    cfg_path = os.path.join(project_root, "Qwen2.5-Omni-main/configs", "triplet_config.yaml")
    cfg = OmegaConf.load(cfg_path)
    cfg.data.root = project_root
    ds = TripletDataset(cfg.data, mode=mode, split_index=split_index, return_text=True)
    return ds

def plot_timeseries(eeg, ch_indices=None, title="EEG Time Series"):
    if ch_indices is None:
        ch_indices = list(range(min(8, eeg.shape[0])))
    t = np.arange(eeg.shape[1])
    cols = 4
    rows = int(np.ceil(len(ch_indices)/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3*rows))
    axes = np.array(axes).reshape(-1)
    for i, ch in enumerate(ch_indices):
        axes[i].plot(t, eeg[ch])
        axes[i].set_title(f"Ch {ch}")
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Amp")
    for j in range(i+1, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    return fig
def plot_heatmap(eeg, title="EEG Channels x Time"):
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(eeg, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Channel")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig
def plot_psd(eeg, fs=512.0, ch_indices=None, title="EEG PSD (Welch)"):
    if ch_indices is None:
        ch_indices = [0, 32, 64, 96]
    fig, ax = plt.subplots(figsize=(10, 6))
    for ch in ch_indices:
        f, Pxx = welch(eeg[ch], fs=fs, nperseg=256)
        ax.semilogy(f, Pxx, label=f"Ch {ch}")
    ax.set_title(title)
    ax.set_xlabel("Freq (Hz)")
    ax.set_ylabel("Power")
    ax.legend()
    plt.tight_layout()
    return fig
def plot_embeddings_stats(img_vec, txt_vec, title="CLIP-Aligned Embeddings"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(img_vec.cpu().numpy(), bins=50, alpha=0.7, label="img")
    axes[0].hist(txt_vec.cpu().numpy(), bins=50, alpha=0.7, label="txt")
    axes[0].set_title("Value Distribution")
    axes[0].legend()
    axes[1].bar(["img_norm", "txt_norm"], [img_vec.norm().item(), txt_vec.norm().item()])
    axes[1].set_title("Vector Norms")
    axes[2].plot(img_vec.cpu().numpy(), label="img")
    axes[2].plot(txt_vec.cpu().numpy(), label="txt")
    axes[2].set_title("Vector Trace")
    axes[2].legend()
    fig.suptitle(title)
    plt.tight_layout()
    return fig
def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ds = load_dataset(project_root, mode="test", split_index=0)
    idx = 0
    sample = ds[idx]
    eeg, img_vec, txt_vec, raw_text = sample
    eeg_np = eeg.numpy()
    # plot_timeseries(eeg_np, title="EEG Time Series (First 8 Channels)")
    # plot_heatmap(eeg_np, title="EEG Heatmap (Channels x Time)")
    # plot_psd(eeg_np, fs=512.0, title="EEG PSD")
    # plot_embeddings_stats(img_vec, txt_vec, title="Image/Text Embedding Stats")
    ts_fig = plot_timeseries(eeg_np, title="EEG Time Series (First 8 Channels)")
    hmap_fig = plot_heatmap(eeg_np, title="EEG Heatmap (Channels x Time)")
    psd_fig = plot_psd(eeg_np, fs=512.0, title="EEG PSD")
    emb_fig = plot_embeddings_stats(img_vec, txt_vec, title="Image/Text Embedding Stats")
    out_dir = os.path.join(project_root, "outputs", "viz")
    os.makedirs(out_dir, exist_ok=True)
    ts_fig.savefig(os.path.join(out_dir, f"timeseries_{idx}.png"), dpi=200)
    hmap_fig.savefig(os.path.join(out_dir, f"heatmap_{idx}.png"), dpi=200)
    psd_fig.savefig(os.path.join(out_dir, f"psd_{idx}.png"), dpi=200)
    emb_fig.savefig(os.path.join(out_dir, f"embeddings_{idx}.png"), dpi=200)
    print("Saved to:", out_dir)
    print("EEG shape:", eeg.shape)
    print("Image vec shape:", img_vec.shape, "norm:", img_vec.norm().item())
    print("Text vec shape:", txt_vec.shape, "norm:", txt_vec.norm().item())
    print("Caption:", raw_text)
    plt.show()

if __name__ == "__main__":
    main()