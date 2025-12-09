import torch
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import os

from models.clip_models_2o import SpatialMoEEncoder
from dataset_2o import TripletDataset

# === 配置 ===
CONFIG_PATH = "configs/triplet_config.yaml"
MODEL_PATH = "temp/best_12.8_change.pth" # 替换你的路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ============

def compute_saliency(model, eeg_batch):
    # 1. 开启 EEG 输入的梯度追踪
    eeg_batch.requires_grad_()
    
    # 2. 前向传播
    # 我们只关心 Router 的输出，所以需要稍微 hack 一下 forward，或者直接调用 router 部分
    # 这里为了方便，我们直接通过 forward 拿到权重
    _, _, weights = model(eeg_batch)
    
    # weights['w_vis_img'] 是 [Batch, 1] 的张量
    # 我们想知道：哪些通道对 w_vis_img 的贡献最大？
    
    target = weights['w_vis_img']
    
    # 3. 反向传播 (计算 d(Weight) / d(Input))
    # 我们对 batch 内所有样本的权重求和，算总梯度
    target.sum().backward()
    
    # 4. 获取梯度 [Batch, 128, 512]
    # 取绝对值（正负贡献都算关注），并在时间维度(512)和Batch维度平均
    saliency = eeg_batch.grad.abs().mean(dim=(0, 2)) # [128]
    
    return saliency.cpu().numpy()

def main():
    cfg = OmegaConf.load(CONFIG_PATH)
    cfg.data.root = os.getcwd()
    
    # 加载数据和模型
    dataset = TripletDataset(cfg.data, mode='val', split_index=0)
    # 取一个大一点的 Batch 来算平均关注度
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = SpatialMoEEncoder(
        n_channels=128, n_samples=512,
        # visual_indices=[], semantic_indices=[], # 如果你删了参数，这里也删掉
        visual_indices=cfg.model.moe_config.visual_indices, # 兼容旧代码
        semantic_indices=cfg.model.moe_config.semantic_indices,
        embedding_dim=512
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()
    
    # 获取一个 Batch
    batch = next(iter(loader))
    eeg, _, _ = batch
    eeg = eeg.to(DEVICE)
    
    print("正在计算 Router 关注度热力图...")
    # 计算 Visual Expert 的关注度
    vis_saliency = compute_saliency(model, eeg)
    
    # 归一化到 0-1 以便绘图
    vis_saliency = (vis_saliency - vis_saliency.min()) / (vis_saliency.max() - vis_saliency.min())
    
    # === 绘图 ===
    plt.figure(figsize=(15, 5))
    
    # 绘制 128 个通道的关注度柱状图
    plt.bar(range(128), vis_saliency, color='blue', alpha=0.7)
    
    # 标记你认为应该是 Visual 的区域 (例如 64-127)
    plt.axvspan(64, 128, color='yellow', alpha=0.2, label='Expected Visual Region (Occipital)')
    
    plt.title("Which Channels determine the Visual Expert's Weight?")
    plt.xlabel("Channel Index (0-127)")
    plt.ylabel("Importance (Gradient Magnitude)")
    plt.legend()
    
    plt.savefig("router_saliency.png")
    print("✅ 分析完成！请查看 router_saliency.png")
    
    # 简单的文本分析
    top_10 = vis_saliency.argsort()[-10:][::-1]
    print(f"Router 最关注的前 10 个通道索引: {top_10}")
    print("如果这些索引大多落在 64-127 (枕叶) 范围内，说明 Router 自动学会了看枕叶！")

if __name__ == "__main__":
    main()