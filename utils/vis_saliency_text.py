import torch
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import os

from models.clip_models import SpatialMoEEncoder
from dataset import TripletDataset

# === 配置 ===
CONFIG_PATH = "configs/triplet_config.yaml"
# 请确保这里指向你训练好的模型路径 (与 vis_saliency.py 保持一致)
MODEL_PATH = "temp/best_12.8_change.pth" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ============

def compute_semantic_saliency(model, eeg_batch):
    """
    计算针对“语言专家权重”的 Saliency Map
    """
    # 1. 开启 EEG 输入的梯度追踪
    eeg_batch.requires_grad_()
    
    # 2. 前向传播
    # 获取 Router 输出的权重字典
    _, _, weights = model(eeg_batch)
    
    # --- 【关键修改 1】 目标改为语言/语义专家权重 'w_sem_txt' ---
    # 我们想知道：哪些通道对 w_sem_txt 的贡献最大？
    target = weights['w_sem_txt']
    
    # 3. 反向传播 (计算 d(Weight_Semantic) / d(Input))
    # 我们对 batch 内所有样本的权重求和，算总梯度
    target.sum().backward()
    
    # 4. 获取梯度 [Batch, 128, 512]
    # 取绝对值（正负贡献都算关注），并在时间维度(512)和Batch维度平均
    saliency = eeg_batch.grad.abs().mean(dim=(0, 2)) # [128]
    
    return saliency.cpu().numpy()

def main():
    # 1. 加载配置
    cfg = OmegaConf.load(CONFIG_PATH)
    cfg.data.root = os.getcwd()
    
    # 2. 加载数据 (使用验证集)
    dataset = TripletDataset(cfg.data, mode='val', split_index=0)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    # === 【关键修改 2】 定义预期的语义区域 (额叶/颞叶: 0-63) ===
    # 对应 configs/triplet_config.yaml 中的 semantic_indices
    # 这只是为了给热力图画个参考框
    expected_semantic_indices = list(range(0, 64))
    # ======================================================

    # 3. 实例化模型
    # 保持与 vis_saliency.py 一致的兼容性逻辑
    try:
        model = SpatialMoEEncoder(
            n_channels=128, n_samples=512,
            embedding_dim=512
        ).to(DEVICE)
    except TypeError:
        print(">>> 检测到模型仍需要索引参数，传入占位数据...")
        model = SpatialMoEEncoder(
            n_channels=128, n_samples=512,
            visual_indices=[], # 仅占位
            semantic_indices=[], # 仅占位
            embedding_dim=512
        ).to(DEVICE)

    # 4. 加载权重
    print(f"Loading model from {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()
    
    # 5. 获取一个 Batch
    batch = next(iter(loader))
    eeg, _, _ = batch 
    eeg = eeg.to(DEVICE)
    
    print("正在计算 Router 对【语言专家】的关注度热力图...")
    vis_saliency = compute_semantic_saliency(model, eeg)
    
    # 归一化 (0-1)
    vis_saliency = (vis_saliency - vis_saliency.min()) / (vis_saliency.max() - vis_saliency.min())
    
    # === 6. 绘图 ===
    plt.figure(figsize=(15, 5))
    
    # 使用绿色柱子代表语义关注度
    plt.bar(range(128), vis_saliency, color='green', alpha=0.7, label='Actual Attention (Semantic)')
    
    # === 【关键修改 3】 绘制语义区域背景 (0-63) ===
    # 使用青色背景标注预期的额叶/颞叶区域
    plt.axvspan(0, 64, color='cyan', alpha=0.2, label='Expected Semantic Region (Frontal/Temporal)')
    
    plt.title("Which Channels determine the Semantic Expert's Weight?")
    plt.xlabel("Channel Index (0-127)")
    plt.ylabel("Importance (Gradient Magnitude)")
    plt.legend()
    
    output_filename = "router_saliency_semantic.png"
    plt.savefig(output_filename)
    print(f"✅ 分析完成！请查看 {output_filename}")
    
    # 打印最关注的 Top 10 通道
    top_10 = vis_saliency.argsort()[-10:][::-1]
    print(f"Router 启用语言专家时最关注的前 10 个通道索引: {top_10}")

if __name__ == "__main__":
    main()