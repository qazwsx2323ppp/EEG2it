import sys
from pathlib import Path
project_root = Path(__file__).parent.parent 
sys.path.append(str(project_root))
import torch
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import os

from models.clip_models import SpatialMoEEncoder
from dataset import TripletDataset

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
    
    # 加载数据
    dataset = TripletDataset(cfg.data, mode='val', split_index=0)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    # === 【修改 1】 手动定义预期的视觉区域 (枕叶: 64-127) ===
    # 这只是为了给热力图画个参考框，和模型实际运行逻辑无关
    expected_visual_indices = list(range(64, 128)) 
    expected_semantic_indices = list(range(0, 64))
    # ======================================================

    # === 【修改 2】 实例化模型时不再从 cfg 读取 ===
    # 注意：如果你之前修改了 SpatialMoEEncoder 的 __init__ 删除了这两个参数，
    # 请把下面这两行 visual_indices=... 也删掉。
    # 如果没删 __init__ 参数，就传空列表 [] 或上面的 expected_visual_indices 占位。
    try:
        # 尝试方式 A: 假设你已经删除了 __init__ 中的参数
        model = SpatialMoEEncoder(
            n_channels=128, n_samples=512,
            embedding_dim=512
        ).to(DEVICE)
    except TypeError:
        # 尝试方式 B: 如果 __init__ 还没改，需要传入占位参数
        print(">>> 检测到模型仍需要索引参数，传入占位数据...")
        model = SpatialMoEEncoder(
            n_channels=128, n_samples=512,
            visual_indices=expected_visual_indices, # 传入占位
            semantic_indices=expected_semantic_indices, # 传入占位
            embedding_dim=512
        ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()
    
    # 获取一个 Batch
    batch = next(iter(loader))
    eeg, _, _ = batch # 解包 dataset 返回的三元组
    eeg = eeg.to(DEVICE)
    
    print("正在计算 Router 关注度热力图...")
    vis_saliency = compute_saliency(model, eeg)
    
    # 归一化
    vis_saliency = (vis_saliency - vis_saliency.min()) / (vis_saliency.max() - vis_saliency.min())
    
    # === 绘图 ===
    plt.figure(figsize=(15, 5))
    plt.bar(range(128), vis_saliency, color='blue', alpha=0.7, label='Actual Attention')
    
    # === 【修改 3】 使用手动定义的区域画框 ===
    plt.axvspan(64, 128, color='yellow', alpha=0.2, label='Expected Visual Region (Occipital)')
    
    plt.title("Which Channels determine the Visual Expert's Weight?")
    plt.xlabel("Channel Index (0-127)")
    plt.ylabel("Importance (Gradient Magnitude)")
    plt.legend()
    
    plt.savefig("router_saliency.png")
    print("✅ 分析完成！请查看 router_saliency.png")
    
    top_10 = vis_saliency.argsort()[-10:][::-1]
    print(f"Router 最关注的前 10 个通道索引: {top_10}")

if __name__ == "__main__":
    main()