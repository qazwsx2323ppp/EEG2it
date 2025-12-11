import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
from omegaconf import OmegaConf

# 引入你的模块
from models.clip_models_2o import SpatialMoEEncoder
from dataset_2o import TripletDataset
from utils.loss_methods import InfoNCE

# === 配置区域 ===
CONFIG_PATH = "configs/triplet_config.yaml"
# 替换为你刚刚训练出的最佳权重路径
MODEL_PATH = "temp/best_12.8_change.pth"  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ================

def run_validation(model, loader, loss_fn, ablation=None, desc="Validating"):
    model.eval()
    total_img_loss = 0.0
    total_txt_loss = 0.0
    
    # 用于存储权重分布
    all_vis_weights = []
    all_sem_weights = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            eeg, img_vecs, txt_vecs = batch
            eeg = eeg.to(DEVICE)
            img_vecs = img_vecs.to(DEVICE)
            txt_vecs = txt_vecs.to(DEVICE)

            # 传入 ablation 参数
            eeg_img, eeg_txt, weights = model(eeg, ablation=ablation)

            # 计算 Loss
            loss_i = loss_fn(eeg_img, img_vecs)
            loss_t = loss_fn(eeg_txt, txt_vecs)

            total_img_loss += loss_i.item()
            total_txt_loss += loss_t.item()
            
            # 收集权重 (取 batch 平均或所有样本)
            if weights:
                all_vis_weights.extend(weights['w_vis_img'].cpu().numpy().flatten())
                all_sem_weights.extend(weights['w_sem_txt'].cpu().numpy().flatten())

    avg_img_loss = total_img_loss / len(loader)
    avg_txt_loss = total_txt_loss / len(loader)
    
    return avg_img_loss, avg_txt_loss, np.array(all_vis_weights), np.array(all_sem_weights)

def main():
    # 1. 加载配置
    cfg = OmegaConf.load(CONFIG_PATH)
    
    cfg.data.root = os.getcwd()  # 获取当前工作目录

    # 2. 准备数据 (使用验证集)
    val_dataset = TripletDataset(cfg.data, mode='val', split_index=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 3. 加载模型
    print(f"Loading model from {MODEL_PATH}...")
    model = SpatialMoEEncoder(
        n_channels=128,
        n_samples=512, # 注意这里是插值后的长度
        visual_indices=cfg.model.moe_config.visual_indices,
        semantic_indices=cfg.model.moe_config.semantic_indices,
        embedding_dim=512
    ).to(DEVICE)
    
    # 加载权重 (注意处理 strict)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    
    loss_fn = InfoNCE(initial_temperature=0.07).to(DEVICE)

    # 4. === 实验一：基准测试 & 权重分布 ===
    print("\n>>> 正在运行基准测试 (Normal)...")
    base_img_loss, base_txt_loss, vis_w, sem_w = run_validation(model, val_loader, loss_fn, ablation=None)
    
    print(f"Base Image Loss: {base_img_loss:.4f}")
    print(f"Base Text  Loss: {base_txt_loss:.4f}")
    print(f"Visual Weight Mean: {vis_w.mean():.4f} | Std: {vis_w.std():.4f}")
    print(f"Semantic Weight Mean: {sem_w.mean():.4f} | Std: {sem_w.std():.4f}")

    # 绘制直方图
    plt.figure(figsize=(10, 5))
    plt.hist(vis_w, bins=50, alpha=0.5, label='Visual Expert Weights')
    plt.hist(sem_w, bins=50, alpha=0.5, label='Semantic Expert Weights')
    plt.legend()
    plt.title("Router Weight Distribution")
    plt.savefig("router_distribution.png")
    print("权重分布图已保存为 router_distribution.png")

    # 5. === 实验二：切除视觉专家 ===
    print("\n>>> 正在切除视觉专家 (Kill Visual)...")
    no_vis_img_loss, no_vis_txt_loss, _, _ = run_validation(model, val_loader, loss_fn, ablation='kill_visual')
    
    img_delta = no_vis_img_loss - base_img_loss
    txt_delta = no_vis_txt_loss - base_txt_loss
    
    print(f"Image Loss 变化: {base_img_loss:.4f} -> {no_vis_img_loss:.4f} (Delta: +{img_delta:.4f})")
    print(f"Text  Loss 变化: {base_txt_loss:.4f} -> {no_vis_txt_loss:.4f} (Delta: +{txt_delta:.4f})")

    # 6. === 实验三：切除语义专家 ===
    print("\n>>> 正在切除语义专家 (Kill Semantic)...")
    no_sem_img_loss, no_sem_txt_loss, _, _ = run_validation(model, val_loader, loss_fn, ablation='kill_semantic')
    
    img_delta_2 = no_sem_img_loss - base_img_loss
    txt_delta_2 = no_sem_txt_loss - base_txt_loss
    
    print(f"Image Loss 变化: {base_img_loss:.4f} -> {no_sem_img_loss:.4f} (Delta: +{img_delta_2:.4f})")
    print(f"Text  Loss 变化: {base_txt_loss:.4f} -> {no_sem_txt_loss:.4f} (Delta: +{txt_delta_2:.4f})")

    # === 结论分析 ===
    print("\n=== 结论分析 ===")
    if img_delta > txt_delta and img_delta > 0.1:
        print("✅ 验证成功：切除视觉专家对图像任务的破坏更大，说明视觉专家确实在专注处理视觉信息！")
    elif txt_delta_2 > img_delta_2 and txt_delta_2 > 0.1:
        print("✅ 验证成功：切除语义专家对文本任务的破坏更大，说明语义专家确实在专注处理语义信息！")
    else:
        print("⚠️ 验证不显著：专家的分工可能不够明确，或者 Router 仍在‘和稀泥’。")

if __name__ == "__main__":
    main()