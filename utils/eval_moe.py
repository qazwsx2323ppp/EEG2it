import sys
from pathlib import Path
project_root = Path(__file__).parent.parent 
sys.path.append(str(project_root))
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from omegaconf import OmegaConf
import seaborn as sns # 如果没有安装，可以使用 pip install seaborn，会让图表更好看

# 引入你的模块
from models.clip_models import SpatialMoEEncoder
from dataset import TripletDataset
from utils.loss_methods import InfoNCE

# === 配置区域 ===
CONFIG_PATH = "configs/triplet_config.yaml"
# 替换为你刚刚训练出的最佳权重路径
MODEL_PATH = "temp/best_12.8_change.pth"  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "results/ablation_study" # 结果保存目录
# ================

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
                # 注意：根据你的模型返回，weights 可能在 GPU 上，需要转到 CPU
                if 'w_vis_img' in weights:
                    all_vis_weights.extend(weights['w_vis_img'].cpu().numpy().flatten())
                if 'w_sem_txt' in weights:
                    all_sem_weights.extend(weights['w_sem_txt'].cpu().numpy().flatten())

    avg_img_loss = total_img_loss / len(loader)
    avg_txt_loss = total_txt_loss / len(loader)
    
    return avg_img_loss, avg_txt_loss, np.array(all_vis_weights), np.array(all_sem_weights)

def plot_results(results_df):
    """绘制论文可用的 Loss 对比图"""
    plt.figure(figsize=(10, 6))
    
    # 转换数据格式以便绘图 (Melt)
    df_melted = results_df.melt(id_vars=["Experiment"], 
                                value_vars=["Image Loss", "Text Loss"], 
                                var_name="Modality", 
                                value_name="Loss")
    
    # 设置 Seaborn 风格（可选）
    try:
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(data=df_melted, x="Experiment", y="Loss", hue="Modality", palette="viridis")
    except NameError:
        # 如果没有 seaborn，使用 matplotlib 标准绘图
        df_melted.pivot(index='Experiment', columns='Modality', values='Loss').plot(kind='bar')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.title("Impact of Expert Ablation on Retrieval Loss", fontsize=14)
    plt.ylabel("Loss (Lower is Better)", fontsize=12)
    plt.xlabel("Ablation Condition", fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title="Task Modality")
    
    save_path = os.path.join(OUTPUT_DIR, "ablation_comparison.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"📊 Loss 对比图已保存为: {save_path}")
    plt.close()

def plot_weights(vis_w, sem_w):
    """绘制权重分布直方图"""
    plt.figure(figsize=(10, 5))
    plt.hist(vis_w, bins=50, alpha=0.6, color='blue', label='Visual Expert Weights (w_vis)', density=True)
    plt.hist(sem_w, bins=50, alpha=0.6, color='orange', label='Semantic Expert Weights (w_sem)', density=True)
    
    plt.title("Distribution of Router Weights (Validation Set)", fontsize=14)
    plt.xlabel("Gate Value (0.0 - 1.0)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    save_path = os.path.join(OUTPUT_DIR, "router_distribution.png")
    plt.savefig(save_path, dpi=300)
    print(f"📊 权重分布图已保存为: {save_path}")
    plt.close()

def main():
    # 1. 加载配置
    cfg = OmegaConf.load(CONFIG_PATH)
    cfg.data.root = os.getcwd()  # 获取当前工作目录

    # 2. 准备数据
    val_dataset = TripletDataset(cfg.data, mode='val', split_index=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 3. 加载模型
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = SpatialMoEEncoder(
            n_channels=128,
            n_samples=512, 
            embedding_dim=512
        ).to(DEVICE)
    except TypeError:
        # 兼容性处理：如果你的模型定义还需要 indices 参数
        print("Model requires indices args, passing empty lists...")
        model = SpatialMoEEncoder(
            n_channels=128, n_samples=512,
            visual_indices=[], semantic_indices=[],
            embedding_dim=512
        ).to(DEVICE)
    
    # 加载权重
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    
    loss_fn = InfoNCE(initial_temperature=0.07).to(DEVICE)

    # === 定义实验列表 ===
    experiments = [
        ("Baseline (Full)", None),
        ("Kill Visual", "kill_visual"),
        ("Kill Semantic", "kill_semantic")
    ]

    results_data = []
    base_img_loss = 0
    base_txt_loss = 0

    # 4. 循环运行实验
    for exp_name, ablation_mode in experiments:
        print(f"\n>>> Running Experiment: {exp_name} ...")
        img_loss, txt_loss, vis_w, sem_w = run_validation(
            model, val_loader, loss_fn, ablation=ablation_mode, desc=exp_name
        )

        # 如果是 Baseline，保存权重分布图
        if ablation_mode is None:
            base_img_loss = img_loss
            base_txt_loss = txt_loss
            plot_weights(vis_w, sem_w)
        
        # 计算 Delta (变化量)
        delta_img = img_loss - base_img_loss
        delta_txt = txt_loss - base_txt_loss
        
        print(f"   Image Loss: {img_loss:.4f} (Delta: {delta_img:+.4f})")
        print(f"   Text Loss:  {txt_loss:.4f} (Delta: {delta_txt:+.4f})")
        
        # 记录数据
        results_data.append({
            "Experiment": exp_name,
            "Image Loss": img_loss,
            "Text Loss": txt_loss,
            "Image Loss Delta": delta_img,
            "Text Loss Delta": delta_txt
        })

    # 5. 保存数据到 CSV
    df = pd.DataFrame(results_data)
    csv_path = os.path.join(OUTPUT_DIR, "ablation_results.csv")
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\n✅ 实验数据已保存到: {csv_path}")
    print(df)

    # 6. 绘制 Loss 对比图
    plot_results(df)

    # === 结论分析 (自动写入文本文件) ===
    analysis_path = os.path.join(OUTPUT_DIR, "analysis_report.txt")
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write("=== MoE Ablation Study Analysis ===\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        
        # 简单的自动结论
        baseline = df[df['Experiment'] == "Baseline (Full)"].iloc[0]
        kill_vis = df[df['Experiment'] == "Kill Visual"].iloc[0]
        kill_sem = df[df['Experiment'] == "Kill Semantic"].iloc[0]

        if kill_vis['Image Loss'] > baseline['Image Loss'] + 0.05:
            msg = "✅ [验证成功] 切除视觉专家导致 Image Loss 显著上升，证明视觉专家主要负责视觉任务。\n"
            print(msg.strip())
            f.write(msg)
        
        if kill_sem['Text Loss'] > baseline['Text Loss'] + 0.05:
            msg = "✅ [验证成功] 切除语义专家导致 Text Loss 显著上升，证明语义专家主要负责文本任务。\n"
            print(msg.strip())
            f.write(msg)

    print(f"📄 详细分析报告已保存到: {analysis_path}")

if __name__ == "__main__":
    main()