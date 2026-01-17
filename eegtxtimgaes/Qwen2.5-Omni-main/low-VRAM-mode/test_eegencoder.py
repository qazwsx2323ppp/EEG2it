# test_text_embedding.py (修复版)

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from dataset import TripletDataset
from models.clip_models import SpatialMoEEncoder


def cosine_similarity_matrix(a, b):
    """计算余弦相似度矩阵"""
    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1)
    similarity = torch.mm(a_norm, b_norm.t())
    return similarity


def compute_retrieval_metrics_fixed(eeg_embeddings, text_embeddings, true_text_indices):
    """
    修复后的检索指标计算
    eeg_embeddings: (N, D) - EEG编码器输出的文本向量
    text_embeddings: (M, D) - 所有可用的真实文本向量 (可能 M != N)
    true_text_indices: (N,) - 每个EEG样本对应的真实文本向量索引 (在text_embeddings中的索引)
    """
    device = eeg_embeddings.device
    N = eeg_embeddings.shape[0]
    M = text_embeddings.shape[0]
    
    # 计算相似度矩阵: (N, M)
    similarity_matrix = cosine_similarity_matrix(eeg_embeddings, text_embeddings)
    
    # 1. 计算正样本相似度（每个EEG与其真实匹配文本的相似度）
    positive_similarities = similarity_matrix[torch.arange(N, device=device), true_text_indices]
    avg_positive_sim = positive_similarities.mean().item()
    std_positive_sim = positive_similarities.std().item()
    
    # 2. 计算负样本相似度（每个EEG与其他所有文本的平均相似度）
    negative_similarities_list = []
    for i in range(N):
        true_idx = true_text_indices[i].item()
        # 排除正样本
        neg_mask = torch.ones(M, dtype=torch.bool, device=device)
        neg_mask[true_idx] = False
        neg_sims = similarity_matrix[i][neg_mask]
        negative_similarities_list.append(neg_sims.mean().item())
    
    avg_negative_sim = np.mean(negative_similarities_list)
    std_negative_sim = np.std(negative_similarities_list)
    
    # 3. 计算Top-K检索准确率
    top1_correct = 0
    top5_correct = 0
    top10_correct = 0
    ranks = []
    
    for i in range(N):
        true_idx = true_text_indices[i].item()
        sim_scores = similarity_matrix[i]  # (M,)
        
        # 获取排序后的索引（降序）
        sorted_indices = torch.argsort(sim_scores, descending=True)
        
        # 找到正样本的排名
        rank = (sorted_indices == true_idx).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)
        
        # 检查Top-K
        if rank <= 1:
            top1_correct += 1
        if rank <= 5:
            top5_correct += 1
        if rank <= 10:
            top10_correct += 1
    
    top1_acc = top1_correct / N
    top5_acc = top5_correct / N
    top10_acc = top10_correct / N
    
    # 4. 计算平均排名
    mean_rank = np.mean(ranks)
    median_rank = np.median(ranks)
    
    # 5. 计算分离度
    separation = avg_positive_sim - avg_negative_sim
    
    # 6. 计算Recall@K
    recall_at_1 = top1_acc
    recall_at_5 = top5_acc
    recall_at_10 = top10_acc
    
    metrics = {
        'avg_positive_similarity': avg_positive_sim,
        'std_positive_similarity': std_positive_sim,
        'avg_negative_similarity': avg_negative_sim,
        'std_negative_similarity': std_negative_sim,
        'separation': separation,
        'top1_accuracy': top1_acc,
        'top5_accuracy': top5_acc,
        'top10_accuracy': top10_acc,
        'recall_at_1': recall_at_1,
        'recall_at_5': recall_at_5,
        'recall_at_10': recall_at_10,
        'mean_rank': mean_rank,
        'median_rank': median_rank,
        'min_rank': int(np.min(ranks)),
        'max_rank': int(np.max(ranks)),
    }
    
    return metrics, ranks


def evaluate_text_embedding(
    model_path,
    config_path=None,
    test_batch_size=32,
    device='cuda',
    max_samples=None
):
    """评估训练后的EEG编码器的文本向量部分"""
    print("=" * 80)
    print("EEG编码器文本向量评估 (修复版)")
    print("=" * 80)
    
    # 1. 加载配置
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), 
            "..", 
            "configs", 
            "triplet_config.yaml"
        )
    
    if not os.path.exists(config_path):
        config_path = "configs/triplet_config.yaml"
    
    cfg = OmegaConf.load(config_path)
    project_root = os.path.abspath(os.path.dirname(__file__))
    if 'low-VRAM-mode' in project_root:
        project_root = os.path.dirname(project_root)
    cfg.data.root = project_root
    
    print(f"\n配置文件路径: {config_path}")
    print(f"数据根目录: {cfg.data.root}")
    
    # 2. 初始化模型
    print(f"\n>>> 正在初始化模型...")
    model = SpatialMoEEncoder(
        n_channels=cfg.model.n_channels,
        n_samples=cfg.model.n_samples,
        embedding_dim=cfg.model.embedding_dim,
        pretrained_path=None
    ).to(device)
    
    # 3. 加载权重
    print(f"\n>>> 正在加载模型权重: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v
    
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"权重加载详情:")
    print(f"  - 缺失的键: {len(msg.missing_keys)} 个")
    print(f"  - 意外的键: {len(msg.unexpected_keys)} 个")
    
    model.eval()
    print(">>> 模型已加载并设置为评估模式\n")
    
    # 4. 加载测试集和数据统计
    print(f">>> 正在加载测试数据集...")
    try:
        test_dataset = TripletDataset(cfg.data, mode='test', split_index=cfg.data.split_index)
        print(f"测试集大小: {len(test_dataset)}")
    except Exception as e:
        print(f"加载测试集失败: {e}")
        test_dataset = TripletDataset(cfg.data, mode='val', split_index=cfg.data.split_index)
        print(f"验证集大小: {len(test_dataset)}")
    
    # 加载所有文本向量用于检索
    all_text_vectors = torch.from_numpy(np.load(cfg.data.text_vec_path)).float()
    print(f"总文本向量数量: {len(all_text_vectors)}")
    
    # 限制样本数
    if max_samples is not None and max_samples < len(test_dataset):
        indices = torch.randperm(len(test_dataset))[:max_samples]
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
        print(f"限制测试样本数为: {max_samples}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    
    # 5. 提取特征并记录真实的文本索引
    print(f"\n>>> 正在提取特征（共 {len(test_dataset)} 个样本）...")
    all_eeg_text_embeddings = []
    true_text_indices_list = []  # 记录每个EEG对应的真实文本索引
    
    # 需要访问原始数据集来获取image索引
    original_dataset = test_dataset.dataset if hasattr(test_dataset, 'dataset') else test_dataset
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="提取特征")):
            eeg_signals, image_vectors, text_vectors = batch
            eeg_signals = eeg_signals.to(device)
            
            # 通过模型获取EEG编码的文本向量
            _, eeg_text_emb, _ = model(eeg_signals)
            all_eeg_text_embeddings.append(eeg_text_emb.cpu())
            
            # 记录每个样本对应的真实文本索引
            batch_start = batch_idx * test_batch_size
            batch_size_actual = eeg_signals.shape[0]
            
            for i in range(batch_size_actual):
                sample_idx = batch_start + i
                # 获取原始EEG索引
                if hasattr(original_dataset, 'indices'):
                    # Subset情况
                    original_eeg_idx = original_dataset.indices[sample_idx]
                else:
                    original_eeg_idx = sample_idx
                
                # 获取对应的image索引（即文本向量索引）
                eeg_item = original_dataset.all_eeg_items[original_eeg_idx]
                text_index = eeg_item['image']
                true_text_indices_list.append(text_index)
    
    # 合并所有批次
    all_eeg_text_embeddings = torch.cat(all_eeg_text_embeddings, dim=0)
    true_text_indices = torch.tensor(true_text_indices_list, dtype=torch.long)
    
    print(f"EEG文本向量形状: {all_eeg_text_embeddings.shape}")
    print(f"真实文本索引数量: {len(true_text_indices)}")
    print(f"唯一文本索引数量: {len(torch.unique(true_text_indices))}")
    
    # 6. 计算指标（使用所有文本向量作为检索库）
    print(f"\n>>> 正在计算评估指标...")
    all_eeg_text_embeddings = all_eeg_text_embeddings.to(device)
    all_text_vectors = all_text_vectors.to(device)
    true_text_indices = true_text_indices.to(device)
    
    metrics, ranks = compute_retrieval_metrics_fixed(
        all_eeg_text_embeddings,
        all_text_vectors,  # 使用所有文本向量作为检索库
        true_text_indices
    )
    
    # 7. 打印结果
    print("\n" + "=" * 80)
    print("评估结果（文本向量部分）")
    print("=" * 80)
    print(f"\n【相似度统计】")
    print(f"  正样本平均相似度: {metrics['avg_positive_similarity']:.4f} ± {metrics['std_positive_similarity']:.4f}")
    print(f"  负样本平均相似度: {metrics['avg_negative_similarity']:.4f} ± {metrics['std_negative_similarity']:.4f}")
    print(f"  分离度 (正样本 - 负样本): {metrics['separation']:.4f}")
    
    print(f"\n【检索准确率】")
    print(f"  Top-1 准确率: {metrics['top1_accuracy']:.4f} ({metrics['top1_accuracy']*100:.2f}%)")
    print(f"  Top-5 准确率: {metrics['top5_accuracy']:.4f} ({metrics['top5_accuracy']*100:.2f}%)")
    print(f"  Top-10 准确率: {metrics['top10_accuracy']:.4f} ({metrics['top10_accuracy']*100:.2f}%)")
    
    print(f"\n【排名统计】")
    print(f"  平均排名 (Mean Rank): {metrics['mean_rank']:.2f}")
    print(f"  中位数排名 (Median Rank): {metrics['median_rank']:.2f}")
    print(f"  最佳排名 (Min Rank): {metrics['min_rank']}")
    print(f"  最差排名 (Max Rank): {metrics['max_rank']}")
    
    print(f"\n【Recall指标】")
    print(f"  Recall@1: {metrics['recall_at_1']:.4f}")
    print(f"  Recall@5: {metrics['recall_at_5']:.4f}")
    print(f"  Recall@10: {metrics['recall_at_10']:.4f}")
    
    # 8. 诊断信息
    print(f"\n【诊断信息】")
    if metrics['avg_positive_similarity'] < 0.1:
        print(f"  ⚠️ 警告：正样本相似度极低 ({metrics['avg_positive_similarity']:.4f})")
        print(f"     可能原因：")
        print(f"     1. 模型未学到有效表示（检查训练过程）")
        print(f"     2. 文本向量未归一化（检查数据预处理）")
        print(f"     3. 模型输出未归一化（检查模型forward）")
    
    if metrics['separation'] < 0.1:
        print(f"  ⚠️ 警告：分离度很小 ({metrics['separation']:.4f})")
        print(f"     说明模型无法区分正样本和负样本")
    
    if metrics['top1_accuracy'] < 0.01:
        print(f"  ⚠️ 警告：Top-1准确率接近随机 ({metrics['top1_accuracy']*100:.2f}%)")
        print(f"     模型表现可能需要改进")
    else:
        print(f"  ✅ 模型学习到了一定的表示能力")
    
    print("\n" + "=" * 80)
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试EEG编码器的文本向量部分")
    parser.add_argument("--model_path", type=str, required=True, help="模型权重路径")
    parser.add_argument("--config_path", type=str, default=None, help="配置文件路径")
    parser.add_argument("--batch_size", type=int, default=32, help="测试batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    parser.add_argument("--max_samples", type=int, default=None, help="最大测试样本数")
    
    args = parser.parse_args()
    
    metrics = evaluate_text_embedding(
        model_path=args.model_path,
        config_path=args.config_path,
        test_batch_size=args.batch_size,
        device=args.device,
        max_samples=args.max_samples
    )
    
    print("\n评估完成！")