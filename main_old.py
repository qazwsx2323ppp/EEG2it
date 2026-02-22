# 忽略兼容警告
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import json
import signal
import sys
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 混合精度和调度器 ---
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup

# 导入本地代码
from models.clip_models import SpatialMoEEncoder
from utils.loss_methods import InfoNCE
from dataset import TripletDataset

# 设置 PyTorch 以获得更好的性能
torch.backends.cudnn.benchmark = True

# ============ 信号处理和优雅退出 ============
_STOP_REQUESTED = False

def _install_signal_handlers():
    """安装信号处理器，支持优雅退出"""
    global _STOP_REQUESTED
    prev_int = None
    try:
        prev_int = signal.getsignal(signal.SIGINT)
    except Exception:
        prev_int = None

    def _handler_int(signum, frame):
        global _STOP_REQUESTED
        _STOP_REQUESTED = True
        print("\n[信号] 收到中断信号，正在安全退出...")
        if callable(prev_int):
            prev_int(signum, frame)
        raise KeyboardInterrupt()

    def _handler_term(signum, frame):
        global _STOP_REQUESTED
        _STOP_REQUESTED = True

    try:
        signal.signal(signal.SIGINT, _handler_int)
    except Exception:
        pass
    try:
        signal.signal(signal.SIGTERM, _handler_term)
    except Exception:
        pass

def _stop_requested() -> bool:
    return bool(_STOP_REQUESTED)

# ============ 指标存储工具函数 ============
def _write_json(path: str, obj) -> None:
    """写入 JSON 文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _append_jsonl(path: str, obj) -> None:
    """追加 JSONL 条目"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ============ WandB 初始化工具 ============
def _maybe_init_wandb(cfg: DictConfig, enabled: bool):
    """可选地初始化 WandB"""
    if not enabled:
        return None
    try:
        import wandb
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        return wandb
    except Exception as e:
        print(f"[警告] WandB 初始化失败，继续训练: {e}")
        return None

# ============ 模型选择指标工具 ============
def _get_metric_from_val(val_results: dict, metric: str) -> float:
    """从验证结果中提取指定指标"""
    metric = metric.strip()
    if metric in {"val/loss_total", "loss", "val_loss"}:
        return float(val_results["loss"])
    if metric == "val/txt_top1":
        return float(val_results["txt_metrics"]["top_1_accuracy"])
    if metric == "val/txt_top5":
        return float(val_results["txt_metrics"]["top_5_accuracy"])
    if metric == "val/img_top1":
        return float(val_results["img_metrics"]["top_1_accuracy"])
    if metric == "val/img_top5":
        return float(val_results["img_metrics"]["top_5_accuracy"])
    # 默认返回 loss
    return float(val_results["loss"])

#已经在下面手动控制了 requires_grad
# def freeze_backbone(model, freeze=True):
#     """
#     冻结或解冻 Backbone 的辅助函数
#     """
#     # 假设你的 SpatialMoEEncoder 中有一个 self.backbone (即 MAEforEEG)
#     if hasattr(model, 'backbone'):
#         for param in model.backbone.parameters():
#             param.requires_grad = not freeze
#         state = "冻结" if freeze else "解冻"
#         print(f">>> Backbone 已{state}。")
#     else:
#         print("警告: 模型中未找到 'backbone' 属性，无法执行冻结/解冻操作。")


def train_one_epoch(model, dataloader, optimizer, loss_fn_img, loss_fn_txt, device, alpha, scaler, scheduler, 
                    grad_accum_steps=1, sanity_check=False):
    """
    执行一个周期的训练 (加入了 AMP、Scheduler、梯度累积和Sanity Check)
    """
    model.train()
    total_loss = 0.0
    total_loss_img = 0.0
    total_loss_txt = 0.0
    steps = 0
    did_sanity = False

    # 用于累计权重值
    total_weights = {
        "w_vis_img": 0.0, "w_fus_img": 0.0, 
        "w_sem_txt": 0.0, "w_fus_txt": 0.0
    }

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        # 检查是否请求停止
        if _stop_requested():
            print("[信号] 检测到停止请求，结束当前 epoch...")
            break
            
        eeg_signals, image_vecs, text_vecs = batch

        # 将数据移动到GPU
        eeg_signals = eeg_signals.to(device)
        image_vecs = image_vecs.to(device)
        text_vecs = text_vecs.to(device)
        
        # 这里的目的是检查输入是否符合 Mean=0, Std=1 的标准
        #print(f"\n[DEBUG Check] Input Mean: {eeg_signals.mean().item():.4f}, Std: {eeg_signals.std().item():.4f}, Max: {eeg_signals.max().item():.4f}, Min: {eeg_signals.min().item():.4f}")

        # --- 【修改】 混合精度前向传播 ---
        with autocast():
            # 前向传播
            outputs = model(eeg_signals)
            
            if len(outputs) == 3:
                eeg_img_embeddings, eeg_text_embeddings, weights_info = outputs
                # 累加权重
                if weights_info:
                    for k, v in weights_info.items():
                        total_weights[k] += v.item()
            else:
                # 兼容旧接口，防止报错
                eeg_img_embeddings, eeg_text_embeddings = outputs
                weights_info = None
            
            # === 【调试探针】请插入这段代码 ===
            # if batch_idx % 50 == 0: # 每50个batch打印一次
            #     print(f"\n[DEBUG Step {batch_idx}]")
                
                # 1. 检查输入 EEG 是否正常 (应该 Mean≈0, Std≈1)
                #print(f"  Input EEG: Mean={eeg_signals.mean().item():.3f}, Std={eeg_signals.std().item():.3f}, Min={eeg_signals.min().item():.3f}")
                # if torch.isnan(eeg_signals).any():
                #     print("  !!! ALERT: Input EEG contains NaN!")
    
                # 2. 检查输出 Embedding 的模长 (如果 L2 生效，Norm 应该严格等于 1.0)
                # img_norm = eeg_img_embeddings.norm(dim=-1).mean().item()
                # txt_norm = eeg_text_embeddings.norm(dim=-1).mean().item()
                #print(f"  Output Norm: Img={img_norm:.4f}, Txt={txt_norm:.4f} (Expect 1.0)")
                
                # 3. 检查输出是否“死”了 (即所有样本输出都一样)
                # 计算 Batch 内第一个样本和第二个样本的相似度。如果接近 1.0，说明模型发生了坍塌。
                # sim = torch.nn.functional.cosine_similarity(eeg_img_embeddings[0], eeg_img_embeddings[1], dim=0)
                #print(f"  Diversity Check: Sim(Batch[0], Batch[1]) = {sim.item():.4f} (接近 1.0 说明模型坍塌)")
            # ================================

            # 计算损失
            loss_img = loss_fn_img(eeg_img_embeddings, image_vecs)
            loss_txt = loss_fn_txt(eeg_text_embeddings, text_vecs)

            # 加权联合损失
            loss = (alpha * loss_img) + ((1 - alpha) * loss_txt)

        # --- 【修改】 混合精度反向传播 ---
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_loss_img += loss_img.item()
        total_loss_txt += loss_txt.item()

    avg_loss = total_loss / len(dataloader)
    avg_loss_img = total_loss_img / len(dataloader)
    avg_loss_txt = total_loss_txt / len(dataloader)

    # 返回平均权重字典
    avg_weights = {}
    if total_weights["w_vis_img"] > 0: 
        for k in total_weights:
            avg_weights[k] = total_weights[k] / len(dataloader)
            
    return avg_loss, avg_loss_img, avg_loss_txt, avg_weights

# 添加到 main.py 中

def compute_retrieval_metrics(eeg_embeddings, target_embeddings, k_values=[1, 5, 10]):
    """
    计算检索准确率
    """
    # 计算余弦相似度矩阵
    similarity_matrix = torch.matmul(eeg_embeddings, target_embeddings.T)
    
    # 获取排序后的索引
    _, sorted_indices = torch.sort(similarity_matrix, dim=1, descending=True)
    
    # 创建正确标签
    correct_labels = torch.arange(similarity_matrix.shape[0], 
                                  device=similarity_matrix.device)
    
    metrics = {}
    for k in k_values:
        top_k_indices = sorted_indices[:, :k]
        hits = (top_k_indices == correct_labels.unsqueeze(1)).any(dim=1)
        accuracy = hits.float().mean().item()
        metrics[f"top_{k}_accuracy"] = accuracy
    
    # 计算平均相似度
    positive_sim = torch.diag(similarity_matrix).mean().item()
    
    # 计算负样本相似度（非对角线元素）
    mask = ~torch.eye(similarity_matrix.shape[0], 
                     dtype=torch.bool, 
                     device=similarity_matrix.device)
    negative_sim = similarity_matrix[mask].mean().item()
    
    metrics['mean_positive_similarity'] = positive_sim
    metrics['mean_negative_similarity'] = negative_sim
    metrics['separation_ratio'] = positive_sim / (negative_sim + 1e-8)
    
    return metrics


def enhanced_validate(model, dataloader, loss_fn_img, loss_fn_txt, device, alpha):
    """
    增强的验证函数，包含更多指标
    """
    model.eval()
    total_loss_val = 0.0
    total_loss_val_img = 0.0
    total_loss_val_txt = 0.0
    
    all_eeg_img_embeddings = []
    all_eeg_txt_embeddings = []
    all_target_img_embeddings = []
    all_target_txt_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            eeg_signals, image_vecs, text_vecs = batch
            
            eeg_signals = eeg_signals.to(device)
            image_vecs = image_vecs.to(device)
            text_vecs = text_vecs.to(device)
            
            outputs = model(eeg_signals)
            
            if len(outputs) == 3:
                eeg_img_embedding, eeg_txt_embedding, _ = outputs
            else:
                eeg_img_embedding, eeg_txt_embedding = outputs
            
            loss_img = loss_fn_img(eeg_img_embedding, image_vecs)
            loss_txt = loss_fn_txt(eeg_txt_embedding, text_vecs)
            loss = (alpha * loss_img) + ((1 - alpha) * loss_txt)
            
            total_loss_val += loss.item()
            total_loss_val_img += loss_img.item()
            total_loss_val_txt += loss_txt.item()
            
            # 收集所有嵌入向量用于计算检索指标
            all_eeg_img_embeddings.append(eeg_img_embedding.cpu())
            all_eeg_txt_embeddings.append(eeg_txt_embedding.cpu())
            all_target_img_embeddings.append(image_vecs.cpu())
            all_target_txt_embeddings.append(text_vecs.cpu())
    
    # 合并所有batch
    all_eeg_img = torch.cat(all_eeg_img_embeddings, dim=0).to(device)
    all_eeg_txt = torch.cat(all_eeg_txt_embeddings, dim=0).to(device)
    all_target_img = torch.cat(all_target_img_embeddings, dim=0).to(device)
    all_target_txt = torch.cat(all_target_txt_embeddings, dim=0).to(device)
    
    # 计算检索指标
    img_metrics = compute_retrieval_metrics(all_eeg_img, all_target_img)
    txt_metrics = compute_retrieval_metrics(all_eeg_txt, all_target_txt)
    
    avg_loss_val = total_loss_val / len(dataloader)
    avg_loss_val_img = total_loss_val_img / len(dataloader)
    avg_loss_val_txt = total_loss_val_txt / len(dataloader)
    
    return {
        'loss': avg_loss_val,
        'loss_img': avg_loss_val_img,
        'loss_txt': avg_loss_val_txt,
        'img_metrics': img_metrics,
        'txt_metrics': txt_metrics
    }


def check_parameter_updates(model, optimizer):
    """
    检查参数是否在更新（用于验证微调是否生效）
    """
    trainable_params = []
    frozen_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append((name, param.numel()))
        else:
            frozen_params.append((name, param.numel()))
    
    total_trainable = sum(p[1] for p in trainable_params)
    total_frozen = sum(p[1] for p in frozen_params)
    
    print(f"\n【参数更新检查】")
    print(f"可训练参数: {total_trainable:,} ({total_trainable/(total_trainable+total_frozen)*100:.1f}%)")
    print(f"冻结参数: {total_frozen:,} ({total_frozen/(total_trainable+total_frozen)*100:.1f}%)")
    print(f"\n可训练参数示例（前5个）:")
    for name, numel in trainable_params[:5]:
        print(f"  - {name}: {numel:,}")
    
    return {
        'trainable_count': total_trainable,
        'frozen_count': total_frozen,
        'trainable_ratio': total_trainable / (total_trainable + total_frozen)
    }


def validate(model, dataloader, loss_fn_img, loss_fn_txt, device, alpha):
    """
    在验证集上评估模型
    """
    model.eval()
    total_loss_val = 0.0
    total_loss_val_img = 0.0
    total_loss_val_txt = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            eeg_signals, image_vecs, text_vecs = batch

            eeg_signals = eeg_signals.to(device)
            image_vecs = image_vecs.to(device)
            text_vecs = text_vecs.to(device)

            # 验证集通常不需要 autocast，除非显存非常紧缺
            outputs = model(eeg_signals)
            
            # 处理返回值解包
            if len(outputs) == 3:
                eeg_img_embedding, eeg_txt_embedding, _ = outputs
            else:
                eeg_img_embedding, eeg_txt_embedding = outputs

            loss_img = loss_fn_img(eeg_img_embedding, image_vecs)
            loss_txt = loss_fn_txt(eeg_txt_embedding, text_vecs)

            loss = (alpha * loss_img) + ((1 - alpha) * loss_txt)

            total_loss_val += loss.item()
            total_loss_val_img += loss_img.item()
            total_loss_val_txt += loss_txt.item()

    avg_loss_val = total_loss_val / len(dataloader)
    avg_loss_val_img = total_loss_val_img / len(dataloader)
    avg_loss_val_txt = total_loss_val_txt / len(dataloader)

    return avg_loss_val, avg_loss_val_img, avg_loss_val_txt


@hydra.main(version_base=None, config_path="configs", config_name="triplet_config")
def main(cfg: DictConfig):
    print("Hydra 配置:\n", OmegaConf.to_yaml(cfg))

    # 初始化 WandB
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name,
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    device = torch.device(cfg.training.device)

    print(f"正在初始化模型: SpatialMoEEncoder (Backbone: DreamDiffusion)")

    # 1. 实例化 SpatialMoEEncoder
    # --- 【关键修正】 传入 pretrained_path ---
    model = SpatialMoEEncoder(
        n_channels=cfg.model.n_channels,
        n_samples=cfg.model.n_samples,
        # visual_indices=cfg.model.moe_config.visual_indices,
        # semantic_indices=cfg.model.moe_config.semantic_indices,
        embedding_dim=cfg.model.embedding_dim,
        pretrained_path=cfg.model.get("pretrained_path", None) # 确保从 Config 读取路径
    ).to(device)
        
    print(">>> 成功初始化 Spatial MoE Encoder")

    # 2. 准备数据
    split_index = cfg.data.get("split_index", 0)

    train_dataset = TripletDataset(cfg.data, mode='train', split_index=split_index)
    val_dataset = TripletDataset(cfg.data, mode='val', split_index=split_index)
    test_dataset = TripletDataset(cfg.data, mode='test', split_index=split_index)

    # --- 建议增大 Batch Size (得益于 AMP) ---
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True,
                              num_workers=cfg.training.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False,
                            num_workers=cfg.training.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False,
                             num_workers=cfg.training.num_workers, pin_memory=True)

    # 3. 初始化损失函数、优化器、调度器、Scaler
    # loss_fn_img = InfoNCE(temperature=cfg.training.temperature).to(device)
    # loss_fn_txt = InfoNCE(temperature=cfg.training.temperature).to(device)
    
    # 修改后：将参数名 temperature 改为 initial_temperature
    loss_fn_img = InfoNCE(initial_temperature=cfg.training.temperature).to(device)
    loss_fn_txt = InfoNCE(initial_temperature=cfg.training.temperature).to(device)

    # # 使用 AdamW，通常 ViT 微调需要较小的 LR，但这里通过调度器控制
    # optimizer = optim.AdamW(
    #     model.parameters(), 
    #     lr=cfg.training.learning_rate,
    #     weight_decay=cfg.training.get("weight_decay", 0.05)
    # )

    # 1. 将参数分组
    # backbone_params = []
    # head_params = []

    # for name, param in model.named_parameters():
    #     if "backbone" in name:
    #         backbone_params.append(param)
    #     else:
    #         # 包括 router, expert_heads 等
    #         head_params.append(param)
    params_backbone_frozen = [] # 前面的层 (冻结)
    params_backbone_active = [] # 最后的层 (微调)
    params_head = []            # Expert Heads (全速训练)
    loss_params = list(loss_fn_img.parameters()) + list(loss_fn_txt.parameters())

    for name, param in model.named_parameters():
        if "backbone" in name:
            # 这里的 "blocks.23" 对应 DreamDiffusion (Depth=24) 的最后一层
            # 也可以加上 "norm" 层
            if "blocks.23" in name or "blocks.22" in name or "norm." in name:
                params_backbone_active.append(param)
                param.requires_grad = True # 确保设为 True
            else:
                params_backbone_frozen.append(param)
                param.requires_grad = False # 确保冻结
        else:
            params_head.append(param)
            param.requires_grad = True

    # # 2. 定义优化器，给 Backbone 一个更小的学习率 (通常是主学习率的 1/10 或 1/100)
    # optimizer = optim.AdamW(
    #     [
    #         # Backbone 使用非常小的学习率，小心翼翼地微调
    #         {"params": backbone_params, "lr": cfg.training.learning_rate * 0.01}, 
    #         # Heads 使用正常的学习率
    #         {"params": head_params, "lr": cfg.training.learning_rate}, 
    #     ],
    #     weight_decay=cfg.training.get("weight_decay", 0.05)
    # )

       # 核心修改：保护你的 DreamDiffusion Backbone
    # optimizer = optim.AdamW(
    #     [
    #         # 给 Backbone 一个极小的学习率 (如 1e-6 或 5e-6)
    #         {"params": backbone_params, "lr": cfg.training.learning_rate * 0.05}, 
    #         # 给 Router 和 Heads 正常的学习率 (如 1e-4)
    #         {"params": head_params, "lr": cfg.training.learning_rate}, 
    #     ],
    #        weight_decay=cfg.training.get("weight_decay", 0.1)
    # )
    optimizer = optim.AdamW(
        [
            # 头部：大学习率
            {"params": params_head, "lr": cfg.training.learning_rate}, 
            # 尾部 Backbone：小学习率 (防止破坏特征)
            {"params": params_backbone_active, "lr": cfg.training.learning_rate * 0.1}, 
            # 【新增】 Loss 的参数 (温度)
            # CLIP 官方对这个参数通常不使用 weight decay
            {"params": loss_params, "lr": cfg.training.learning_rate, "weight_decay": 0.0},
        ],
        weight_decay=0.15
    )

    # --- 【新增】 学习率调度器 ---
    # 总步数 = epoch * steps_per_epoch
    num_training_steps = cfg.training.epochs * len(train_loader)
    # 预热步数设为总步数的 10%
    num_warmup_steps = int(0.1 * num_training_steps)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # --- 【新增】 混合精度 Scaler ---
    scaler = GradScaler()
    param_info = check_parameter_updates(model, optimizer)

    wandb.log({
        "trainable_params": param_info['trainable_count'],
        "frozen_params": param_info['frozen_count'],
        "trainable_ratio": param_info['trainable_ratio']
    })
    # 4. 训练循环
    print("开始训练...")
    best_val_loss = float('inf')
    
    patience = cfg.training.get("patience", cfg.training.epochs) 
    min_delta = cfg.training.get("min_delta", 0.0)
    epochs_no_improve = 0
    
    # 设定解冻 Backbone 的 Epoch (例如第 10 个 Epoch)
    #既然数据量这么小，微调 Backbone 极易导致“灾难性遗忘”和过拟合。 
    # DreamDiffusion 的预训练权重已经包含了非常通用的脑电特征（在 10 万+ 样本上练出来的）。
    # 完全信任它，只训练后面的“翻译层”（Expert Heads）
    # 将解冻轮数设为一个不可能达到的数字，实现全程冻结
    unfreeze_epoch = 10000 
    
    # --- 【新增】 初始冻结 Backbone ---
    # freeze_backbone(model, freeze=True)

    for epoch in range(cfg.training.epochs):
        #在前面已经手动控制冻结
        # --- 【新增】 在指定 Epoch 解冻 ---
        # if epoch == unfreeze_epoch:
        #     print(f">>> 达到第 {epoch} 轮，开始解冻 Backbone 进行全局微调...")
        #     freeze_backbone(model, freeze=False)
        #     # 可选：解冻后可以重置学习率或调整
            
        avg_loss, avg_loss_img, avg_loss_txt, avg_weights = train_one_epoch(
            model, train_loader, optimizer, loss_fn_img, loss_fn_txt, device, 
            cfg.training.alpha, scaler, scheduler
        )

        val_results = enhanced_validate(
            model, val_loader, loss_fn_img, loss_fn_txt, device, cfg.training.alpha
        )

        avg_loss_val = val_results['loss']
        avg_loss_val_img = val_results['loss_img']
        avg_loss_val_txt = val_results['loss_txt']

        # 获取当前学习率用于记录（修复：添加这行）
        current_lr = optimizer.param_groups[0]["lr"]

        # === 【增强】 详细的训练效果展示 ===
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{cfg.training.epochs} | LR: {current_lr:.6f}")
        print(f"{'='*70}")
        
        # 损失指标
        print(f"\n【损失指标】")
        print(f"  训练损失: {avg_loss:.4f} (图像: {avg_loss_img:.4f}, 文本: {avg_loss_txt:.4f})")
        print(f"  验证损失: {avg_loss_val:.4f} (图像: {avg_loss_val_img:.4f}, 文本: {avg_loss_val_txt:.4f})")
        
        # 图像检索指标
        img_metrics = val_results['img_metrics']
        print(f"\n【图像检索效果】")
        print(f"  Top-1 准确率: {img_metrics['top_1_accuracy']*100:.2f}%")
        print(f"  Top-5 准确率: {img_metrics['top_5_accuracy']*100:.2f}%")
        print(f"  Top-10 准确率: {img_metrics['top_10_accuracy']*100:.2f}%")
        print(f"  正样本平均相似度: {img_metrics['mean_positive_similarity']:.4f}")
        print(f"  负样本平均相似度: {img_metrics['mean_negative_similarity']:.4f}")
        print(f"  分离比 (正/负): {img_metrics['separation_ratio']:.4f}")
        
        # 文本检索指标
        txt_metrics = val_results['txt_metrics']
        print(f"\n【文本检索效果】")
        print(f"  Top-1 准确率: {txt_metrics['top_1_accuracy']*100:.2f}%")
        print(f"  Top-5 准确率: {txt_metrics['top_5_accuracy']*100:.2f}%")
        print(f"  Top-10 准确率: {txt_metrics['top_10_accuracy']*100:.2f}%")
        print(f"  正样本平均相似度: {txt_metrics['mean_positive_similarity']:.4f}")
        print(f"  负样本平均相似度: {txt_metrics['mean_negative_similarity']:.4f}")
        print(f"  分离比 (正/负): {txt_metrics['separation_ratio']:.4f}")
        
        # 训练健康度检查
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"\n【训练健康度】")
            if img_metrics['separation_ratio'] > 1.5 and txt_metrics['separation_ratio'] > 1.5:
                print(f"  ✅ 分离比良好，模型正在学习区分正负样本")
            else:
                print(f"  ⚠️  分离比较低，可能需要调整学习率或检查数据")
            
            if img_metrics['top_1_accuracy'] > 0.1 or txt_metrics['top_1_accuracy'] > 0.1:
                print(f"  ✅ 检索准确率 > 10%，模型表现正常")
            else:
                print(f"  ⚠️  检索准确率较低，模型可能还在学习初期")
        
        print(f"{'='*70}\n")

        # 记录到WandB
        log_dict = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss_total": avg_loss,
            "train_loss_image": avg_loss_img,
            "train_loss_text": avg_loss_txt,
            "val_loss_total": avg_loss_val,
            "val_loss_image": avg_loss_val_img,
            "val_loss_text": avg_loss_val_txt,
            # 图像检索指标
            "val_img_top1": img_metrics['top_1_accuracy'],
            "val_img_top5": img_metrics['top_5_accuracy'],
            "val_img_top10": img_metrics['top_10_accuracy'],
            "val_img_pos_sim": img_metrics['mean_positive_similarity'],
            "val_img_neg_sim": img_metrics['mean_negative_similarity'],
            "val_img_separation": img_metrics['separation_ratio'],
            # 文本检索指标
            "val_txt_top1": txt_metrics['top_1_accuracy'],
            "val_txt_top5": txt_metrics['top_5_accuracy'],
            "val_txt_top10": txt_metrics['top_10_accuracy'],
            "val_txt_pos_sim": txt_metrics['mean_positive_similarity'],
            "val_txt_neg_sim": txt_metrics['mean_negative_similarity'],
            "val_txt_separation": txt_metrics['separation_ratio'],
        }
        if avg_weights:
            log_dict.update(avg_weights)
            print(f"\n【MoE权重分布】")
            print(f"  视觉专家权重: {avg_weights.get('w_vis_img', 0):.4f}")
            print(f"  语义专家权重: {avg_weights.get('w_sem_txt', 0):.4f}")
            
            # 检查权重是否正常（应该在0-1之间，且不应该极端）
            if avg_weights.get('w_vis_img', 0) < 0.1 or avg_weights.get('w_vis_img', 0) > 0.9:
                print(f"  ⚠️  视觉专家权重异常，MoE可能没有正常工作")
        
        wandb.log(log_dict)

        # 保存最佳模型 & 早停逻辑
        if (best_val_loss - avg_loss_val) > min_delta:
            best_val_loss = avg_loss_val
            model_path = os.path.join(wandb.run.dir, "best_eeg_encoder.pth")
            torch.save(model.state_dict(), model_path)
            print(f"模型已保存到: {model_path}")
            epochs_no_improve = 0 
        else:
            epochs_no_improve += 1 

        if epochs_no_improve >= patience:
            print(f"验证损失连续 {patience} 个 epoch 没有改善，触发 Early Stopping。")
            break 

    print("训练完成。")
    # 测试集评估
    print("【测试集评估】")
    avg_loss_test, avg_loss_test_img, avg_loss_test_txt = validate(
        model, test_loader, loss_fn_img, loss_fn_txt, device, cfg.training.alpha
    )

    print(f"Test Total Loss: {avg_loss_test:.4f}")
    
    wandb.log({
        "test_loss_total": avg_loss_test,
        "test_loss_image": avg_loss_test_img,
        "test_loss_text": avg_loss_test_txt
    })
    wandb.finish()


if __name__ == "__main__":
    main()