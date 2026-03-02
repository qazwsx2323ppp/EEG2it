# main.py - EEG-CLIP 训练脚本 (改进版)
# 改进内容: 信号处理、可选WandB、梯度累积、灵活模型选择、JSONL存储、可配置Backbone解冻、Sanity Check

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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 混合精度和调度器
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup

# 导入本地代码
from models.clip_models import SpatialMoEEncoder
from utils.loss_methods import ClipSoftmaxLoss, InfoNCE
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
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _append_jsonl(path: str, obj) -> None:
    """追加 JSONL 条目"""
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
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
        return float(val_results.get("txt_metrics", {}).get("top_1_accuracy", 0.0))
    if metric == "val/txt_top5":
        return float(val_results.get("txt_metrics", {}).get("top_5_accuracy", 0.0))
    if metric == "val/txt_top10":
        return float(val_results.get("txt_metrics", {}).get("top_10_accuracy", 0.0))
    if metric == "val/img_top1":
        return float(val_results.get("img_metrics", {}).get("top_1_accuracy", 0.0))
    if metric == "val/img_top5":
        return float(val_results.get("img_metrics", {}).get("top_5_accuracy", 0.0))
    if metric == "val/img_top10":
        return float(val_results.get("img_metrics", {}).get("top_10_accuracy", 0.0))
    return float(val_results["loss"])


def _format_topk(metrics: dict, k_values) -> str:
    parts = []
    for k in k_values:
        key = f"top_{int(k)}_accuracy"
        if key in metrics:
            parts.append(f"{metrics[key] * 100:.2f}%")
    return " / ".join(parts) if parts else "N/A"


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return x


def train_one_epoch(model, dataloader, optimizer, loss_fn_img, loss_fn_txt, device, alpha, scaler, scheduler, 
                    grad_accum_steps=1, sanity_check=False,
                    loss_mode: str = "infonce",
                    text_only: bool = False,
                    all_image_targets=None,
                    all_text_targets=None):
    """
    执行一个周期的训练 (支持 AMP、Scheduler、梯度累积和 Sanity Check)
    """
    model.train()
    total_loss = 0.0
    total_loss_img = 0.0
    total_loss_txt = 0.0
    steps = 0
    did_sanity = False

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
            
        target_ids = batch[3] if len(batch) >= 4 else None
        eeg_signals, image_vecs, text_vecs = batch[:3]

        eeg_signals = eeg_signals.to(device)
        image_vecs = image_vecs.to(device)
        text_vecs = text_vecs.to(device)

        # 混合精度前向传播
        with autocast():
            outputs = model(eeg_signals)
            
            if len(outputs) == 3:
                eeg_img_embeddings, eeg_text_embeddings, weights_info = outputs
                if weights_info:
                    for k, v in weights_info.items():
                        total_weights[k] += v.item()
            else:
                eeg_img_embeddings, eeg_text_embeddings = outputs
                weights_info = None

            # === Sanity Check: 首次batch检查数据和模型状态 ===
            if sanity_check and not did_sanity:
                with torch.no_grad():
                    eeg_mean = float(eeg_signals.mean().item())
                    eeg_std = float(eeg_signals.std().item())
                    print(f"\n[Sanity Check] EEG: mean={eeg_mean:.3f}, std={eeg_std:.3f}")
                    uniq_img = torch.unique(image_vecs, dim=0).shape[0]
                    uniq_txt = torch.unique(text_vecs, dim=0).shape[0]
                    bsz = image_vecs.shape[0]
                    print(f"[Sanity Check] Batch唯一向量: img={uniq_img}/{bsz}, txt={uniq_txt}/{bsz}")
                    img_norm = eeg_img_embeddings.norm(dim=-1).mean().item()
                    txt_norm = eeg_text_embeddings.norm(dim=-1).mean().item()
                    print(f"[Sanity Check] 输出模长: img={img_norm:.4f}, txt={txt_norm:.4f}")
                    if bsz >= 2:
                        sim = torch.nn.functional.cosine_similarity(
                            eeg_img_embeddings[0], eeg_img_embeddings[1], dim=0)
                        print(f"[Sanity Check] 样本间相似度: {sim.item():.4f} (接近1.0说明坍塌)")
                did_sanity = True

            # 计算损失
            loss_mode_l = str(loss_mode or "infonce").strip().lower()
            if loss_mode_l == "softmax_all":
                if target_ids is None:
                    raise ValueError("loss_mode=softmax_all requires target_id returned by the dataset (set data.return_target_id=true).")
                if all_text_targets is None:
                    raise ValueError("loss_mode=softmax_all requires all_text_targets.")
                loss_txt = loss_fn_txt(eeg_text_embeddings, all_text_targets, target_ids.to(device))
                if text_only:
                    loss_img = loss_txt.detach() * 0.0
                    loss = loss_txt
                else:
                    if all_image_targets is None:
                        raise ValueError("loss_mode=softmax_all (text_only=false) requires all_image_targets.")
                    loss_img = loss_fn_img(eeg_img_embeddings, all_image_targets, target_ids.to(device))
                    loss = (alpha * loss_img) + ((1 - alpha) * loss_txt)
            else:
                loss_txt = loss_fn_txt(eeg_text_embeddings, text_vecs)
                if text_only:
                    loss_img = loss_txt.detach() * 0.0
                    loss = loss_txt
                else:
                    loss_img = loss_fn_img(eeg_img_embeddings, image_vecs)
                    loss = (alpha * loss_img) + ((1 - alpha) * loss_txt)
            loss_to_backprop = loss / max(1, grad_accum_steps)

        # 混合精度反向传播 (支持梯度累积)
        scaler.scale(loss_to_backprop).backward()
        
        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()
        total_loss_img += loss_img.item()
        total_loss_txt += loss_txt.item()
        steps += 1

    denom = max(1, steps)
    avg_loss = total_loss / denom
    avg_loss_img = total_loss_img / denom
    avg_loss_txt = total_loss_txt / denom

    avg_weights = {}
    if total_weights["w_vis_img"] > 0: 
        for k in total_weights:
            avg_weights[k] = total_weights[k] / denom
            
    return avg_loss, avg_loss_img, avg_loss_txt, avg_weights


def compute_retrieval_metrics(
    eeg_embeddings,
    target_embeddings,
    k_values=[1, 5, 10],
    query_ids=None,
    candidate_ids=None,
    normalize: bool = True,
):
    """计算检索准确率（支持按 target_id 命中，避免重复目标时低估 Top-k）。"""
    if normalize:
        eeg_embeddings = F.normalize(eeg_embeddings, p=2, dim=-1)
        target_embeddings = F.normalize(target_embeddings, p=2, dim=-1)

    similarity_matrix = torch.matmul(eeg_embeddings, target_embeddings.T)
    _, sorted_indices = torch.sort(similarity_matrix, dim=1, descending=True)

    if query_ids is not None:
        qids = torch.as_tensor(query_ids, device=similarity_matrix.device, dtype=torch.long).view(-1)
        cids_src = candidate_ids if candidate_ids is not None else query_ids
        cids = torch.as_tensor(cids_src, device=similarity_matrix.device, dtype=torch.long).view(-1)
    else:
        qids = None
        cids = None
    
    metrics = {}
    for k in k_values:
        top_k_indices = sorted_indices[:, :k]
        if qids is None:
            correct_labels = torch.arange(similarity_matrix.shape[0], device=similarity_matrix.device)
            hits = (top_k_indices == correct_labels.unsqueeze(1)).any(dim=1)
        else:
            hits = (cids[top_k_indices] == qids.unsqueeze(1)).any(dim=1)
        accuracy = hits.float().mean().item()
        metrics[f"top_{k}_accuracy"] = accuracy
    
    if qids is not None:
        pos_mask = (qids[:, None] == cids[None, :])
        pos_vals = similarity_matrix[pos_mask]
        positive_sim = pos_vals.mean().item() if pos_vals.numel() > 0 else float("nan")
        neg_vals = similarity_matrix[~pos_mask]
        negative_sim = neg_vals.mean().item() if neg_vals.numel() > 0 else float("nan")
    else:
        diag = torch.diagonal(similarity_matrix, 0)
        positive_sim = diag.mean().item() if diag.numel() > 0 else float("nan")
        if similarity_matrix.shape[0] == similarity_matrix.shape[1]:
            mask = ~torch.eye(similarity_matrix.shape[0], dtype=torch.bool, device=similarity_matrix.device)
            negative_sim = similarity_matrix[mask].mean().item()
        else:
            negative_sim = float("nan")
    
    metrics['mean_positive_similarity'] = positive_sim
    metrics['mean_negative_similarity'] = negative_sim
    metrics['separation_ratio'] = positive_sim / (negative_sim + 1e-8)
    
    return metrics


def enhanced_validate(
    model,
    dataloader,
    loss_fn_img,
    loss_fn_txt,
    device,
    alpha,
    k_values=None,
    loss_mode: str = "infonce",
    text_only: bool = False,
    all_image_targets=None,
    all_text_targets=None,
):
    """增强的验证函数，包含检索指标"""
    model.eval()
    total_loss_val = 0.0
    total_loss_val_img = 0.0
    total_loss_val_txt = 0.0
    
    all_eeg_img_embeddings = []
    all_eeg_txt_embeddings = []
    all_target_img_embeddings = []
    all_target_txt_embeddings = []
    all_target_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            target_ids = batch[3] if len(batch) >= 4 else None
            eeg_signals, image_vecs, text_vecs = batch[:3]
            
            eeg_signals = eeg_signals.to(device)
            image_vecs = image_vecs.to(device)
            text_vecs = text_vecs.to(device)
            
            outputs = model(eeg_signals)
            
            if len(outputs) == 3:
                eeg_img_embedding, eeg_txt_embedding, _ = outputs
            else:
                eeg_img_embedding, eeg_txt_embedding = outputs
            
            loss_mode_l = str(loss_mode or "infonce").strip().lower()
            if loss_mode_l == "softmax_all":
                if target_ids is None:
                    raise ValueError("loss_mode=softmax_all requires target_id returned by the dataset (set data.return_target_id=true).")
                if all_text_targets is None:
                    raise ValueError("loss_mode=softmax_all requires all_text_targets.")
                loss_txt = loss_fn_txt(eeg_txt_embedding, all_text_targets, target_ids.to(device))
                if text_only:
                    loss_img = loss_txt.detach() * 0.0
                    loss = loss_txt
                else:
                    if all_image_targets is None:
                        raise ValueError("loss_mode=softmax_all (text_only=false) requires all_image_targets.")
                    loss_img = loss_fn_img(eeg_img_embedding, all_image_targets, target_ids.to(device))
                    loss = (alpha * loss_img) + ((1 - alpha) * loss_txt)
            else:
                loss_txt = loss_fn_txt(eeg_txt_embedding, text_vecs)
                if text_only:
                    loss_img = loss_txt.detach() * 0.0
                    loss = loss_txt
                else:
                    loss_img = loss_fn_img(eeg_img_embedding, image_vecs)
                    loss = (alpha * loss_img) + ((1 - alpha) * loss_txt)
            
            total_loss_val += loss.item()
            total_loss_val_img += loss_img.item()
            total_loss_val_txt += loss_txt.item()
            
            all_eeg_img_embeddings.append(eeg_img_embedding.cpu())
            all_eeg_txt_embeddings.append(eeg_txt_embedding.cpu())
            all_target_img_embeddings.append(image_vecs.cpu())
            all_target_txt_embeddings.append(text_vecs.cpu())
            if target_ids is not None:
                all_target_ids.append(target_ids.cpu())
    
    all_eeg_img = torch.cat(all_eeg_img_embeddings, dim=0).to(device)
    all_eeg_txt = torch.cat(all_eeg_txt_embeddings, dim=0).to(device)
    all_target_img = torch.cat(all_target_img_embeddings, dim=0).to(device)
    all_target_txt = torch.cat(all_target_txt_embeddings, dim=0).to(device)
    target_ids_tensor = torch.cat(all_target_ids, dim=0).to(device) if all_target_ids else None
    
    if k_values is None:
        k_values = [1, 5, 10]

    img_metrics = {}
    if not text_only:
        img_metrics = compute_retrieval_metrics(
            all_eeg_img,
            all_target_img,
            k_values=k_values,
            query_ids=target_ids_tensor,
            candidate_ids=target_ids_tensor,
        )
    txt_metrics = compute_retrieval_metrics(
        all_eeg_txt,
        all_target_txt,
        k_values=k_values,
        query_ids=target_ids_tensor,
        candidate_ids=target_ids_tensor,
    )
    
    avg_loss_val = total_loss_val / len(dataloader)
    avg_loss_val_img = total_loss_val_img / len(dataloader)
    avg_loss_val_txt = total_loss_val_txt / len(dataloader)
    
    return {
        'loss': avg_loss_val,
        'loss_img': avg_loss_val_img,
        'loss_txt': avg_loss_val_txt,
        'img_metrics': img_metrics or {},
        'txt_metrics': txt_metrics
    }


def check_parameter_updates(model):
    """检查参数是否在更新"""
    trainable_params = []
    frozen_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append((name, param.numel()))
        else:
            frozen_params.append((name, param.numel()))
    
    total_trainable = sum(p[1] for p in trainable_params)
    total_frozen = sum(p[1] for p in frozen_params)
    
    print(f"\n[参数更新检查]")
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
    """在验证集上评估模型"""
    model.eval()
    total_loss_val = 0.0
    total_loss_val_img = 0.0
    total_loss_val_txt = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            eeg_signals, image_vecs, text_vecs = batch[:3]

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

    avg_loss_val = total_loss_val / len(dataloader)
    avg_loss_val_img = total_loss_val_img / len(dataloader)
    avg_loss_val_txt = total_loss_val_txt / len(dataloader)

    return avg_loss_val, avg_loss_val_img, avg_loss_val_txt


@hydra.main(version_base=None, config_path="configs", config_name="triplet_config")
def main(cfg: DictConfig):
    # 安装信号处理器
    _install_signal_handlers()
    
    print("Hydra 配置:\n", OmegaConf.to_yaml(cfg))
    
    # 获取运行目录和指标文件路径
    run_dir = os.getcwd()
    metrics_jsonl = os.path.join(run_dir, "metrics_epoch.jsonl")
    metrics_summary_path = os.path.join(run_dir, "metrics_summary.json")

    # 可选 WandB 初始化
    use_wandb = bool(cfg.training.get("use_wandb", True))
    wandb = _maybe_init_wandb(cfg, enabled=use_wandb)

    device = torch.device(cfg.training.device)

    print(f"正在初始化模型: SpatialMoEEncoder (Backbone: DreamDiffusion)")

    # 实例化模型
    model = SpatialMoEEncoder(
        n_channels=cfg.model.n_channels,
        n_samples=cfg.model.n_samples,
        embedding_dim=cfg.model.embedding_dim,
        pretrained_path=cfg.model.get("pretrained_path", None),
        router_mode=cfg.model.get("router_mode", "moe"),
        head_dropout=cfg.model.get("head_dropout", 0.5),
    ).to(device)
        
    print(">>> 成功初始化 Spatial MoE Encoder")

    # 准备数据
    split_index = cfg.data.get("split_index", 0)

    train_dataset = TripletDataset(cfg.data, mode='train', split_index=split_index)
    val_dataset = TripletDataset(cfg.data, mode='val', split_index=split_index)
    test_dataset = TripletDataset(cfg.data, mode='test', split_index=split_index)

    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True,
                              num_workers=cfg.training.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False,
                            num_workers=cfg.training.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False,
                             num_workers=cfg.training.num_workers, pin_memory=True)

    loss_mode = str(cfg.training.get("loss_mode", "infonce")).strip().lower()
    text_only = bool(cfg.training.get("text_only", False))
    if text_only:
        print("[配置] text_only=true：将以文本分支为主（仅优化 EEG→Text 对齐）。")

    # 初始化损失函数
    if loss_mode == "softmax_all":
        loss_fn_txt = ClipSoftmaxLoss(initial_temperature=cfg.training.temperature).to(device)
        loss_fn_img = None if text_only else ClipSoftmaxLoss(initial_temperature=cfg.training.temperature).to(device)
    else:
        loss_fn_txt = InfoNCE(initial_temperature=cfg.training.temperature).to(device)
        loss_fn_img = None if text_only else InfoNCE(initial_temperature=cfg.training.temperature).to(device)

    # 参数分组 - 可配置的 Backbone 解冻
    params_backbone_active = []
    params_head = []
    loss_params = list(loss_fn_txt.parameters())
    if loss_fn_img is not None:
        loss_params += list(loss_fn_img.parameters())
    
    unfreeze_last_blocks = int(cfg.training.get("unfreeze_last_blocks", 2))
    unfreeze_patch_embed = bool(cfg.training.get("unfreeze_patch_embed", False))
    
    # 获取 backbone 深度
    try:
        depth = len(model.backbone.blocks) if hasattr(model.backbone, 'blocks') else 24
    except:
        depth = 24
    first_unfrozen = max(0, depth - unfreeze_last_blocks)

    for name, param in model.named_parameters():
        if "backbone" in name:
            # 判断是否解冻 patch_embed
            if unfreeze_patch_embed and ("patch_embed" in name or "pos_embed" in name):
                params_backbone_active.append(param)
                param.requires_grad = True
                continue
            
            # 解冻最后 N 层
            blk_idx = None
            try:
                if "blocks." in name:
                    blk_idx = int(name.split("blocks.", 1)[1].split(".", 1)[0])
            except:
                blk_idx = None

            if "norm." in name or (blk_idx is not None and blk_idx >= first_unfrozen):
                params_backbone_active.append(param)
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            params_head.append(param)
            param.requires_grad = True

    # 优化器
    optimizer = optim.AdamW(
        [
            {"params": params_head, "lr": cfg.training.learning_rate}, 
            {"params": params_backbone_active, "lr": cfg.training.learning_rate * 0.1}, 
            {"params": loss_params, "lr": cfg.training.learning_rate, "weight_decay": 0.0},
        ],
        weight_decay=float(cfg.training.get("weight_decay", 0.15))
    )

    # 梯度累积配置
    grad_accum_steps = int(cfg.training.get("grad_accum_steps", 1))
    
    # 学习率调度器 (考虑梯度累积)
    effective_steps_per_epoch = len(train_loader) // grad_accum_steps
    num_training_steps = cfg.training.epochs * effective_steps_per_epoch
    num_warmup_steps = int(float(cfg.training.get("warmup_ratio", 0.1)) * num_training_steps)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    scaler = GradScaler()
    param_info = check_parameter_updates(model)

    if wandb is not None:
        wandb.log({
            "trainable_params": param_info['trainable_count'],
            "frozen_params": param_info['frozen_count'],
            "trainable_ratio": param_info['trainable_ratio']
        })

    # 训练配置
    print("开始训练...")

    k_values = list(cfg.training.get("k_values", [1, 5, 10]))

    # For softmax_all, keep full target matrices on device for efficient logits.
    all_image_targets = None
    all_text_targets = None
    if loss_mode == "softmax_all":
        all_text_targets = getattr(train_loader.dataset, "all_text_vectors", None)
        if all_text_targets is None:
            raise ValueError("loss_mode=softmax_all requires dataset.all_text_vectors")
        all_text_targets = all_text_targets.to(device)
        if not text_only:
            all_image_targets = getattr(train_loader.dataset, "all_image_vectors", None)
            if all_image_targets is None:
                raise ValueError("loss_mode=softmax_all requires dataset.all_image_vectors")
            all_image_targets = all_image_targets.to(device)
    
    # 模型选择配置
    selection_metric = str(cfg.training.get("selection_metric", "val/loss_total"))
    selection_mode = str(cfg.training.get("selection_mode", "min")).lower()
    best_score = float("-inf") if selection_mode == "max" else float("inf")
    best_epoch = -1
    best_val_snapshot = None
    best_model_path = None
    
    patience = cfg.training.get("patience", cfg.training.epochs) 
    min_delta = cfg.training.get("min_delta", 0.0)
    epochs_no_improve = 0
    sanity_check = bool(cfg.training.get("sanity_check", True))

    for epoch in range(cfg.training.epochs):
        avg_loss, avg_loss_img, avg_loss_txt, avg_weights = train_one_epoch(
            model, train_loader, optimizer, loss_fn_img, loss_fn_txt, device, 
            cfg.training.alpha, scaler, scheduler, 
            grad_accum_steps=grad_accum_steps,
            sanity_check=(sanity_check and epoch == 0),
            loss_mode=loss_mode,
            text_only=text_only,
            all_image_targets=all_image_targets,
            all_text_targets=all_text_targets,
        )

        val_results = enhanced_validate(
            model,
            val_loader,
            loss_fn_img,
            loss_fn_txt,
            device,
            cfg.training.alpha,
            k_values=k_values,
            loss_mode=loss_mode,
            text_only=text_only,
            all_image_targets=all_image_targets,
            all_text_targets=all_text_targets,
        )

        avg_loss_val = val_results['loss']
        avg_loss_val_img = val_results['loss_img']
        avg_loss_val_txt = val_results['loss_txt']
        img_metrics = val_results['img_metrics']
        txt_metrics = val_results['txt_metrics']

        current_lr = optimizer.param_groups[0]["lr"]

        # 详细的训练效果展示
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{cfg.training.epochs} | LR: {current_lr:.6f}")
        print(f"{'='*70}")
        
        print(f"\n[图像检索效果]")
        if img_metrics:
            print(f"  Top-{('/'.join(str(int(k)) for k in k_values))}: {_format_topk(img_metrics, k_values)}")
            print(f"  正/负相似度: {img_metrics.get('mean_positive_similarity', float('nan')):.4f} / {img_metrics.get('mean_negative_similarity', float('nan')):.4f}")
            print(f"  分离比(正/负): {img_metrics.get('separation_ratio', float('nan')):.4f}")
        else:
            print("  (text_only=true 已跳过图像检索指标)")
        
        print(f"\n[文本检索效果]")
        print(f"  Top-{('/'.join(str(int(k)) for k in k_values))}: {_format_topk(txt_metrics, k_values)}")
        print(f"  正/负相似度: {txt_metrics.get('mean_positive_similarity', float('nan')):.4f} / {txt_metrics.get('mean_negative_similarity', float('nan')):.4f}")
        print(f"  分离比(正/负): {txt_metrics.get('separation_ratio', float('nan')):.4f}")

        print(f"\n[损失指标]")
        print(f"  训练损失: {avg_loss:.4f} (图像: {avg_loss_img:.4f}, 文本: {avg_loss_txt:.4f})")
        print(f"  验证损失: {avg_loss_val:.4f} (图像: {avg_loss_val_img:.4f}, 文本: {avg_loss_val_txt:.4f})")
        
        print(f"{'='*70}\n")

        # 保存指标到 JSONL
        epoch_metrics = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "epoch": epoch,
            "train": {"loss_total": avg_loss, "loss_image": avg_loss_img, "loss_text": avg_loss_txt},
            "val": {
                "loss_total": avg_loss_val, "loss_image": avg_loss_val_img, "loss_text": avg_loss_val_txt,
                "img_metrics": img_metrics, "txt_metrics": txt_metrics
            },
            "lr": current_lr
        }
        _append_jsonl(metrics_jsonl, epoch_metrics)

        # 记录到 WandB
        log_dict = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss_total": avg_loss,
            "train_loss_image": avg_loss_img,
            "train_loss_text": avg_loss_txt,
            "val_loss_total": avg_loss_val,
            "val_loss_image": avg_loss_val_img,
            "val_loss_text": avg_loss_val_txt,
            "val_img_top1": (img_metrics.get('top_1_accuracy') if img_metrics else None),
            "val_img_top5": (img_metrics.get('top_5_accuracy') if img_metrics else None),
            "val_txt_top1": txt_metrics.get('top_1_accuracy'),
            "val_txt_top5": txt_metrics.get('top_5_accuracy'),
        }
        if avg_weights:
            log_dict.update(avg_weights)
        
        if wandb is not None:
            wandb.log(log_dict)

        # 模型选择和早停
        score = _get_metric_from_val(val_results, selection_metric)
        improved = (score > best_score + min_delta) if selection_mode == "max" else (score < best_score - min_delta)
        
        if improved:
            best_score = score
            best_epoch = epoch
            if wandb is not None:
                model_path = os.path.join(wandb.run.dir, "best_eeg_encoder.pth")
            else:
                model_path = os.path.join(run_dir, "best_eeg_encoder.pth")
            torch.save(model.state_dict(), model_path)
            best_model_path = model_path
            best_val_snapshot = {
                "epoch": int(epoch),
                "learning_rate": _safe_float(current_lr),
                "train": {"loss_total": _safe_float(avg_loss), "loss_image": _safe_float(avg_loss_img), "loss_text": _safe_float(avg_loss_txt)},
                "val": {
                    "loss_total": _safe_float(avg_loss_val),
                    "loss_image": _safe_float(avg_loss_val_img),
                    "loss_text": _safe_float(avg_loss_val_txt),
                    "img_metrics": {k: _safe_float(v) for k, v in dict(img_metrics).items()},
                    "txt_metrics": {k: _safe_float(v) for k, v in dict(txt_metrics).items()},
                },
                "selection_metric": str(selection_metric),
                "selection_mode": str(selection_mode),
                "selection_score": _safe_float(score),
                "k_values": [int(k) for k in k_values],
            }
            print(f"模型已保存到: {model_path} ({selection_metric}={score:.6f})")
            epochs_no_improve = 0 
        else:
            epochs_no_improve += 1 

        if epochs_no_improve >= patience:
            print(f"验证指标连续 {patience} 个 epoch 没有改善，触发 Early Stopping。（best {selection_metric}={best_score:.6f}）")
            break

        # 检查是否请求停止
        if _stop_requested():
            print("[信号] 检测到停止请求，保存当前状态并退出...")
            break

    # 保存摘要
    summary = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "selection_metric": selection_metric,
        "selection_mode": selection_mode,
        "k_values": [int(k) for k in k_values],
        "best_model_path": best_model_path,
        "best_val": best_val_snapshot,
        "run_dir": run_dir,
    }
    _write_json(metrics_summary_path, summary)

    print("训练完成。")
    if best_val_snapshot is not None:
        bm = best_val_snapshot["val"].get("img_metrics", {}) or {}
        tm = best_val_snapshot["val"]["txt_metrics"]
        print("\n[最佳验证集指标汇总]")
        print(f"  best_epoch: {best_epoch + 1} | {selection_metric}={best_score:.6f}")
        print(f"  best_model: {best_model_path}")
        print("\n  [图像 Top-k]")
        if bm:
            print(f"    Top-{('/'.join(str(int(k)) for k in k_values))}: {_format_topk(bm, k_values)}")
            print(
                f"    正/负相似度: {bm.get('mean_positive_similarity', float('nan')):.4f} / {bm.get('mean_negative_similarity', float('nan')):.4f}"
                f" | 分离比: {bm.get('separation_ratio', float('nan')):.4f}"
            )
        else:
            print("    (text_only=true 已跳过图像检索指标)")
        print("\n  [文本 Top-k]")
        print(f"    Top-{('/'.join(str(int(k)) for k in k_values))}: {_format_topk(tm, k_values)}")
        print(
            f"    正/负相似度: {tm.get('mean_positive_similarity', float('nan')):.4f} / {tm.get('mean_negative_similarity', float('nan')):.4f}"
            f" | 分离比: {tm.get('separation_ratio', float('nan')):.4f}"
        )
        print("\n  [Loss（辅助）]")
        print(f"    val_loss_total: {best_val_snapshot['val']['loss_total']:.4f} (img: {best_val_snapshot['val']['loss_image']:.4f}, txt: {best_val_snapshot['val']['loss_text']:.4f})")
    
    # 测试集评估
    print("[测试集评估]")
    if best_model_path and os.path.isfile(best_model_path):
        try:
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print(f"[测试集评估] 已加载最佳模型权重: {best_model_path}")
        except Exception as e:
            print(f"[测试集评估] 警告：加载最佳模型失败，将使用当前模型参数继续测试：{e}")
    test_results = enhanced_validate(
        model,
        test_loader,
        loss_fn_img,
        loss_fn_txt,
        device,
        cfg.training.alpha,
        k_values=k_values,
        loss_mode=loss_mode,
        text_only=text_only,
        all_image_targets=all_image_targets,
        all_text_targets=all_text_targets,
    )
    test_img = test_results["img_metrics"]
    test_txt = test_results["txt_metrics"]
    print("\n[测试集 Top-k]")
    print(f"  [图像] Top-{('/'.join(str(int(k)) for k in k_values))}: {_format_topk(test_img, k_values)}")
    print(f"  [文本] Top-{('/'.join(str(int(k)) for k in k_values))}: {_format_topk(test_txt, k_values)}")
    print("\n[测试集 Loss（辅助）]")
    print(f"  test_loss_total: {test_results['loss']:.4f} (img: {test_results['loss_img']:.4f}, txt: {test_results['loss_txt']:.4f})")
    
    if wandb is not None:
        wandb.log({
            "test_loss_total": test_results["loss"],
            "test_loss_image": test_results["loss_img"],
            "test_loss_text": test_results["loss_txt"],
            "test_img_top1": test_img.get("top_1_accuracy"),
            "test_img_top5": test_img.get("top_5_accuracy"),
            "test_txt_top1": test_txt.get("top_1_accuracy"),
            "test_txt_top5": test_txt.get("top_5_accuracy"),
        })
        wandb.finish()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[退出] 用户中断训练")
        sys.exit(130)
