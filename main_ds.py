# Ignore compatibility warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import os
import json
import signal
import sys
from datetime import datetime

import hydra
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from dataset import TripletDataset
from models.clip_models import SpatialMoEEncoder
from utils.loss_methods import InfoNCE
from utils.batch_samplers import RepeatBatchSampler, UniqueConceptBatchSampler


def _get_dist_env():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world_size, rank, local_rank


def _dist_enabled(cfg: DictConfig) -> bool:
    if bool(cfg.training.get("distributed", {}).get("enabled", False)):
        return True
    world_size, _, _ = _get_dist_env()
    return world_size > 1


def _dist_setup(cfg: DictConfig):
    import torch.distributed as dist

    world_size, rank, local_rank = _get_dist_env()
    backend = cfg.training.get("distributed", {}).get("backend", "nccl")

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this build.")

    if dist.is_initialized():
        return torch.device("cuda", local_rank), rank, world_size

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    return torch.device("cuda", local_rank), rank, world_size


def _dist_cleanup():
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _dist_abort_best_effort():
    """
    Best-effort hard stop for DDP. Helpful when CTRL+C happens mid-iteration and
    some ranks might otherwise hang in collectives.
    """
    try:
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return
        try:
            # torch>=2.0 (backend dependent)
            dist.abort()
        except Exception:
            pass
    except Exception:
        pass


def _is_rank0(rank: int) -> bool:
    return rank == 0


_STOP_REQUESTED = False


def _install_signal_handlers():
    """
    Install handlers that:
    - keep default SIGINT -> KeyboardInterrupt behavior (CTRL+C should stop immediately)
    - still set a shared stop flag so loops can exit cleanly
    """
    prev_int = None
    try:
        prev_int = signal.getsignal(signal.SIGINT)
    except Exception:
        prev_int = None

    def _handler_int(signum, frame):
        global _STOP_REQUESTED
        _STOP_REQUESTED = True
        if callable(prev_int):
            prev_int(signum, frame)  # likely raises KeyboardInterrupt
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


def _dist_any_stop(device) -> bool:
    """
    If distributed is initialized, propagate stop/interrupt across ranks.
    Returns True if any rank requested stop.
    """
    try:
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return _stop_requested()
        flag = torch.tensor([1 if _stop_requested() else 0], device=device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.SUM)
        return bool(flag.item() > 0)
    except Exception:
        return _stop_requested()


def _maybe_init_wandb(cfg: DictConfig, enabled: bool, is_rank0: bool):
    if not (enabled and is_rank0):
        return None
    try:
        import wandb  # type: ignore

        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        return wandb
    except Exception as e:
        print(f"[warn] wandb init failed, continuing without wandb: {e}")
        return None


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _append_jsonl(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _to_device(batch, device):
    if len(batch) == 3:
        eeg_signals, image_vecs, text_vecs = batch
        concept_ids = None
    elif len(batch) == 4:
        eeg_signals, image_vecs, text_vecs, concept_ids = batch
    else:
        raise ValueError(f"Unexpected batch size: {len(batch)}")

    eeg_signals = eeg_signals.to(device, non_blocking=True)
    image_vecs = image_vecs.to(device, non_blocking=True)
    text_vecs = text_vecs.to(device, non_blocking=True)
    if concept_ids is not None:
        concept_ids = torch.as_tensor(concept_ids, device=device)
    return eeg_signals, image_vecs, text_vecs, concept_ids


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    loss_fn_img,
    loss_fn_txt,
    device,
    alpha: float,
    scaler,
    scheduler,
    grad_accum_steps: int = 1,
    max_steps: int | None = None,
    log_every: int = 50,
    sanity_check: bool = False,
    sanity_check_once: bool = True,
    rank: int = 0,
):
    model.train()
    total_loss = 0.0
    total_loss_img = 0.0
    total_loss_txt = 0.0

    optimizer.zero_grad(set_to_none=True)

    use_cuda_amp = device.type == "cuda"
    autocast_ctx = torch.amp.autocast("cuda", enabled=use_cuda_amp)

    step = 0
    opt_step = 0
    did_sanity = False
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training", disable=rank != 0)):
        if _dist_any_stop(device):
            raise KeyboardInterrupt()
        eeg_signals, image_vecs, text_vecs, concept_ids = _to_device(batch, device)

        with autocast_ctx:
            outputs = model(eeg_signals)
            if len(outputs) == 3:
                eeg_img_embeddings, eeg_text_embeddings, _ = outputs
            else:
                eeg_img_embeddings, eeg_text_embeddings = outputs

            if loss_fn_img is not None and alpha > 0:
                loss_img = loss_fn_img(eeg_img_embeddings, image_vecs)
            else:
                loss_img = torch.tensor(0.0, device=device)
            loss_txt = loss_fn_txt(eeg_text_embeddings, text_vecs)
            loss = (alpha * loss_img) + ((1 - alpha) * loss_txt)

            loss_to_backprop = loss / max(1, grad_accum_steps)

        if sanity_check and not did_sanity and _is_rank0(rank):
            # Quick sanity checks for prompt-supervision training:
            # 1) image/text targets shouldn't be identical (cosine < 1.0)
            # 2) batch targets should be unique for InfoNCE (unique count ~= batch_size)
            with torch.no_grad():
                cos = torch.nn.functional.cosine_similarity(image_vecs, text_vecs, dim=-1).mean().item()
                uniq_img = torch.unique(image_vecs, dim=0).shape[0]
                uniq_txt = torch.unique(text_vecs, dim=0).shape[0]
                bsz = int(image_vecs.shape[0])
                msg = f"[sanity] cos(image_vec, text_vec) mean={cos:.4f} | unique image targets={uniq_img}/{bsz} | unique text targets={uniq_txt}/{bsz}"
                if concept_ids is not None:
                    uniq_c = int(torch.unique(concept_ids).numel())
                    msg += f" | unique concept_ids={uniq_c}/{bsz}"
                print(msg)

                # EEG scale checks: if values are near-constant or extremely small, training can stall near ln(batch).
                eeg_mean = float(eeg_signals.mean().item())
                eeg_std = float(eeg_signals.std(unbiased=False).item())
                eeg_min = float(eeg_signals.min().item())
                eeg_max = float(eeg_signals.max().item())
                zero_frac = float((eeg_signals.abs() < 1e-12).float().mean().item())
                print(
                    f"[sanity] eeg mean={eeg_mean:.3e} std={eeg_std:.3e} min={eeg_min:.3e} max={eeg_max:.3e} | near_zero={zero_frac*100:.2f}%"
                )

                # Embedding scale checks: norms should be reasonable and logits should have diagonal > off-diagonal.
                txt_norm = float(text_vecs.norm(dim=-1).mean().item())
                eeg_norm = float(eeg_text_embeddings.norm(dim=-1).mean().item())
                logits = eeg_text_embeddings @ text_vecs.T
                diag = float(torch.diag(logits).mean().item())
                off = float((logits.sum() - torch.diag(logits).sum()).div(max(1, logits.numel() - logits.shape[0])).item())
                print(f"[sanity] norms: eeg_text={eeg_norm:.3f} text={txt_norm:.3f} | logits diag={diag:.3f} off={off:.3f}")
            did_sanity = True
            if sanity_check_once:
                sanity_check = False

        scaler.scale(loss_to_backprop).backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1
            if scheduler is not None:
                scheduler.step()

        total_loss += float(loss.detach().cpu())
        total_loss_img += float(loss_img.detach().cpu())
        total_loss_txt += float(loss_txt.detach().cpu())

        step += 1
        # metrics logging handled at epoch-level (and optional wandb in main)

        if max_steps is not None and step >= max_steps:
            break

    denom = max(1, step)
    return (total_loss / denom), (total_loss_img / denom), (total_loss_txt / denom), step


@torch.no_grad()
def validate_loss_only(model, dataloader, loss_fn_img, loss_fn_txt, device, alpha: float, max_steps: int | None = None, rank: int = 0):
    model.eval()
    total_loss = 0.0
    total_loss_img = 0.0
    total_loss_txt = 0.0
    steps = 0

    for batch in tqdm(dataloader, desc="Validation", disable=rank != 0):
        if _dist_any_stop(device):
            raise KeyboardInterrupt()
        eeg_signals, image_vecs, text_vecs, _ = _to_device(batch, device)
        outputs = model(eeg_signals)
        if len(outputs) == 3:
            eeg_img_embeddings, eeg_text_embeddings, _ = outputs
        else:
            eeg_img_embeddings, eeg_text_embeddings = outputs

        if loss_fn_img is not None and alpha > 0:
            loss_img = loss_fn_img(eeg_img_embeddings, image_vecs)
        else:
            loss_img = torch.tensor(0.0, device=device)
        loss_txt = loss_fn_txt(eeg_text_embeddings, text_vecs)
        loss = (alpha * loss_img) + ((1 - alpha) * loss_txt)

        total_loss += float(loss.detach().cpu())
        total_loss_img += float(loss_img.detach().cpu())
        total_loss_txt += float(loss_txt.detach().cpu())
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break

    denom = max(1, steps)
    return (total_loss / denom), (total_loss_img / denom), (total_loss_txt / denom), steps


def compute_retrieval_metrics(similarity_matrix: torch.Tensor, k_values=(1, 5, 10)) -> dict:
    # similarity_matrix: [N, N] with correct pair on diagonal
    _, sorted_indices = torch.sort(similarity_matrix, dim=1, descending=True)
    correct_labels = torch.arange(similarity_matrix.shape[0], device=similarity_matrix.device)
    metrics = {}
    for k in k_values:
        top_k_indices = sorted_indices[:, :k]
        hits = (top_k_indices == correct_labels.unsqueeze(1)).any(dim=1)
        metrics[f"top_{k}_accuracy"] = hits.float().mean().item()

    positive_sim = torch.diag(similarity_matrix).mean().item()
    mask = ~torch.eye(similarity_matrix.shape[0], dtype=torch.bool, device=similarity_matrix.device)
    negative_sim = similarity_matrix[mask].mean().item()
    metrics["mean_positive_similarity"] = positive_sim
    metrics["mean_negative_similarity"] = negative_sim
    metrics["separation_ratio"] = positive_sim / (negative_sim + 1e-8)
    return metrics


@torch.no_grad()
def evaluate_concept_retrieval(
    model,
    dataloader,
    loss_fn_img,
    loss_fn_txt,
    device,
    alpha: float,
    max_steps: int | None,
    num_concepts: int,
    embedding_dim: int,
    distributed: bool,
    rank: int,
    k_values=(1, 5, 10),
):
    """
    Evaluate by averaging EEG embeddings per concept id and computing retrieval against concept targets.
    This is more meaningful than InfoNCE loss when the dataset has repeated concepts.
    """
    model.eval()
    sums_img = torch.zeros((num_concepts, embedding_dim), device=device, dtype=torch.float32)
    sums_txt = torch.zeros((num_concepts, embedding_dim), device=device, dtype=torch.float32)
    counts = torch.zeros((num_concepts,), device=device, dtype=torch.float32)

    total_loss = torch.tensor(0.0, device=device)
    total_loss_img = torch.tensor(0.0, device=device)
    total_loss_txt = torch.tensor(0.0, device=device)
    steps = 0

    for batch in tqdm(dataloader, desc="Validation", disable=rank != 0):
        if _dist_any_stop(device):
            raise KeyboardInterrupt()
        eeg_signals, image_vecs, text_vecs, concept_ids = _to_device(batch, device)
        if concept_ids is None:
            raise RuntimeError("Concept retrieval eval requires data.return_concept_id=true")

        outputs = model(eeg_signals)
        if len(outputs) == 3:
            eeg_img_emb, eeg_txt_emb, _ = outputs
        else:
            eeg_img_emb, eeg_txt_emb = outputs

        loss_img = loss_fn_img(eeg_img_emb, image_vecs) if (loss_fn_img is not None and alpha > 0) else torch.tensor(0.0, device=device)
        loss_txt = loss_fn_txt(eeg_txt_emb, text_vecs)
        loss = (alpha * loss_img) + ((1 - alpha) * loss_txt)

        total_loss += loss.detach()
        total_loss_img += loss_img.detach()
        total_loss_txt += loss_txt.detach()

        concept_ids = concept_ids.long()
        if loss_fn_img is not None and alpha > 0:
            sums_img.index_add_(0, concept_ids, eeg_img_emb.float())
        sums_txt.index_add_(0, concept_ids, eeg_txt_emb.float())
        counts.index_add_(0, concept_ids, torch.ones_like(concept_ids, dtype=torch.float32))

        steps += 1
        if max_steps is not None and steps >= max_steps:
            break

    if distributed:
        import torch.distributed as dist

        if loss_fn_img is not None and alpha > 0:
            dist.all_reduce(sums_img, op=dist.ReduceOp.SUM)
        dist.all_reduce(sums_txt, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        if loss_fn_img is not None and alpha > 0:
            dist.all_reduce(total_loss_img, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_loss_txt, op=dist.ReduceOp.SUM)
        steps_t = torch.tensor(float(steps), device=device)
        dist.all_reduce(steps_t, op=dist.ReduceOp.SUM)
        steps = int(steps_t.item())

    denom_steps = max(1, steps)
    avg_loss = (total_loss / denom_steps).item()
    avg_loss_img = (total_loss_img / denom_steps).item()
    avg_loss_txt = (total_loss_txt / denom_steps).item()

    # Concept means and normalization
    counts_clamped = counts.clamp_min(1.0).unsqueeze(1)
    eeg_img_mean = sums_img / counts_clamped
    eeg_txt_mean = sums_txt / counts_clamped
    eeg_img_mean = eeg_img_mean / (eeg_img_mean.norm(dim=-1, keepdim=True) + 1e-12)
    eeg_txt_mean = eeg_txt_mean / (eeg_txt_mean.norm(dim=-1, keepdim=True) + 1e-12)

    # Use the dataset's concept targets
    ds = dataloader.dataset
    target_img = ds.all_image_vectors.to(device)
    target_txt = ds.all_text_vectors.to(device)
    target_img = target_img / (target_img.norm(dim=-1, keepdim=True) + 1e-12)
    target_txt = target_txt / (target_txt.norm(dim=-1, keepdim=True) + 1e-12)

    sim_img = (eeg_img_mean @ target_img.T) if (loss_fn_img is not None and alpha > 0) else None
    sim_txt = eeg_txt_mean @ target_txt.T

    # Evaluate retrieval only over concepts actually present in this split.
    present_mask = counts > 0
    present_idx = present_mask.nonzero(as_tuple=False).squeeze(1)
    present = int(present_idx.numel())

    if present >= 2:
        sim_txt_eval = sim_txt[present_idx][:, present_idx]
        txt_metrics = compute_retrieval_metrics(sim_txt_eval, k_values=k_values)

        if sim_img is not None:
            sim_img_eval = sim_img[present_idx][:, present_idx]
            img_metrics = compute_retrieval_metrics(sim_img_eval, k_values=k_values)
        else:
            img_metrics = None
    else:
        # Not enough concepts for meaningful retrieval. Report zeros.
        txt_metrics = {f"top_{k}_accuracy": 0.0 for k in k_values}
        txt_metrics.update({"mean_positive_similarity": 0.0, "mean_negative_similarity": 0.0, "separation_ratio": 0.0})
        img_metrics = None
    return {
        "loss": avg_loss,
        "loss_img": avg_loss_img,
        "loss_txt": avg_loss_txt,
        "img_metrics": img_metrics,
        "txt_metrics": txt_metrics,
        "present_concepts": present,
        "steps": steps,
    }


def _get_metric_from_val(val_results: dict, metric: str) -> float:
    metric = metric.strip()
    if metric in {"val/loss_total", "loss", "val_loss"}:
        return float(val_results["loss"])
    if metric == "val/txt_top1":
        return float(val_results["txt_metrics"]["top_1_accuracy"])
    if metric == "val/txt_top5":
        return float(val_results["txt_metrics"]["top_5_accuracy"])
    if metric == "val/txt_top10":
        return float(val_results["txt_metrics"]["top_10_accuracy"])
    if metric == "val/img_top1":
        if val_results.get("img_metrics") is None:
            return float("-inf")
        return float(val_results["img_metrics"]["top_1_accuracy"])
    if metric == "val/img_top5":
        if val_results.get("img_metrics") is None:
            return float("-inf")
        return float(val_results["img_metrics"]["top_5_accuracy"])
    if metric == "val/img_top10":
        if val_results.get("img_metrics") is None:
            return float("-inf")
        return float(val_results["img_metrics"]["top_10_accuracy"])
    raise ValueError(f"Unknown metric: {metric}")


def check_parameter_updates(model):
    trainable_params = []
    frozen_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append((name, param.numel()))
        else:
            frozen_params.append((name, param.numel()))

    total_trainable = sum(p[1] for p in trainable_params)
    total_frozen = sum(p[1] for p in frozen_params)

    print("\n【参数更新检查】")
    print(f"可训练参数: {total_trainable:,} ({total_trainable/(total_trainable+total_frozen)*100:.1f}%)")
    print(f"冻结参数: {total_frozen:,} ({total_frozen/(total_trainable+total_frozen)*100:.1f}%)")
    print("\n可训练参数示例（前5个）:")
    for name, numel in trainable_params[:5]:
        print(f"  - {name}: {numel:,}")

    return {"trainable_count": total_trainable, "frozen_count": total_frozen, "trainable_ratio": total_trainable / (total_trainable + total_frozen)}


@hydra.main(version_base=None, config_path="configs", config_name="ds003825_triplet_config")
def main(cfg: DictConfig):
    torch.backends.cudnn.benchmark = True
    _install_signal_handlers()

    distributed = _dist_enabled(cfg)
    rank = 0
    world_size = 1
    if distributed:
        device, rank, world_size = _dist_setup(cfg)
    else:
        device = torch.device(cfg.training.device)

    is_rank0 = _is_rank0(rank)
    run_dir = os.getcwd()  # Hydra sets CWD to the run directory by default.
    metrics_jsonl = os.path.join(run_dir, "metrics_epoch.jsonl")
    metrics_summary_path = os.path.join(run_dir, "metrics_summary.json")
    metrics_txt_path = os.path.join(run_dir, "metrics_summary.txt")

    if is_rank0:
        print("Hydra 配置:\n", OmegaConf.to_yaml(cfg))
        print(f"[rank0] run_dir: {run_dir}")

    use_wandb = bool(cfg.training.get("use_wandb", False))
    wandb = _maybe_init_wandb(cfg, enabled=use_wandb, is_rank0=is_rank0)

    if is_rank0:
        print(f"[rank {rank}/{world_size}] 初始化模型: SpatialMoEEncoder")
    model = SpatialMoEEncoder(
        n_channels=cfg.model.n_channels,
        n_samples=cfg.model.n_samples,
        embedding_dim=cfg.model.embedding_dim,
        pretrained_path=cfg.model.get("pretrained_path", None),
    ).to(device)

    if distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP

        find_unused = bool(cfg.training.get("distributed", {}).get("find_unused_parameters", True))
        model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=find_unused,
        )

    split_index = int(cfg.data.get("split_index", 0))

    # Select dataset implementation without modifying legacy dataset.py.
    dataset_impl = str(cfg.data.get("dataset_impl", "")).strip().lower()
    if dataset_impl in {"ds003825_bids", "bids_paper", "dataset_ds"} or str(cfg.data.get("backend", "")).lower() in {"ds003825_bids", "bids_paper"}:
        from dataset_ds import Ds003825TripletDataset as _TripletDataset

        train_dataset = _TripletDataset(cfg.data, mode="train", split_index=split_index)
        val_dataset = _TripletDataset(cfg.data, mode="val", split_index=split_index)
    else:
        train_dataset = TripletDataset(cfg.data, mode="train", split_index=split_index)
        val_dataset = TripletDataset(cfg.data, mode="val", split_index=split_index)

    use_unique_concepts = bool(cfg.data.get("unique_concepts_per_batch", False)) and getattr(train_dataset, "backend", "") == "ds003825"

    train_sampler = None
    val_sampler = None
    shuffle_train = True

    if distributed:
        from torch.utils.data.distributed import DistributedSampler

        if not use_unique_concepts:
            train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
            shuffle_train = False

        if bool(cfg.training.get("distributed", {}).get("val_sampler", True)):
            val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)

    pin_memory = bool(cfg.training.get("pin_memory", True))
    persistent_workers = bool(cfg.training.get("persistent_workers", True)) and int(cfg.training.num_workers) > 0

    train_batch_sampler = None
    if use_unique_concepts:
        # Use batch_sampler to guarantee unique concept ids per batch (rank-aware under torchrun).
        base_sampler = UniqueConceptBatchSampler(
            train_dataset,
            batch_size=int(cfg.training.batch_size),
            drop_last=True,
            shuffle=True,
            seed=int(cfg.training.get("seed", 0)),
        )
        max_steps_per_epoch = cfg.training.get("max_steps_per_epoch", None)
        max_steps_per_epoch = int(max_steps_per_epoch) if max_steps_per_epoch not in (None, "", 0) else None
        if max_steps_per_epoch is not None:
            train_batch_sampler = RepeatBatchSampler(base_sampler, num_batches=max_steps_per_epoch)
        else:
            train_batch_sampler = base_sampler
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=cfg.training.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
    else:
        max_steps_per_epoch = cfg.training.get("max_steps_per_epoch", None)
        max_steps_per_epoch = int(max_steps_per_epoch) if max_steps_per_epoch not in (None, "", 0) else None
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=shuffle_train,
            sampler=train_sampler,
            num_workers=cfg.training.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    # For meaningful retrieval evaluation under repeated concepts, val_loader should include concept ids.
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.training.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    text_only = bool(cfg.training.get("text_only", False))
    alpha = 0.0 if text_only else float(cfg.training.alpha)

    loss_fn_img = None if text_only else InfoNCE(initial_temperature=cfg.training.temperature).to(device)
    loss_fn_txt = InfoNCE(initial_temperature=cfg.training.temperature).to(device)

    # Keep the same parameter-freezing policy as main.py
    params_backbone_active = []
    params_head = []
    loss_params = (list(loss_fn_txt.parameters()) if loss_fn_img is None else (list(loss_fn_img.parameters()) + list(loss_fn_txt.parameters())))

    named_params = model.named_parameters() if not distributed else model.module.named_parameters()
    for name, param in named_params:
        if "backbone" in name:
            if "blocks.23" in name or "blocks.22" in name or "norm." in name:
                params_backbone_active.append(param)
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            params_head.append(param)
            param.requires_grad = True

    optimizer = optim.AdamW(
        [
            {"params": params_head, "lr": cfg.training.learning_rate},
            {"params": params_backbone_active, "lr": cfg.training.learning_rate * 0.1},
            {"params": loss_params, "lr": cfg.training.learning_rate, "weight_decay": 0.0},
        ],
        weight_decay=float(cfg.training.get("weight_decay", 0.15)),
    )

    if is_rank0:
        param_info = check_parameter_updates(model if not distributed else model.module)
        if wandb is not None:
            wandb.log(
                {
                    "trainable_params": param_info["trainable_count"],
                    "frozen_params": param_info["frozen_count"],
                    "trainable_ratio": param_info["trainable_ratio"],
                }
            )

    # Scheduler steps are based on actual optimizer steps
    grad_accum_steps = int(cfg.training.get("grad_accum_steps", 1) or 1)
    effective_batches = len(train_loader) if max_steps_per_epoch is None else min(len(train_loader), max_steps_per_epoch)
    total_opt_steps = max(1, (effective_batches // max(1, grad_accum_steps)) * int(cfg.training.epochs))
    warmup_steps = int(float(cfg.training.get("warmup_ratio", 0.1)) * total_opt_steps)

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_opt_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    selection_metric = str(cfg.training.get("selection_metric", "val/txt_top1"))
    selection_mode = str(cfg.training.get("selection_mode", "max")).lower()  # max or min
    if selection_mode not in {"max", "min"}:
        raise ValueError("training.selection_mode must be 'max' or 'min'")

    best_score = float("-inf") if selection_mode == "max" else float("inf")
    best_epoch = -1
    best_val_loss = float("inf")
    early_stop_enabled = bool(cfg.training.get("early_stopping", True))
    patience = int(cfg.training.get("patience", 10))
    min_delta = float(cfg.training.get("min_delta", 0.0))
    epochs_no_improve = 0
    if is_rank0:
        print(f"[rank {rank}] 开始训练... (max_steps_per_epoch={max_steps_per_epoch}, grad_accum_steps={grad_accum_steps})")
    for epoch in range(int(cfg.training.epochs)):
        # Reseed unique-concept sampler each epoch (rank-aware)
        if use_unique_concepts and train_batch_sampler is not None:
            base = getattr(train_batch_sampler, "batch_sampler", None) or train_batch_sampler
            if hasattr(base, "set_epoch"):
                base.set_epoch(epoch)

        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss, train_loss_img, train_loss_txt, steps = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn_img,
            loss_fn_txt,
            device,
            float(alpha),
            scaler,
            scheduler,
            grad_accum_steps=grad_accum_steps,
            max_steps=max_steps_per_epoch,
            log_every=int(cfg.training.get("log_every", 50)),
            sanity_check=bool(cfg.training.get("sanity_check", False)) and epoch == 0,
            sanity_check_once=True,
            rank=rank,
        )

        val_max_steps = int(cfg.training.get("max_val_steps", 200))
        val_results = evaluate_concept_retrieval(
            model=model,
            dataloader=val_loader,
            loss_fn_img=loss_fn_img,
            loss_fn_txt=loss_fn_txt,
            device=device,
            alpha=float(alpha),
            max_steps=val_max_steps if val_max_steps not in (0, None) else None,
            num_concepts=int(cfg.data.get("num_concepts", 1854)),
            embedding_dim=int(cfg.model.embedding_dim),
            distributed=distributed,
            rank=rank,
            k_values=tuple(int(k) for k in cfg.training.get("k_values", [1, 5, 10])),
        )
        val_loss = float(val_results["loss"])
        val_loss_img = float(val_results["loss_img"])
        val_loss_txt = float(val_results["loss_txt"])
        val_steps = int(val_results.get("steps", 0))

        stop_now = False
        if is_rank0:
            epoch_metrics = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "epoch": epoch,
                "text_only": bool(text_only),
                "train": {
                    "loss_total": train_loss,
                    "loss_image": train_loss_img,
                    "loss_text": train_loss_txt,
                    "steps": steps,
                },
                "val": {
                    "loss_total": val_loss,
                    "loss_image": val_loss_img,
                    "loss_text": val_loss_txt,
                    "steps": val_steps,
                    "present_concepts": int(val_results.get("present_concepts", 0)),
                    "img_metrics": val_results["img_metrics"],
                    "txt_metrics": val_results["txt_metrics"],
                },
                "lr": float(optimizer.param_groups[0]["lr"]),
                "selection_metric": selection_metric,
                "selection_mode": selection_mode,
            }
            _append_jsonl(metrics_jsonl, epoch_metrics)

            if wandb is not None:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss_total": train_loss,
                        "train/loss_image": train_loss_img,
                        "train/loss_text": train_loss_txt,
                        "train/steps": steps,
                        "val/loss_total": val_loss,
                        "val/loss_image": val_loss_img,
                        "val/loss_text": val_loss_txt,
                        "val/steps": val_steps,
                        "val/present_concepts": val_results.get("present_concepts", 0),
                        "val/img_top1": (0.0 if val_results.get("img_metrics") is None else val_results["img_metrics"]["top_1_accuracy"]),
                        "val/img_top5": (0.0 if val_results.get("img_metrics") is None else val_results["img_metrics"]["top_5_accuracy"]),
                        "val/img_top10": (0.0 if val_results.get("img_metrics") is None else val_results["img_metrics"]["top_10_accuracy"]),
                        "val/txt_top1": val_results["txt_metrics"]["top_1_accuracy"],
                        "val/txt_top5": val_results["txt_metrics"]["top_5_accuracy"],
                        "val/txt_top10": val_results["txt_metrics"]["top_10_accuracy"],
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )

            if bool(cfg.training.get("print_epoch_losses", True)):
                if text_only:
                    print(
                        f"[epoch {epoch}] "
                        f"train total={train_loss:.4f} txt={train_loss_txt:.4f} | "
                        f"val total={val_loss:.4f} txt={val_loss_txt:.4f} | "
                        f"val txt@1={val_results['txt_metrics']['top_1_accuracy']*100:.2f}% | "
                        f"steps={steps}"
                    )
                else:
                    print(
                        f"[epoch {epoch}] "
                        f"train total={train_loss:.4f} img={train_loss_img:.4f} txt={train_loss_txt:.4f} | "
                        f"val total={val_loss:.4f} img={val_loss_img:.4f} txt={val_loss_txt:.4f} | "
                        f"val img@1={(0.0 if val_results.get('img_metrics') is None else val_results['img_metrics']['top_1_accuracy']*100):.2f}% "
                        f"txt@1={val_results['txt_metrics']['top_1_accuracy']*100:.2f}% | "
                        f"steps={steps}"
                    )
            else:
                print(f"[epoch {epoch}] train={train_loss:.4f} val={val_loss:.4f} (steps={steps})")

            score = _get_metric_from_val(val_results, selection_metric)
            improved = (score > best_score + min_delta) if selection_mode == "max" else (score < best_score - min_delta)
            if improved:
                best_score = score
                best_epoch = epoch
                best_val_loss = val_loss
                out_path = os.path.join(run_dir, "best_eeg_encoder_ds003825.pth")
                state = model.module.state_dict() if distributed else model.state_dict()
                torch.save(state, out_path)
                print(f"保存 best checkpoint: {out_path} ({selection_metric}={score:.6f})")

            if early_stop_enabled:
                if improved:
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"EarlyStopping: {selection_metric} 未改善达到 {patience} 个 epoch（min_delta={min_delta}），停止训练。")
                    stop_now = True

        if distributed:
            import torch.distributed as dist

            flag = torch.tensor([1 if stop_now else 0], device=device)
            dist.broadcast(flag, src=0)
            stop_now = bool(flag.item())

        if stop_now:
            break

        if is_rank0:
            summary = {
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "selection_metric": selection_metric,
                "selection_mode": selection_mode,
                "run_dir": run_dir,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            _write_json(metrics_summary_path, summary)
            with open(metrics_txt_path, "w", encoding="utf-8") as f:
                f.write(f"best_epoch: {best_epoch}\n")
                f.write(f"best_val_loss: {best_val_loss}\n")
                f.write(f"best_score ({selection_metric}): {best_score}\n")
                f.write(f"metrics_epoch_jsonl: {metrics_jsonl}\n")
                f.write(f"best_checkpoint: {os.path.join(run_dir, 'best_eeg_encoder_ds003825.pth')}\n")
            if wandb is not None:
                wandb.finish()
    _dist_cleanup()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _, rank, _ = _get_dist_env()
        if rank == 0:
            print("[exit] KeyboardInterrupt received, aborting DDP...")
        _dist_abort_best_effort()
        try:
            _dist_cleanup()
        except Exception:
            pass
        os._exit(130)
    except BaseException:
        # Ensure we don't leave NCCL process groups behind on errors.
        _dist_abort_best_effort()
        try:
            _dist_cleanup()
        except Exception:
            pass
        raise
