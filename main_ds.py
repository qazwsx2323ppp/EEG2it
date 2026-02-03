# Ignore compatibility warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import os

import hydra
import torch
import torch.optim as optim
import wandb
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


def _is_rank0(rank: int) -> bool:
    return rank == 0


def _to_device(batch, device):
    eeg_signals, image_vecs, text_vecs = batch
    return eeg_signals.to(device, non_blocking=True), image_vecs.to(device, non_blocking=True), text_vecs.to(device, non_blocking=True)


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
        eeg_signals, image_vecs, text_vecs = _to_device(batch, device)

        with autocast_ctx:
            outputs = model(eeg_signals)
            if len(outputs) == 3:
                eeg_img_embeddings, eeg_text_embeddings, _ = outputs
            else:
                eeg_img_embeddings, eeg_text_embeddings = outputs

            loss_img = loss_fn_img(eeg_img_embeddings, image_vecs)
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
                print(f"[sanity] cos(image_vec, text_vec) mean={cos:.4f} | unique image targets={uniq_img}/{bsz} | unique text targets={uniq_txt}/{bsz}")
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
        if _is_rank0(rank) and log_every and step % log_every == 0:
            wandb.log(
                {
                    "train/step_loss_total": total_loss / step,
                    "train/step_loss_image": total_loss_img / step,
                    "train/step_loss_text": total_loss_txt / step,
                    "train/optimizer_steps": opt_step,
                }
            )

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
        eeg_signals, image_vecs, text_vecs = _to_device(batch, device)
        outputs = model(eeg_signals)
        if len(outputs) == 3:
            eeg_img_embeddings, eeg_text_embeddings, _ = outputs
        else:
            eeg_img_embeddings, eeg_text_embeddings = outputs

        loss_img = loss_fn_img(eeg_img_embeddings, image_vecs)
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

    distributed = _dist_enabled(cfg)
    rank = 0
    world_size = 1

    if distributed:
        device, rank, world_size = _dist_setup(cfg)
    else:
        device = torch.device(cfg.training.device)

    is_rank0 = _is_rank0(rank)
    if is_rank0:
        print("Hydra 配置:\n", OmegaConf.to_yaml(cfg))
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

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
    train_dataset = TripletDataset(cfg.data, mode="train", split_index=split_index)
    val_dataset = TripletDataset(cfg.data, mode="val", split_index=split_index)

    use_unique_concepts = bool(cfg.data.get("unique_concepts_per_batch", False)) and getattr(train_dataset, "backend", "") == "ds003825"

    if distributed and not use_unique_concepts:
        from torch.utils.data.distributed import DistributedSampler

        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True

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

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.training.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    loss_fn_img = InfoNCE(initial_temperature=cfg.training.temperature).to(device)
    loss_fn_txt = InfoNCE(initial_temperature=cfg.training.temperature).to(device)

    # Keep the same parameter-freezing policy as main.py
    params_backbone_active = []
    params_head = []
    loss_params = list(loss_fn_img.parameters()) + list(loss_fn_txt.parameters())

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

    best_val = float("inf")
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
            float(cfg.training.alpha),
            scaler,
            scheduler,
            grad_accum_steps=grad_accum_steps,
            max_steps=max_steps_per_epoch,
            log_every=int(cfg.training.get("log_every", 50)),
            sanity_check=bool(cfg.training.get("sanity_check", False)) and epoch == 0,
            sanity_check_once=True,
            rank=rank,
        )

        val_loss, val_loss_img, val_loss_txt, val_steps = validate_loss_only(
            model, val_loader, loss_fn_img, loss_fn_txt, device, float(cfg.training.alpha), max_steps=int(cfg.training.get("max_val_steps", 200)), rank=rank
        )

        stop_now = False
        if is_rank0:
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
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

            if bool(cfg.training.get("print_epoch_losses", True)):
                print(
                    f"[epoch {epoch}] "
                    f"train total={train_loss:.4f} img={train_loss_img:.4f} txt={train_loss_txt:.4f} | "
                    f"val total={val_loss:.4f} img={val_loss_img:.4f} txt={val_loss_txt:.4f} | "
                    f"steps={steps}"
                )
            else:
                print(f"[epoch {epoch}] train={train_loss:.4f} val={val_loss:.4f} (steps={steps})")

            prev_best = best_val
            if val_loss < best_val:
                best_val = val_loss
                out_path = os.path.join(wandb.run.dir, "best_eeg_encoder_ds003825.pth")
                state = model.module.state_dict() if distributed else model.state_dict()
                torch.save(state, out_path)
                print(f"保存 best checkpoint: {out_path}")

            if early_stop_enabled:
                improvement = prev_best - val_loss
                if improvement > min_delta:
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"EarlyStopping: val_loss 未改善达到 {patience} 个 epoch（min_delta={min_delta}），停止训练。")
                    stop_now = True

        if distributed:
            import torch.distributed as dist

            flag = torch.tensor([1 if stop_now else 0], device=device)
            dist.broadcast(flag, src=0)
            stop_now = bool(flag.item())

        if stop_now:
            break

    if is_rank0:
        wandb.finish()
    _dist_cleanup()


if __name__ == "__main__":
    main()
