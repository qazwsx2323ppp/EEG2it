"""
Stage-2 training: EEG -> (frozen EEG encoder) -> 512-d embedding -> (trainable eeg_projector) -> Qwen2.5-Omni Thinker hidden
-> teacher-forcing to generate a text prompt/caption.

This is designed for the workflow:
  EEG -> prompt/description (Qwen) -> Stable Diffusion (image generation)

Config: configs/eeg_qwen_projector.yaml
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import hydra
from omegaconf import DictConfig, OmegaConf


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_kwargs(cfg_section) -> dict[str, Any]:
    if cfg_section is None:
        return {}
    try:
        d = OmegaConf.to_container(cfg_section, resolve=True)
    except Exception:
        d = dict(cfg_section)
    d = d or {}
    return {k: v for k, v in d.items() if v is not None}


def _normalize_torch_dtype(kwargs: dict[str, Any]) -> dict[str, Any]:
    import torch

    dt = kwargs.get("torch_dtype", None)
    if dt is None:
        return kwargs
    if isinstance(dt, str):
        s = dt.strip().lower()
        if s in {"auto"}:
            kwargs.pop("torch_dtype", None)
            return kwargs
        if s in {"fp16", "float16", "half"}:
            kwargs["torch_dtype"] = torch.float16
            return kwargs
        if s in {"bf16", "bfloat16"}:
            kwargs["torch_dtype"] = torch.bfloat16
            return kwargs
        if s in {"fp32", "float32"}:
            kwargs["torch_dtype"] = torch.float32
            return kwargs
    return kwargs


def _add_low_vram_to_syspath(low_vram_dir: str) -> None:
    p = Path(low_vram_dir).resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"low_vram_dir not found: {p}")
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _ensure_eeg_token(tokenizer, model, eeg_token: str) -> int:
    if eeg_token not in tokenizer.get_vocab():
        tokenizer.add_tokens([eeg_token])
        if hasattr(model, "thinker") and hasattr(model.thinker, "resize_token_embeddings"):
            model.thinker.resize_token_embeddings(len(tokenizer))
        elif hasattr(model, "resize_token_embeddings"):
            model.resize_token_embeddings(len(tokenizer))
    eeg_tok_id = int(tokenizer.convert_tokens_to_ids(eeg_token))
    # Injection logic reads thinker.config.eeg_token_index (see masked_scatter block in low-VRAM thinker forward)
    if hasattr(model, "thinker") and hasattr(model.thinker, "config"):
        model.thinker.config.eeg_token_index = eeg_tok_id
    elif hasattr(model, "config"):
        model.config.eeg_token_index = eeg_tok_id
    return eeg_tok_id


def _dist_info():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world_size, rank, local_rank


def _is_rank0(rank: int) -> bool:
    return rank == 0


def _tokenizer_eos_id(tokenizer) -> Optional[int]:
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        return int(eos)
    # fallback
    for name in ("sep_token_id", "pad_token_id"):
        v = getattr(tokenizer, name, None)
        if v is not None:
            return int(v)
    return None


@dataclass
class _BatchText:
    input_ids: "torch.Tensor"
    attention_mask: "torch.Tensor"
    labels: "torch.Tensor"


def _build_teacher_forcing_batch(tokenizer, eeg_token: str, instruction: str, targets: list[str], max_length: int) -> _BatchText:
    import torch

    eos_id = _tokenizer_eos_id(tokenizer)
    if eos_id is None:
        raise RuntimeError("Tokenizer has no eos_token_id/sep_token_id/pad_token_id; cannot build labels safely.")

    prefix_text = f"{eeg_token} {instruction}".strip() + "\n"
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids

    all_input_ids: list[list[int]] = []
    all_labels: list[list[int]] = []

    for t in targets:
        t = (t or "").strip()
        tgt_ids = tokenizer(t, add_special_tokens=False).input_ids
        if eos_id is not None:
            tgt_ids = tgt_ids + [eos_id]

        # Truncate to max_length
        max_tgt = max(1, max_length - len(prefix_ids))
        tgt_ids = tgt_ids[:max_tgt]

        ids = prefix_ids + tgt_ids
        labels = ([-100] * len(prefix_ids)) + tgt_ids

        all_input_ids.append(ids[:max_length])
        all_labels.append(labels[:max_length])

    max_len = max(len(x) for x in all_input_ids)
    max_len = min(max_len, max_length)

    pad_id = int(getattr(tokenizer, "pad_token_id", eos_id))

    def _pad(seq: list[int], pad_value: int) -> list[int]:
        return seq + [pad_value] * (max_len - len(seq))

    input_ids = torch.tensor([_pad(x[:max_len], pad_id) for x in all_input_ids], dtype=torch.long)
    labels = torch.tensor([_pad(x[:max_len], -100) for x in all_labels], dtype=torch.long)
    attention_mask = (input_ids != pad_id).long()

    return _BatchText(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


def _try_load_clip(cfg_clip: DictConfig):
    try:
        from transformers import CLIPModel, CLIPProcessor
    except Exception as e:
        print(f"[评估] 未找到 transformers CLIP 组件，跳过 CLIP 评估：{e}")
        return None, None, None

    model_name = str(cfg_clip.get("model", "openai/clip-vit-base-patch32"))
    allow_download = bool(cfg_clip.get("allow_download", False))
    local_files_only = not allow_download
    try:
        model = CLIPModel.from_pretrained(model_name, local_files_only=local_files_only)
        proc = CLIPProcessor.from_pretrained(model_name, local_files_only=local_files_only)
    except Exception as e:
        print(f"[评估] 无法加载 CLIP({model_name}, local_files_only={local_files_only})，跳过 CLIP 评估：{e}")
        return None, None, None

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    return model, proc, device


def _clip_text_embed(clip_model, clip_proc, device, texts: list[str]) -> "torch.Tensor":
    import torch

    feats_all = []
    bs = 64
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            chunk = texts[i : i + bs]
            inputs = clip_proc(text=chunk, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            feats = clip_model.get_text_features(**inputs)
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
            feats_all.append(feats.detach().cpu())
    return torch.cat(feats_all, dim=0)


def _evaluate_generation(
    *,
    cfg: DictConfig,
    model,
    tokenizer,
    eeg_encoder,
    val_loader,
    eeg_tok_id: int,
    all_text_vectors_cpu: Optional["torch.Tensor"],
):
    import torch

    model.eval()
    eeg_encoder.eval()

    max_batches = int(cfg.eval.get("max_batches", 4))
    max_new_tokens = int(cfg.eval.get("max_new_tokens", 64))
    instruction = str(cfg.eval.get("instruction", "Describe the image as a short Stable Diffusion prompt."))
    eeg_token = str(cfg.prompt.get("eeg_token", "<EEG>"))

    # CLIP evaluator (optional)
    clip_model = None
    clip_proc = None
    clip_device = None
    if bool(cfg.eval.get("clip", {}).get("enabled", True)):
        clip_model, clip_proc, clip_device = _try_load_clip(cfg.eval.clip)

    texts_gen: list[str] = []
    target_ids_all: list[int] = []
    gt_text_vecs: list[torch.Tensor] = []

    device = next(model.parameters()).device

    # Build fixed prompt ids for generation
    prompt_ids = tokenizer(f"{eeg_token} {instruction}\n", add_special_tokens=False).input_ids
    prompt_ids_t = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    if int((prompt_ids_t == int(eeg_tok_id)).sum().item()) != 1:
        raise RuntimeError(
            f"Eval prompt must contain exactly one EEG token, got ids={prompt_ids}. "
            f"Check tokenizer vocab for {eeg_token}."
        )

    with torch.no_grad():
        for bidx, batch in enumerate(val_loader):
            if bidx >= max_batches:
                break
            eeg = batch[0].to(device)
            tgt_ids = batch[3] if len(batch) >= 4 else None
            if tgt_ids is None:
                continue
            tgt_ids = tgt_ids.cpu().tolist()
            target_ids_all.extend([int(x) for x in tgt_ids])

            # GT CLIP vector from dataset (already normalized in dataset.py)
            gt_text_vecs.append(batch[2].float().cpu())

            # EEG -> CLIP-dim embedding (frozen) -> projector/Q-Former
            emb_img, emb_txt, _ = eeg_encoder(eeg)
            if getattr(model, "use_qformer", False) and getattr(model, "qformer", None) is not None:
                kv = torch.stack([emb_img, emb_txt], dim=1)
                eeg_embed_h = model.qformer(kv, return_sequence=False)
            else:
                eeg_embed_h = model.eeg_projector(emb_txt)  # [B, H]

            # Expand prompt to batch
            input_ids = prompt_ids_t.expand(eeg.shape[0], -1).contiguous()
            gen = model.thinker.generate(
                input_ids=input_ids,
                eeg_embeds=eeg_embed_h.to(device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            gen_new = gen[:, input_ids.size(1) :]
            texts = tokenizer.batch_decode(gen_new, skip_special_tokens=True)
            texts_gen.extend([t.strip() for t in texts])

    result = {
        "eval/gen_samples": len(texts_gen),
    }

    if not texts_gen:
        return result

    if clip_model is None:
        return result

    gen_clip = _clip_text_embed(clip_model, clip_proc, clip_device, texts_gen)  # [M, 512] on CPU
    gt_text = torch.cat(gt_text_vecs, dim=0)  # [M, 512] on CPU

    # cosine sim since both normalized
    sim = (gen_clip * gt_text).sum(dim=-1)
    result["eval/clip_text_sim_mean"] = float(sim.mean().item())
    result["eval/clip_text_sim_median"] = float(sim.median().item())

    # Retrieval hit@1: generated text CLIP embedding vs all candidate text vectors
    if all_text_vectors_cpu is not None and target_ids_all:
        # Optional downsample candidates for speed
        full = bool(cfg.eval.get("retrieval", {}).get("full", False))
        num_cand = int(cfg.eval.get("retrieval", {}).get("num_candidates", 0))
        all_text_full = all_text_vectors_cpu
        n_all = int(all_text_full.shape[0])

        # Build a candidate set that always includes GT ids
        gt_ids = [int(x) for x in target_ids_all if 0 <= int(x) < n_all]
        gt_unique = sorted(set(gt_ids))
        if full or num_cand <= 0 or num_cand >= n_all:
            cand_idx = list(range(n_all))
        else:
            cand_idx = list(gt_unique)
            # fill with random negatives
            import random

            random.seed(int(cfg.training.get("seed", 42)) + 1234)
            need = max(0, int(num_cand) - len(cand_idx))
            if need > 0:
                pool = [i for i in range(n_all) if i not in set(cand_idx)]
                if need >= len(pool):
                    cand_idx.extend(pool)
                else:
                    cand_idx.extend(random.sample(pool, need))

        cand = all_text_full[cand_idx]  # [N, 512] on CPU

        # compute on CPU to avoid requiring extra GPU memory in eval
        logits = gen_clip @ cand.T  # [M, N]
        pred_pos = logits.argmax(dim=-1).cpu().tolist()
        pred = [int(cand_idx[p]) for p in pred_pos]
        hits = 0
        for p, gt_id in zip(pred, target_ids_all):
            hits += int(int(p) == int(gt_id))
        result["eval/clip_hit1"] = float(hits / max(1, len(target_ids_all)))

    return result


@hydra.main(version_base=None, config_path="../configs", config_name="eeg_qwen_projector")
def main(cfg: DictConfig) -> None:
    world_size, rank, local_rank = _dist_info()
    is_dist = world_size > 1
    if not is_dist or _is_rank0(rank):
        print("Hydra 配置:\n", OmegaConf.to_yaml(cfg))

    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from dataset import TripletDataset
    from models.clip_models import SpatialMoEEncoder

    _set_seed(int(cfg.training.get("seed", 42)))

    if is_dist:
        import torch.distributed as dist

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(str(cfg.training.get("device", "cuda")))

    # ---- Data ----
    split_index = int(cfg.data.get("split_index", 0))
    train_ds = TripletDataset(cfg.data, mode="train", split_index=split_index)
    val_ds = TripletDataset(cfg.data, mode="val", split_index=split_index)

    train_sampler = None
    if is_dist:
        from torch.utils.data import DistributedSampler

        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.training.get("batch_size", 2)),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=int(cfg.training.get("num_workers", 2)),
        pin_memory=True,
    )

    val_loader = None
    if not is_dist or _is_rank0(rank):
        val_loader = DataLoader(
            val_ds,
            batch_size=int(cfg.training.get("eval_batch_size", cfg.training.get("batch_size", 2))),
            shuffle=False,
            num_workers=int(cfg.training.get("num_workers", 2)),
            pin_memory=True,
        )

    # ---- Stage-1 EEG encoder (frozen) ----
    eeg_encoder = SpatialMoEEncoder(
        n_channels=int(cfg.stage1.model.n_channels),
        n_samples=int(cfg.stage1.model.n_samples),
        embedding_dim=int(cfg.stage1.model.embedding_dim),
        pretrained_path=str(cfg.stage1.model.get("pretrained_path", "")) or None,
        router_mode=str(cfg.stage1.model.get("router_mode", "moe")),
        head_dropout=float(cfg.stage1.model.get("head_dropout", 0.5)),
    ).to(device)

    ckpt_path = str(cfg.stage1.get("ckpt_path", "")).strip()
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"stage1.ckpt_path not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    eeg_encoder.load_state_dict(state, strict=False)
    eeg_encoder.eval()
    for p in eeg_encoder.parameters():
        p.requires_grad = False

    # ---- Qwen2.5-Omni (Thinker) ----
    low_vram_dir = str(cfg.qwen.get("low_vram_dir", "")).strip()
    _add_low_vram_to_syspath(low_vram_dir)
    import modeling_qwen2_5_omni_low_VRAM_mode as qwen_mod

    from transformers import AutoTokenizer

    qwen_model_dir = str(cfg.qwen.get("model_dir", "")).strip()
    if not qwen_model_dir:
        raise ValueError("qwen.model_dir is empty. Please set it to your local Qwen2.5-Omni model folder.")

    qwen_kwargs = _to_kwargs(cfg.qwen.get("from_pretrained_kwargs", {}))
    qwen_kwargs = _normalize_torch_dtype(qwen_kwargs)

    if is_dist and "device_map" in qwen_kwargs:
        if _is_rank0(rank):
            print("[Stage-2] DDP detected: ignoring device_map to replicate model on each GPU.")
        qwen_kwargs.pop("device_map", None)
    tok_kwargs = _to_kwargs(cfg.qwen.get("tokenizer_kwargs", {}))
    tokenizer = AutoTokenizer.from_pretrained(qwen_model_dir, **tok_kwargs)

    model = qwen_mod.Qwen2_5OmniForConditionalGeneration.from_pretrained(qwen_model_dir, **qwen_kwargs)
    # If device_map is used, Accelerate will place modules automatically.
    if not qwen_kwargs.get("device_map"):
        model = model.to(device)

    # Optional: activation checkpointing to reduce memory
    if bool(cfg.training.get("grad_checkpointing", False)):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            try:
                model.config.use_cache = False
            except Exception:
                pass
        if hasattr(model, "thinker") and hasattr(model.thinker, "config"):
            try:
                model.thinker.config.use_cache = False
            except Exception:
                pass

    # Attach frozen EEG encoder for convenience (optional, used only in eval helpers)
    model.eeg_encoder = eeg_encoder

    eeg_token = str(cfg.prompt.get("eeg_token", "<EEG>"))
    eeg_tok_id = _ensure_eeg_token(tokenizer, model, eeg_token=eeg_token)
    if not is_dist or _is_rank0(rank):
        print(f"[Stage-2] EEG token id: {eeg_tok_id}")

    # ---- Optional EEG Q-Former ----
    use_qformer = bool(cfg.get("qformer", {}).get("enabled", False))
    if use_qformer:
        try:
            from out_qformer import EEGQFormer
        except Exception as e:
            raise ImportError(f"Failed to import EEGQFormer from low_vram_dir: {e}")

        qf_cfg = cfg.get("qformer", {})
        thinker_cfg = getattr(model.thinker, "config", None)
        thinker_hidden = None
        if thinker_cfg is not None:
            thinker_hidden = getattr(getattr(thinker_cfg, "text_config", thinker_cfg), "hidden_size", None)
        thinker_hidden = int(thinker_hidden or model.thinker.config.hidden_size)
        qformer = EEGQFormer(
            hidden_size=thinker_hidden,
            kv_dim=int(qf_cfg.get("kv_dim", 512)),
            num_queries=int(qf_cfg.get("num_queries", 4)),
            num_layers=int(qf_cfg.get("num_layers", 2)),
            num_heads=int(qf_cfg.get("num_heads", 8)),
            dropout=float(qf_cfg.get("dropout", 0.1)),
            resid_scale=float(qf_cfg.get("resid_scale", 0.5)),
        ).to(device=device, dtype=torch.float32)

        qformer_ckpt = str(qf_cfg.get("ckpt_path", "")).strip()
        if qformer_ckpt:
            if not os.path.isfile(qformer_ckpt):
                raise FileNotFoundError(f"qformer.ckpt_path not found: {qformer_ckpt}")
            qformer.load_state_dict(torch.load(qformer_ckpt, map_location="cpu"))

        model.qformer = qformer
        model.use_qformer = True

    # Freeze everything except eeg_projector / qformer (if enabled)
    for p in model.parameters():
        p.requires_grad = False

    params_to_train = []
    if use_qformer and model.qformer is not None:
        for p in model.qformer.parameters():
            p.requires_grad = True
        params_to_train += list(model.qformer.parameters())
    else:
        for p in model.eeg_projector.parameters():
            p.requires_grad = True
        # Keep trainable projector in fp32 to avoid GradScaler unscale errors on fp16 params.
        model.eeg_projector = model.eeg_projector.to(device=device, dtype=torch.float32)
        params_to_train += list(model.eeg_projector.parameters())

    # Wrap qformer or eeg_projector with DDP if distributed
    if is_dist:
        from torch.nn.parallel import DistributedDataParallel as DDP
        if use_qformer and model.qformer is not None:
            model.qformer = DDP(
                model.qformer,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )
        else:
            model.eeg_projector = DDP(
                model.eeg_projector,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )

    # Optimizer
    optim = torch.optim.AdamW(
        params_to_train,
        lr=float(cfg.training.get("lr", 1e-4)),
        weight_decay=float(cfg.training.get("weight_decay", 0.0)),
    )

    use_amp = bool(cfg.training.get("amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    grad_accum = int(cfg.training.get("grad_accum_steps", 1))

    instruction = str(cfg.prompt.get("instruction", "Generate a concise Stable Diffusion prompt."))
    target_template = str(cfg.prompt.get("target_template", "{caption}"))
    max_length = int(cfg.prompt.get("max_length", 128))

    out_dir = Path(os.getcwd())
    metrics_path = out_dir / "stage2_metrics.jsonl"

    best_score = float("-inf")
    best_path = out_dir / ("best_eeg_qformer.pth" if use_qformer else "best_eeg_projector.pth")

    def _append_metrics(obj: dict[str, Any]) -> None:
        if is_dist and not _is_rank0(rank):
            return
        with metrics_path.open("a", encoding="utf-8") as f:
            import json

            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    global_step = 0
    epochs = int(cfg.training.get("epochs", 1))
    log_every = int(cfg.training.get("log_every", 50))
    max_steps = int(cfg.training.get("max_train_steps", 0))

    warn_no_caption = True

    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running = 0.0
        nloss = 0
        optim.zero_grad(set_to_none=True)

        it = train_loader
        if not is_dist or _is_rank0(rank):
            it = tqdm(train_loader, desc=f"Stage2 Train (epoch {epoch+1}/{epochs})")

        for batch in it:
            eeg = batch[0].to(device)
            target_ids = batch[3].cpu().tolist() if len(batch) >= 4 else [0] * int(eeg.shape[0])
            captions = batch[4] if len(batch) >= 5 else None
            if captions is None:
                if warn_no_caption and (not is_dist or _is_rank0(rank)):
                    print("[Stage-2] Warning: dataset did not return caption/text; using target_id template for supervision (usually worse).")
                    warn_no_caption = False
                captions = ["" for _ in target_ids]

            targets = []
            for tid, cap in zip(target_ids, captions):
                cap_s = str(cap or "").strip()
                targets.append(target_template.format(caption=cap_s, target_id=int(tid)).strip())

            txt_batch = _build_teacher_forcing_batch(
                tokenizer,
                eeg_token=eeg_token,
                instruction=instruction,
                targets=targets,
                max_length=max_length,
            )
            input_ids = txt_batch.input_ids.to(device)
            attn = txt_batch.attention_mask.to(device)
            labels = txt_batch.labels.to(device)

            # Ensure exactly one EEG token per sample (required by masked_scatter injection).
            eeg_counts = (input_ids == int(eeg_tok_id)).sum(dim=1)
            if int(eeg_counts.min().item()) != 1 or int(eeg_counts.max().item()) != 1:
                raise RuntimeError(
                    f"Expected exactly one {eeg_token} token per sample, got counts={eeg_counts.tolist()}.\n"
                    "Check that the tokenizer includes the EEG token as a single vocab item."
                )

            # EEG -> frozen encoder -> CLIP-dim embedding
            with torch.no_grad():
                emb_img, emb_txt, _ = eeg_encoder(eeg)

            with torch.cuda.amp.autocast(enabled=use_amp):
                if use_qformer and model.qformer is not None:
                    kv = torch.stack([emb_img, emb_txt], dim=1)
                    eeg_embed_h = model.qformer(kv, return_sequence=False)
                else:
                    eeg_embed_h = model.eeg_projector(emb_txt)  # [B, H], trainable
                out = model(
                    input_ids=input_ids,
                    attention_mask=attn,
                    labels=labels,
                    eeg_embeds=eeg_embed_h,
                )
                loss = out.loss / max(1, grad_accum)

            scaler.scale(loss).backward()

            if (global_step + 1) % grad_accum == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            running += float(loss.item() * max(1, grad_accum))
            nloss += 1
            global_step += 1

            if log_every and (global_step % log_every == 0) and (not is_dist or _is_rank0(rank)):
                print(f"[Stage-2] step={global_step} loss={running / max(1, nloss):.4f}")

            if max_steps and global_step >= max_steps:
                break

        train_loss = running / max(1, nloss)
        if not is_dist or _is_rank0(rank):
            print(f"[Stage-2] epoch={epoch} train_loss={train_loss:.4f}")

        # Eval (optional)
        eval_metrics = {}
        if (not is_dist or _is_rank0(rank)) and val_loader is not None and bool(cfg.eval.get("enabled", True)):
            all_text_vectors_cpu = getattr(val_loader.dataset, "all_text_vectors", None)
            eval_metrics = _evaluate_generation(
                cfg=cfg,
                model=model,
                tokenizer=tokenizer,
                eeg_encoder=eeg_encoder,
                val_loader=val_loader,
                eeg_tok_id=eeg_tok_id,
                all_text_vectors_cpu=all_text_vectors_cpu,
            )
            print(f"[Stage-2] eval: {eval_metrics}")

        # Select best
        selection_metric = str(cfg.training.get("selection_metric", "eval/clip_hit1"))
        selection_mode = str(cfg.training.get("selection_mode", "max")).lower()
        min_delta = float(cfg.training.get("min_delta", 0.0))

        if not is_dist or _is_rank0(rank):
            score = float(eval_metrics.get(selection_metric, -train_loss))
            improved = (score > best_score + min_delta) if selection_mode == "max" else (score < best_score - min_delta)
            if improved:
                best_score = score
                if use_qformer and model.qformer is not None:
                    qf = model.qformer.module if hasattr(model.qformer, "module") else model.qformer
                    torch.save(qf.state_dict(), best_path)
                    print(f"[Stage-2] Saved best_eeg_qformer to: {best_path} ({selection_metric}={best_score:.6f})")
                else:
                    proj = model.eeg_projector.module if hasattr(model.eeg_projector, "module") else model.eeg_projector
                    torch.save(proj.state_dict(), best_path)
                    print(f"[Stage-2] Saved best_eeg_projector to: {best_path} ({selection_metric}={best_score:.6f})")

        _append_metrics(
            {
                "epoch": int(epoch),
                "global_step": int(global_step),
                "train/loss": float(train_loss),
                **{k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))},
                "best/score": float(best_score),
                "best/path": str(best_path),
            }
        )

        if max_steps and global_step >= max_steps:
            break

    if not is_dist or _is_rank0(rank):
        print("[Stage-2] Training complete.")

    if is_dist:
        import torch.distributed as dist

        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
