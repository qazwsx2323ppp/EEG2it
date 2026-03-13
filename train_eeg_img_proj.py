import argparse
import os
from typing import List

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from dataset import TripletDataset
from models.clip_models import SpatialMoEEncoder
from transformers import CLIPTokenizer
from diffusers import StableDiffusionPipeline


def _make_cfg_data(args):
    cfg = OmegaConf.create(
        {
            "root": args.data_root,
            "eeg_path": args.eeg_path,
            "image_vec_path": args.image_vec_path,
            "text_vec_path": args.text_vec_path,
            "splits_path": args.splits_path,
            "return_target_id": True,
            "return_caption": False,
            "captions_dir": "",
            "captions_pattern": "{image_id}.txt",
        }
    )
    return cfg


def _load_eeg_images_list(eeg_pth: str) -> List[str]:
    if not os.path.isfile(eeg_pth):
        return []
    obj = torch.load(eeg_pth, map_location="cpu")
    if isinstance(obj, dict) and isinstance(obj.get("images"), list):
        return obj.get("images") or []
    return []


def _build_prompts(target_ids: torch.Tensor, eeg_images: List[str]) -> List[str]:
    prompts = []
    for tid in target_ids.tolist():
        syn = ""
        if 0 <= int(tid) < len(eeg_images):
            name = str(eeg_images[int(tid)])
            base = os.path.basename(name)
            stem, _ = os.path.splitext(base)
            syn = stem.split("_")[0] if "_" in stem else stem
        if syn:
            prompts.append(f"a photo of {syn}")
        else:
            prompts.append("a photo")
    return prompts


def main():
    ap = argparse.ArgumentParser(description="Train EEG->SD text space projection (512 -> 768).")
    ap.add_argument("--data_root", type=str, default="/media/wsqlab/data/ctp_file/EEG2it")
    ap.add_argument("--eeg_path", type=str, default="/media/wsqlab/data/ctp_file/EEG2it/data/EEG_data/eeg_55_95_std.pth")
    ap.add_argument("--image_vec_path", type=str, default="/media/wsqlab/data/ctp_file/EEG2it/data/image_vectors_aligned.npy")
    ap.add_argument("--text_vec_path", type=str, default="/media/wsqlab/data/ctp_file/EEG2it/data/text_vectors_aligned.npy")
    ap.add_argument("--splits_path", type=str, default="/media/wsqlab/data/ctp_file/EEG2it/data/EEG_data/block_splits_by_image_all.pth")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    ap.add_argument("--val_split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--split_index", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_batches", type=int, default=0, help="0 = all batches")
    ap.add_argument("--max_val_batches", type=int, default=0, help="0 = all batches")
    ap.add_argument("--patience", type=int, default=3, help="early stop patience on val loss")
    ap.add_argument("--eeg_ckpt", type=str, default="/media/wsqlab/data/ctp_file/EEG2it/temp/best_fornow.pth")
    ap.add_argument("--sd_model", type=str, default="/media/wsqlab/data/ctp_file/EEG2it/temp/sd15-diffusers")
    ap.add_argument("--sd_tokenizer_dir", type=str, default="/media/wsqlab/data/ctp_file/EEG2it/temp/sd15-diffusers/tokenizer")
    ap.add_argument("--out_ckpt", type=str, default="/media/wsqlab/data/ctp_file/EEG2it/temp/eeg_img_proj_ckpt.pth")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_half = device.type == "cuda"

    # Dataset
    ds = TripletDataset(_make_cfg_data(args), mode=args.split, split_index=int(args.split_index))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_ds = TripletDataset(_make_cfg_data(args), mode=args.val_split, split_index=int(args.split_index))
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    eeg_images = _load_eeg_images_list(args.eeg_path)

    # EEG encoder (frozen)
    eeg_encoder = SpatialMoEEncoder(
        n_channels=128,
        n_samples=512,
        embedding_dim=512,
        pretrained_path=None,
    ).to(device)
    if use_half:
        eeg_encoder = eeg_encoder.half()
    state = torch.load(args.eeg_ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    eeg_encoder.load_state_dict(state, strict=False)
    eeg_encoder.eval()
    for p in eeg_encoder.parameters():
        p.requires_grad = False

    # SD text encoder + tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(args.sd_tokenizer_dir, local_files_only=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_model, safety_checker=None, torch_dtype=torch.float16 if use_half else torch.float32
    )
    text_encoder = pipe.text_encoder.to(device)
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False

    # Trainable projection 512 -> text hidden
    text_hidden = text_encoder.config.hidden_size
    proj = torch.nn.Linear(512, text_hidden).to(device)
    if use_half:
        proj = proj.float()  # keep in fp32 for stability

    optim = torch.optim.AdamW(proj.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_half)

    best_loss = float("inf")
    bad_epochs = 0
    for epoch in range(args.epochs):
        running = 0.0
        steps = 0
        proj.train()
        for batch in dl:
            eeg = batch[0].to(device)
            target_id = batch[3].to(device) if len(batch) >= 4 else None
            if use_half:
                eeg = eeg.half()

            with torch.no_grad():
                emb_img, _, _ = eeg_encoder(eeg)

            prompts = _build_prompts(target_id, eeg_images) if target_id is not None else ["a photo"] * eeg.shape[0]
            text_inputs = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                text_emb = text_encoder(text_inputs.input_ids)[0]
                text_pooled = text_emb[:, 0]  # CLS token

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_half):
                pred = proj(emb_img.float())
                loss = torch.nn.functional.mse_loss(pred, text_pooled.float())

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            running += float(loss.item())
            steps += 1
            if args.max_batches and steps >= args.max_batches:
                break

        avg = running / max(1, steps)
        print(f"[EEG_IMG_PROJ] epoch={epoch} train_loss={avg:.6f}")

        # Validation
        val_loss = None
        if len(val_ds) > 0:
            proj.eval()
            v_running = 0.0
            v_steps = 0
            with torch.no_grad():
                for batch in val_dl:
                    eeg = batch[0].to(device)
                    target_id = batch[3].to(device) if len(batch) >= 4 else None
                    if use_half:
                        eeg = eeg.half()
                    emb_img, _, _ = eeg_encoder(eeg)
                    prompts = _build_prompts(target_id, eeg_images) if target_id is not None else ["a photo"] * eeg.shape[0]
                    text_inputs = tokenizer(
                        prompts,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).to(device)
                    text_emb = text_encoder(text_inputs.input_ids)[0]
                    text_pooled = text_emb[:, 0]
                    pred = proj(emb_img.float())
                    loss = torch.nn.functional.mse_loss(pred, text_pooled.float())
                    v_running += float(loss.item())
                    v_steps += 1
                    if args.max_val_batches and v_steps >= args.max_val_batches:
                        break
            val_loss = v_running / max(1, v_steps)
            print(f"[EEG_IMG_PROJ] epoch={epoch} val_loss={val_loss:.6f}")

        # Selection metric: val_loss if available, else train loss
        sel = val_loss if val_loss is not None else avg
        if sel < best_loss:
            best_loss = sel
            bad_epochs = 0
            torch.save(proj.state_dict(), args.out_ckpt)
            print(f"[EEG_IMG_PROJ] saved best to {args.out_ckpt}")
        else:
            bad_epochs += 1
            if args.patience > 0 and bad_epochs >= args.patience:
                print(f"[EEG_IMG_PROJ] early stop (patience={args.patience})")
                break


if __name__ == "__main__":
    main()
