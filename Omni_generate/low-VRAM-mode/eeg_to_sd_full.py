import argparse
import os
import sys
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

# Allow import from repo root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataset import TripletDataset
from models.clip_models import SpatialMoEEncoder
from painter_sd import StableDiffusionPainter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import modeling_qwen2_5_omni_low_VRAM_mode as qwen_mod
from transformers import AutoTokenizer


def _make_cfg_data(args):
    # TripletDataset expects attribute-style access (getattr).
    cfg = OmegaConf.create(
        {
            "root": args.data_root,
            "eeg_path": args.eeg_path,
            "image_vec_path": args.image_vec_path,
            "text_vec_path": args.text_vec_path,
            "splits_path": args.splits_path,
            "return_target_id": True,
            "return_caption": True,
            "captions_dir": args.captions_dir or "",
            "captions_pattern": "{image_id}.txt",
        }
    )
    return cfg


def _build_dataset(args):
    cfg_data = _make_cfg_data(args)
    return TripletDataset(cfg_data, mode=args.split, split_index=int(args.split_index))


def main():
    parser = argparse.ArgumentParser(description="EEG -> prompt (Qwen) -> SD image (full pipeline)")
    parser.add_argument("--data_root", type=str, default="/media/wsqlab/data/ctp_file/EEG2it")
    parser.add_argument("--eeg_path", type=str, default="/media/wsqlab/data/ctp_file/EEG2it/data/EEG_data/eeg_55_95_std.pth")
    parser.add_argument("--image_vec_path", type=str, default="/media/wsqlab/data/ctp_file/EEG2it/data/image_vectors_aligned.npy")
    parser.add_argument("--text_vec_path", type=str, default="/media/wsqlab/data/ctp_file/EEG2it/data/text_vectors_aligned.npy")
    parser.add_argument("--splits_path", type=str, default="/media/wsqlab/data/ctp_file/EEG2it/data/EEG_data/block_splits_by_image_all.pth")
    parser.add_argument("--captions_dir", type=str, default="")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--split_index", type=int, default=0)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1)

    parser.add_argument("--eeg_ckpt", type=str, default="/media/wsqlab/data/ctp_file/EEG2it/temp/best_fornow.pth")
    parser.add_argument("--eeg_projector_ckpt", type=str, default="/media/wsqlab/data/ctp_file/EEG2it/temp/best_eeg_projector.pth")

    parser.add_argument("--qwen_dir", type=str, default="/media/wsqlab/data/ctp_file/EEG2it/temp/Qwen2.5-Omni-3B")
    parser.add_argument("--sd_model", type=str, default="/media/wsqlab/data/ctp_file/EEG2it/temp/sd15-diffusers")
    parser.add_argument("--sd_config", type=str, default="")
    parser.add_argument("--eeg_img_proj_ckpt", type=str, default="")

    parser.add_argument("--prompt_instruction", type=str, default="Describe the image as a short Stable Diffusion prompt.")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--out", type=str, default="/media/wsqlab/data/ctp_file/EEG2it/output_eeg_to_sd.png")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--out_prefix", type=str, default="eeg_to_sd")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # 1) Build dataset
    ds = _build_dataset(args)

    # 2) EEG encoder
    eeg_encoder = SpatialMoEEncoder(
        n_channels=128,
        n_samples=512,
        embedding_dim=512,
        pretrained_path=None,
    ).to(device).half()
    state = torch.load(args.eeg_ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    eeg_encoder.load_state_dict(state, strict=False)
    eeg_encoder.eval()

    # 3) Qwen prompt generation
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_dir, trust_remote_code=True, local_files_only=True)
    model = qwen_mod.Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.qwen_dir, trust_remote_code=True, local_files_only=True, torch_dtype="auto"
    ).to(device)
    model.eeg_encoder = eeg_encoder
    model.eeg_projector.load_state_dict(torch.load(args.eeg_projector_ckpt, map_location="cpu"))

    # 4) SD generation with EEG visual token
    painter = StableDiffusionPainter(
        model_id=args.sd_model,
        device=str(device),
        torch_dtype=torch.float16,
        original_config_file=(args.sd_config or None),
    )
    sd_hidden = painter.pipe.text_encoder.config.hidden_size
    eeg_img_proj = torch.nn.Linear(512, sd_hidden).to(device).half()
    if args.eeg_img_proj_ckpt and os.path.exists(args.eeg_img_proj_ckpt):
        eeg_img_proj.load_state_dict(torch.load(args.eeg_img_proj_ckpt, map_location="cpu"))

    # 5) Iterate samples
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be >= 1")

    start_idx = int(args.sample_index if args.num_samples == 1 else args.start_index)
    end_idx = min(len(ds), start_idx + int(args.num_samples))
    if start_idx < 0 or start_idx >= len(ds):
        raise IndexError(f"start_index out of range: {start_idx} (len={len(ds)})")

    out_dir = args.out_dir.strip()
    if args.num_samples > 1:
        if not out_dir:
            out_dir = os.path.dirname(args.out) or "."
        os.makedirs(out_dir, exist_ok=True)

    for idx in range(start_idx, end_idx):
        item = ds[idx]
        eeg = item[0]
        if eeg.dim() == 2:
            eeg = eeg.unsqueeze(0)
        eeg = eeg.to(device)

        with torch.no_grad():
            emb_img, emb_txt, _ = eeg_encoder(eeg.half())

        gen_ids = model.generate_from_eeg(
            eeg_input=eeg,
            tokenizer=tokenizer,
            prompt_text=args.prompt_instruction,
            max_new_tokens=args.max_new_tokens,
        )
        prompt = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
        print(f"[Prompt][{idx}] {prompt}")

        image = painter.generate_with_eeg(
            prompt=prompt,
            eeg_img_emb=emb_img,
            eeg_proj=eeg_img_proj,
            negative_prompt="blurry, low quality, distorted",
            num_inference_steps=25,
            guidance_scale=7.5,
            height=512,
            width=512,
            seed=args.seed,
        )

        if args.num_samples == 1:
            out_path = args.out
        else:
            out_path = os.path.join(out_dir, f"{args.out_prefix}_{idx:05d}.png")

        image.save(out_path)
        print(f\"Saved image: {out_path}\")


if __name__ == "__main__":
    main()
