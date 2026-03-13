import argparse
import os
import re
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

def _safe_name(name: str) -> str:
    name = str(name).strip()
    if not name:
        return ""
    base = os.path.basename(name)
    stem, _ = os.path.splitext(base)
    stem = stem.strip()
    if not stem:
        return ""
    # Keep only safe characters for filenames.
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    return stem[:120]

def _load_image_name_map(image_data_path: str):
    if not image_data_path:
        return None
    if not os.path.isfile(image_data_path):
        return None
    try:
        data = torch.load(image_data_path, map_location="cpu")
    except Exception:
        return None

    # Common patterns: list of dicts, dict with "images"/"data", list of strings
    if isinstance(data, dict):
        for key in ("images", "data", "image_data", "items"):
            if key in data:
                data = data[key]
                break

    if isinstance(data, list):
        # If list elements are dicts, try common fields
        if data and isinstance(data[0], dict):
            def _get_name(i: int):
                item = data[i]
                return (
                    item.get("file_name")
                    or item.get("filename")
                    or item.get("path")
                    or item.get("image_path")
                    or item.get("image")
                    or ""
                )
            return _get_name
        # If list of strings
        if data and isinstance(data[0], str):
            return lambda i: data[i]

    # If dict mapping id->name
    if isinstance(data, dict):
        return lambda i: data.get(int(i), data.get(str(i), ""))

    return None

def _load_eeg_image_list(eeg_pth: str):
    if not eeg_pth or not os.path.isfile(eeg_pth):
        return []
    try:
        eeg = torch.load(eeg_pth, map_location="cpu")
    except Exception:
        return []
    if isinstance(eeg, dict) and isinstance(eeg.get("images"), list):
        return eeg.get("images") or []
    return []

def _build_image_list(image_root: str, exts: tuple[str, ...]):
    if not image_root or not os.path.isdir(image_root):
        return []
    image_root = os.path.abspath(image_root)
    rel_paths: list[str] = []
    for cls in sorted(os.listdir(image_root)):
        cls_dir = os.path.join(image_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = []
        for fn in os.listdir(cls_dir):
            if fn.lower().endswith(exts):
                files.append(fn)
        for fn in sorted(files):
            rel_paths.append(os.path.join(cls, fn))
    return rel_paths

def _resolve_eeg_image_name(eeg_name: str, image_root: str, exts: tuple[str, ...]) -> str:
    name = str(eeg_name).strip()
    if not name:
        return ""
    # If already has path/extension, use as-is
    if "/" in name or "\\" in name or "." in os.path.basename(name):
        return name
    # Try to resolve under image_root with inferred class folder
    synset = name.split("_")[0] if "_" in name else ""
    if image_root and synset:
        base_dir = os.path.join(image_root, synset)
        if os.path.isdir(base_dir):
            # Try known extensions
            for ext in exts:
                cand = os.path.join(synset, f"{name}{ext}")
                if os.path.isfile(os.path.join(image_root, cand)):
                    return cand
            # Fallback: any file starting with name
            for fn in os.listdir(base_dir):
                if fn.startswith(name + "."):
                    return os.path.join(synset, fn)
    return name


def main():
    parser = argparse.ArgumentParser(description="EEG -> prompt (Qwen) -> SD image (full pipeline)")
    parser.add_argument("--data_root", type=str, default="/media/wsqlab/data/ctp_file/EEG2it")
    parser.add_argument("--eeg_path", type=str, default="/media/wsqlab/data/ctp_file/EEG2it/data/EEG_data/eeg_55_95_std.pth")
    parser.add_argument("--image_vec_path", type=str, default="/media/wsqlab/data/ctp_file/EEG2it/data/image_vectors_aligned.npy")
    parser.add_argument("--text_vec_path", type=str, default="/media/wsqlab/data/ctp_file/EEG2it/data/text_vectors_aligned.npy")
    parser.add_argument("--splits_path", type=str, default="/media/wsqlab/data/ctp_file/EEG2it/data/EEG_data/block_splits_by_image_all.pth")
    parser.add_argument("--image_data_path", type=str, default="/media/wsqlab/data/ctp_file/EEG2it/data/EEG_data/image_data.pth")
    parser.add_argument("--image_root", type=str, default="")
    parser.add_argument("--image_exts", type=str, default=".jpg,.jpeg,.png,.bmp,.webp,.jpeg,.jpeg,.JPEG,.JPG")
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
    parser.add_argument("--sd_tokenizer_dir", type=str, default="")
    parser.add_argument("--eeg_img_proj_ckpt", type=str, default="")

    parser.add_argument("--prompt_instruction", type=str, default="Describe the image as a short Stable Diffusion prompt.")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--out", type=str, default="/media/wsqlab/data/ctp_file/EEG2it/output_eeg_to_sd.png")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--out_prefix", type=str, default="eeg_to_sd")
    parser.add_argument("--disable_eeg_token", action="store_true", help="Disable EEG visual token injection")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # 1) Build dataset
    ds = _build_dataset(args)
    get_image_name = _load_image_name_map(args.image_data_path)
    eeg_images = _load_eeg_image_list(args.eeg_path)
    exts = tuple(x.strip().lower() for x in args.image_exts.split(",") if x.strip())
    image_list = _build_image_list(args.image_root, exts)

    # 2) EEG encoder
    use_half = device.type == "cuda"
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

    # 3) Qwen prompt generation
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_dir, trust_remote_code=True, local_files_only=True)
    if device.type == "cuda":
        qwen_dtype = torch.float16
    else:
        qwen_dtype = torch.float32
    model = qwen_mod.Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.qwen_dir, trust_remote_code=True, local_files_only=True, torch_dtype=qwen_dtype
    ).to(device)
    if use_half:
        model = model.half()
    model.eeg_encoder = eeg_encoder
    model.eeg_projector.load_state_dict(torch.load(args.eeg_projector_ckpt, map_location="cpu"))

    # 4) SD generation with EEG visual token
    sd_tokenizer_dir = args.sd_tokenizer_dir.strip()
    if sd_tokenizer_dir:
        vocab_path = os.path.join(sd_tokenizer_dir, "vocab.json")
        merges_path = os.path.join(sd_tokenizer_dir, "merges.txt")
        if not (os.path.exists(vocab_path) and os.path.exists(merges_path)):
            sd_tokenizer_dir = ""
    painter = StableDiffusionPainter(
        model_id=args.sd_model,
        device=str(device),
        torch_dtype=torch.float16,
        original_config_file=(args.sd_config or None),
        tokenizer_dir=(sd_tokenizer_dir or None),
    )
    sd_hidden = painter.pipe.text_encoder.config.hidden_size
    eeg_img_proj = torch.nn.Linear(512, sd_hidden).to(device)
    if use_half:
        eeg_img_proj = eeg_img_proj.half()
    has_img_proj_ckpt = bool(args.eeg_img_proj_ckpt and os.path.exists(args.eeg_img_proj_ckpt))
    if has_img_proj_ckpt:
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
        target_id = None
        if isinstance(item, (list, tuple)) and len(item) >= 4:
            try:
                target_id = int(item[3])
            except Exception:
                target_id = None
        if eeg.dim() == 2:
            eeg = eeg.unsqueeze(0)
        eeg = eeg.to(device)
        if use_half:
            eeg = eeg.half()

        with torch.no_grad():
            emb_img, emb_txt, _ = eeg_encoder(eeg)

        gen_ids = model.generate_from_eeg(
            eeg_input=eeg,
            tokenizer=tokenizer,
            prompt_text=args.prompt_instruction,
            max_new_tokens=args.max_new_tokens,
        )
        raw_prompt = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
        prompt = raw_prompt.strip()
        if not prompt:
            syn = ""
            if image_name_raw:
                base = os.path.basename(str(image_name_raw))
                stem = os.path.splitext(base)[0]
                syn = stem.split("_")[0] if "_" in stem else stem
            if syn:
                prompt = f"a photo of {syn}"
            else:
                prompt = "a photo"
        print(f"[Prompt][{idx}] {prompt}")

        use_eeg_token = has_img_proj_ckpt and (not args.disable_eeg_token)
        if use_eeg_token:
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
        else:
            image = painter.generate(
                prompt=prompt,
                negative_prompt="blurry, low quality, distorted",
                num_inference_steps=25,
                guidance_scale=7.5,
                height=512,
                width=512,
                seed=args.seed,
            )

        image_name_raw = ""
        image_name = ""
        # Priority 1: eeg.pth provides explicit image list
        if target_id is not None and eeg_images:
            try:
                if 0 <= int(target_id) < len(eeg_images):
                    image_name_raw = _resolve_eeg_image_name(eeg_images[int(target_id)], args.image_root, exts)
                    image_name = _safe_name(image_name_raw)
            except Exception:
                image_name = ""
        if target_id is not None and get_image_name is not None:
            try:
                image_name_raw = get_image_name(target_id)
                image_name = _safe_name(image_name_raw)
            except Exception:
                image_name = ""
        if not image_name and target_id is not None and image_list:
            try:
                if 0 <= int(target_id) < len(image_list):
                    image_name_raw = image_list[int(target_id)]
                    image_name = _safe_name(image_name_raw)
            except Exception:
                image_name = ""

        if args.num_samples == 1:
            out_path = args.out
        else:
            parts = [args.out_prefix, f"{idx:05d}"]
            if image_name:
                parts.append(image_name)
            elif target_id is not None:
                parts.append(f"img{target_id:05d}")
            out_path = os.path.join(out_dir, "_".join(parts) + ".png")

        image.save(out_path)
        print(f"Saved image: {out_path}")
        # Save prompt alongside image
        prompt_path = os.path.splitext(out_path)[0] + ".txt"
        try:
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(f"RAW_PROMPT: {raw_prompt}\n")
                f.write(f"USED_PROMPT: {prompt}\n")
                if image_name_raw:
                    f.write(f"IMAGE_NAME: {image_name_raw}\n")
                if target_id is not None:
                    f.write(f"IMAGE_ID: {int(target_id)}\n")
                f.write(f"EEG_INDEX: {idx}\n")
                f.write(f"USE_EEG_TOKEN: {bool(use_eeg_token)}\n")
        except Exception:
            pass


if __name__ == "__main__":
    main()
