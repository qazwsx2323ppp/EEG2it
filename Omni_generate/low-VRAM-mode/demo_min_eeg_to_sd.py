import os
import sys
import argparse
import numpy as np
import torch

# Allow import from Omni_generate root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.clip_models import SpatialMoEEncoder
from painter_sd import StableDiffusionPainter


def _load_eeg_tensor(path: str, default_shape=(1, 128, 512)) -> torch.Tensor:
    if not path:
        return torch.randn(*default_shape)
    if path.endswith(".npy"):
        arr = np.load(path)
        t = torch.from_numpy(arr).float()
    else:
        t = torch.load(path, map_location="cpu")
        if isinstance(t, dict) and "eeg" in t:
            t = t["eeg"]
    if t.dim() == 2:
        t = t.unsqueeze(0)
    return t.float()


def main():
    parser = argparse.ArgumentParser(description="Minimal EEG -> (prompt + EEG visual) -> SD demo")
    parser.add_argument("--eeg_ckpt", type=str, default=os.environ.get("EEG_ENCODER_CKPT", ""))
    parser.add_argument("--eeg_path", type=str, default=os.environ.get("EEG_TENSOR_PATH", ""))
    parser.add_argument("--prompt", type=str, default=os.environ.get("EEG_PROMPT_TEXT", "a photo of an object"))
    parser.add_argument(
        "--sd_model",
        type=str,
        default=os.environ.get(
            "SD_MODEL_ID",
            "/media/wsqlab/data/ctp_file/EEG2it/temp/sd15-diffusers",
        ),
    )
    parser.add_argument(
        "--sd_config",
        type=str,
        default=os.environ.get("SD_CONFIG", ""),
    )
    parser.add_argument("--out", type=str, default=os.environ.get("SD_OUTPUT_PATH", "output_image_sd_eeg.png"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1) EEG encoder
    eeg_encoder = SpatialMoEEncoder(
        n_channels=128,
        n_samples=512,
        embedding_dim=512,
        pretrained_path=None,
    ).to(device).half()

    if args.eeg_ckpt:
        if not os.path.exists(args.eeg_ckpt):
            raise FileNotFoundError(f"EEG encoder checkpoint not found: {args.eeg_ckpt}")
        state = torch.load(args.eeg_ckpt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        eeg_encoder.load_state_dict(state, strict=False)

    eeg_encoder.eval()

    # 2) Load EEG tensor
    eeg = _load_eeg_tensor(args.eeg_path).to(device).half()

    with torch.no_grad():
        emb_img, _, _ = eeg_encoder(eeg)

    # 3) SD painter
    torch_load_args = {"weights_only": False} if args.sd_model.endswith(".ckpt") else None
    painter = StableDiffusionPainter(
        model_id=args.sd_model,
        device=str(device),
        torch_dtype=torch.float16,
        original_config_file=(args.sd_config or None),
        torch_load_args=torch_load_args,
    )

    # 4) EEG -> SD projector (untrained by default)
    sd_hidden_size = painter.pipe.text_encoder.config.hidden_size
    eeg_img_proj = torch.nn.Linear(512, sd_hidden_size).to(device).half()

    # Optional load
    eeg_img_proj_ckpt = os.environ.get("EEG_IMG_PROJ_CKPT", "").strip()
    if eeg_img_proj_ckpt:
        if not os.path.exists(eeg_img_proj_ckpt):
            raise FileNotFoundError(f"EEG_IMG_PROJ_CKPT not found: {eeg_img_proj_ckpt}")
        eeg_img_proj.load_state_dict(torch.load(eeg_img_proj_ckpt, map_location="cpu"))

    # 5) Generate image
    image = painter.generate_with_eeg(
        prompt=args.prompt,
        eeg_img_emb=emb_img,
        eeg_proj=eeg_img_proj,
        negative_prompt="blurry, low quality, distorted",
        num_inference_steps=25,
        guidance_scale=7.5,
        height=512,
        width=512,
        seed=args.seed,
    )

    image.save(args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
