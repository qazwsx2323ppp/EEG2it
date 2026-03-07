import os
import sys
import argparse
import numpy as np
import torch

# Allow import from Omni_generate root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.clip_models import SpatialMoEEncoder
from painter_sd import StableDiffusionPainter


def _load_cfg(path: str) -> dict:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(path)
        return OmegaConf.to_container(cfg, resolve=True) or {}
    except Exception:
        pass
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        raise RuntimeError(f"Failed to load config: {path} ({e})")


def _cfg_get(cfg: dict, keys: list[str], default=None):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


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
    parser.add_argument("--config", type=str, default=os.environ.get("EEG_GEN_CONFIG", ""))
    parser.add_argument("--eeg_ckpt", type=str, default=None)
    parser.add_argument("--eeg_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument(
        "--sd_model",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--sd_config",
        type=str,
        default=None,
    )
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = _load_cfg(args.config)

    def _pick(arg_val, env_key, cfg_keys, fallback):
        if arg_val is not None and str(arg_val).strip() != "":
            return arg_val
        cfg_val = _cfg_get(cfg, cfg_keys, None)
        if cfg_val is not None and str(cfg_val).strip() != "":
            return cfg_val
        env_val = os.environ.get(env_key, "")
        if env_val:
            return env_val
        return fallback

    eeg_ckpt = _pick(args.eeg_ckpt, "EEG_ENCODER_CKPT", ["eeg", "ckpt"], "")
    eeg_path = _pick(args.eeg_path, "EEG_TENSOR_PATH", ["data", "eeg_path"], "")
    prompt = _pick(args.prompt, "EEG_PROMPT_TEXT", ["prompt"], "a photo of an object")
    sd_model = _pick(
        args.sd_model,
        "SD_MODEL_ID",
        ["sd", "model"],
        "/media/wsqlab/data/ctp_file/EEG2it/temp/sd15-diffusers",
    )
    sd_config = _pick(args.sd_config, "SD_CONFIG", ["sd", "config"], "")
    out_path = _pick(args.out, "SD_OUTPUT_PATH", ["sd", "output"], "output_image_sd_eeg.png")
    device_name = _pick(args.device, "DEVICE", ["device"], "cuda")
    seed = int(_pick(args.seed, "SEED", ["seed"], 42))

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    # 1) EEG encoder
    eeg_encoder = SpatialMoEEncoder(
        n_channels=128,
        n_samples=512,
        embedding_dim=512,
        pretrained_path=None,
    ).to(device).half()

    if eeg_ckpt:
        if not os.path.exists(eeg_ckpt):
            raise FileNotFoundError(f"EEG encoder checkpoint not found: {eeg_ckpt}")
        state = torch.load(eeg_ckpt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        eeg_encoder.load_state_dict(state, strict=False)

    eeg_encoder.eval()

    # 2) Load EEG tensor
    eeg = _load_eeg_tensor(eeg_path).to(device).half()

    with torch.no_grad():
        emb_img, _, _ = eeg_encoder(eeg)

    # 3) SD painter
    torch_load_args = {"weights_only": False} if str(sd_model).endswith(".ckpt") else None
    painter = StableDiffusionPainter(
        model_id=sd_model,
        device=str(device),
        torch_dtype=torch.float16,
        original_config_file=(sd_config or None),
        torch_load_args=torch_load_args,
    )

    # 4) EEG -> SD projector (untrained by default)
    sd_hidden_size = painter.pipe.text_encoder.config.hidden_size
    eeg_img_proj = torch.nn.Linear(512, sd_hidden_size).to(device).half()

    # Optional load
    eeg_img_proj_ckpt = os.environ.get("EEG_IMG_PROJ_CKPT", "").strip()
    if not eeg_img_proj_ckpt:
        eeg_img_proj_ckpt = str(_cfg_get(cfg, ["eeg", "img_proj_ckpt"], "") or "").strip()
    if eeg_img_proj_ckpt:
        if not os.path.exists(eeg_img_proj_ckpt):
            raise FileNotFoundError(f"EEG_IMG_PROJ_CKPT not found: {eeg_img_proj_ckpt}")
        eeg_img_proj.load_state_dict(torch.load(eeg_img_proj_ckpt, map_location="cpu"))

    # 5) Generate image
    image = painter.generate_with_eeg(
        prompt=prompt,
        eeg_img_emb=emb_img,
        eeg_proj=eeg_img_proj,
        negative_prompt="blurry, low quality, distorted",
        num_inference_steps=25,
        guidance_scale=7.5,
        height=512,
        width=512,
        seed=seed,
    )

    image.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
