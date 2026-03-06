#!/usr/bin/env python
"""
Extract EEG encoder weights from a StageB DreamDiffusion checkpoint.

Default behavior:
- Load checkpoint (supports PyTorch 2.6 safety changes).
- Extract parameters with prefix "cond_stage_model.mae." (or "mae.").
- Save a new .pth that only contains encoder parameters.

Example:
  python tools/extract_encoder_checkpoint.py \
    --input "/media/wsqlab/data/ctp_file/EEG2it/temp/checkpoint.pth" \
    --output "/media/wsqlab/data/ctp_file/EEG2it/temp/checkpoint1.pth"
"""

from __future__ import annotations

import argparse
import sys
import types
from typing import Dict

import torch


def _ensure_config_safe_globals() -> None:
    """
    For PyTorch 2.6+ safe deserialization, allowlist Config_Generative_Model
    if present, or create a dummy class to satisfy pickled objects.
    """
    try:
        import config  # type: ignore

        cls = getattr(config, "Config_Generative_Model", None)
        if cls is not None and hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([cls])
            return
    except Exception:
        pass

    # Fallback: create a dummy config module/class
    dummy_module = types.ModuleType("config")

    class DummyConfig:
        pass

    DummyConfig.__module__ = "config"
    dummy_module.Config_Generative_Model = DummyConfig
    sys.modules["config"] = dummy_module

    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([DummyConfig])


def _torch_load_any(path: str):
    _ensure_config_safe_globals()
    # Try weights_only=False (PyTorch 2.6+ safety change)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older torch without weights_only arg
        return torch.load(path, map_location="cpu")


def _select_state_dict(ckpt) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        if "model" in ckpt:
            return ckpt["model"]
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint format not recognized: expected dict-like object.")
    return ckpt


def extract_encoder_state(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    target_prefix = "cond_stage_model.mae."
    alt_prefix = "mae."
    new_state: Dict[str, torch.Tensor] = {}

    for k, v in state_dict.items():
        k_clean = k
        if k_clean.startswith("module."):
            k_clean = k_clean[len("module.") :]

        if k_clean.startswith(target_prefix):
            new_state[k_clean[len(target_prefix) :]] = v
        elif k_clean.startswith(alt_prefix):
            new_state[k_clean[len(alt_prefix) :]] = v

    return new_state


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract EEG encoder weights from StageB checkpoint.")
    parser.add_argument("--input", required=True, help="Path to StageB checkpoint.pth")
    parser.add_argument("--output", required=True, help="Output path for encoder-only checkpoint")
    parser.add_argument(
        "--wrap-key",
        default="",
        help="If set, wrap output state_dict into a dict under this key (e.g. 'model').",
    )
    args = parser.parse_args()

    ckpt = _torch_load_any(args.input)
    state_dict = _select_state_dict(ckpt)
    enc_state = extract_encoder_state(state_dict)

    if not enc_state:
        print("ERROR: No encoder parameters found. Check prefixes or input checkpoint.")
        return 2

    to_save = {args.wrap_key: enc_state} if args.wrap_key else enc_state
    torch.save(to_save, args.output)
    print(f"Saved {len(enc_state)} encoder params to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
