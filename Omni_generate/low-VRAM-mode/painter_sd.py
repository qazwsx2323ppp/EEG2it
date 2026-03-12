import torch
import os
from diffusers import StableDiffusionPipeline

class StableDiffusionPainter:
    def __init__(
        self,
        model_id="runwayml/stable-diffusion-v1-5",
        device="cuda",
        torch_dtype=torch.float16,
        original_config_file=None,
        torch_load_args=None,
        tokenizer_dir=None,
    ):
        self.device = device
        self.dtype = torch_dtype
        # Use a local path if internet is restricted, or standard model ID.
        # Support loading from a single .ckpt/.safetensors file when no diffusers config dir exists.
        try:
            if isinstance(model_id, str) and model_id.endswith((".ckpt", ".safetensors")):
                # Some older/PL checkpoints need torch_load_args={"weights_only": False}.
                kwargs = {"torch_dtype": torch_dtype, "safety_checker": None}
                if original_config_file:
                    kwargs["original_config_file"] = original_config_file
                if torch_load_args:
                    kwargs["torch_load_args"] = torch_load_args
                self.pipe = StableDiffusionPipeline.from_single_file(model_id, **kwargs)
            else:
                if tokenizer_dir:
                    try:
                        from transformers import CLIPTokenizer
                        vocab_path = os.path.join(tokenizer_dir, "vocab.json")
                        merges_path = os.path.join(tokenizer_dir, "merges.txt")
                        if os.path.exists(vocab_path) and os.path.exists(merges_path):
                            tokenizer = CLIPTokenizer.from_pretrained(
                                tokenizer_dir, local_files_only=True
                            )
                        else:
                            tokenizer = None
                    except Exception as tok_e:
                        print(f"Failed to load tokenizer from {tokenizer_dir}: {tok_e}")
                        tokenizer = None
                else:
                    tokenizer = None
                kwargs = {
                    "torch_dtype": torch_dtype,
                    "safety_checker": None,
                }
                if tokenizer is not None:
                    kwargs["tokenizer"] = tokenizer
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    **kwargs,
                )
                # Fallback: ensure tokenizer exists (offline setups may only have tokenizer.json)
                if getattr(self.pipe, "tokenizer", None) is None:
                    try:
                        from transformers import AutoTokenizer
                        if tokenizer_dir:
                            self.pipe.tokenizer = AutoTokenizer.from_pretrained(
                                tokenizer_dir, use_fast=True, local_files_only=True
                            )
                        else:
                            self.pipe.tokenizer = AutoTokenizer.from_pretrained(
                                model_id, subfolder="tokenizer", use_fast=True, local_files_only=True
                            )
                    except Exception as tok_e:
                        print(f"Failed to load fallback tokenizer: {tok_e}")
                if getattr(self.pipe, "tokenizer", None) is None:
                    raise ValueError(
                        "Tokenizer is None. Provide --sd_tokenizer_dir with tokenizer.json or vocab/merges."
                    )
        except Exception as e:
            print(f"Failed to load Stable Diffusion model: {e}")
            # Fallback or re-raise
            raise e
            
        self.pipe = self.pipe.to(device)
        # Enable memory optimizations
        try:
            self.pipe.enable_attention_slicing()
        except Exception:
            pass

    def _encode_prompt_embeds(self, prompt, negative_prompt=None, guidance_scale=7.5, reserve_tokens=0):
        # We encode explicitly to guarantee truncation and allow reserving space
        # for extra conditioning tokens (e.g., EEG token).
        do_cfg = guidance_scale is not None and guidance_scale > 1.0
        tokenizer = self.pipe.tokenizer
        text_encoder = self.pipe.text_encoder

        max_len = int(getattr(tokenizer, "model_max_length", 77) or 77)
        if reserve_tokens:
            if reserve_tokens >= max_len:
                raise ValueError(
                    f"reserve_tokens={reserve_tokens} >= tokenizer.model_max_length={max_len}"
                )
            max_len = max_len - int(reserve_tokens)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        )
        prompt_embeds = text_encoder(text_inputs.input_ids.to(self.device))[0]

        negative_prompt_embeds = None
        if do_cfg:
            neg = negative_prompt if negative_prompt is not None else ""
            neg_inputs = tokenizer(
                neg,
                padding="max_length",
                max_length=max_len,
                truncation=True,
                return_tensors="pt",
            )
            negative_prompt_embeds = text_encoder(neg_inputs.input_ids.to(self.device))[0]

        return prompt_embeds, negative_prompt_embeds

    def generate(self, prompt, negative_prompt=None, num_inference_steps=25, guidance_scale=7.5, height=512, width=512, seed=None):
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed else None
        
        image = self.pipe(
            prompt=prompt, 
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator
        ).images[0]
        
        return image

    def generate_with_eeg(
        self,
        prompt,
        eeg_img_emb,
        eeg_proj,
        negative_prompt=None,
        num_inference_steps=25,
        guidance_scale=7.5,
        height=512,
        width=512,
        seed=None,
        eeg_token_scale=1.0,
        eeg_condition_in_uncond=True,
    ):
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed else None

        if eeg_img_emb.dim() == 1:
            eeg_img_emb = eeg_img_emb.unsqueeze(0)

        eeg_img_emb = eeg_img_emb.to(self.device, dtype=self.dtype)
        eeg_proj = eeg_proj.to(self.device, dtype=self.dtype)

        prompt_embeds, negative_prompt_embeds = self._encode_prompt_embeds(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            reserve_tokens=1,
        )

        eeg_token = eeg_proj(eeg_img_emb)
        if eeg_token_scale != 1.0:
            eeg_token = eeg_token * float(eeg_token_scale)
        eeg_token = eeg_token.unsqueeze(1)  # [B, 1, D]

        # Match batch size for CFG (prompt_embeds may be 2*B)
        if prompt_embeds.shape[0] != eeg_token.shape[0]:
            if prompt_embeds.shape[0] % eeg_token.shape[0] != 0:
                raise ValueError(
                    f"Batch mismatch: prompt_embeds={prompt_embeds.shape[0]}, eeg_token={eeg_token.shape[0]}"
                )
            repeat = prompt_embeds.shape[0] // eeg_token.shape[0]
            eeg_token = eeg_token.repeat(repeat, 1, 1)

        prompt_embeds = torch.cat([eeg_token, prompt_embeds], dim=1)

        if negative_prompt_embeds is not None:
            if eeg_condition_in_uncond:
                negative_prompt_embeds = torch.cat([eeg_token, negative_prompt_embeds], dim=1)
            else:
                zero_token = torch.zeros_like(eeg_token)
                negative_prompt_embeds = torch.cat([zero_token, negative_prompt_embeds], dim=1)

        image = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        ).images[0]

        return image
