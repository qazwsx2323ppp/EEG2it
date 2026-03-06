import torch
from diffusers import StableDiffusionPipeline

class StableDiffusionPainter:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device="cuda", torch_dtype=torch.float16):
        self.device = device
        self.dtype = torch_dtype
        # Use a local path if internet is restricted, or standard model ID.
        # Support loading from a single .ckpt/.safetensors file when no diffusers config dir exists.
        try:
            if isinstance(model_id, str) and model_id.endswith((".ckpt", ".safetensors")):
                self.pipe = StableDiffusionPipeline.from_single_file(
                    model_id, torch_dtype=torch_dtype, safety_checker=None
                )
            else:
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    model_id, torch_dtype=torch_dtype, safety_checker=None
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

    def _encode_prompt_embeds(self, prompt, negative_prompt=None, guidance_scale=7.5):
        do_cfg = guidance_scale is not None and guidance_scale > 1.0
        if hasattr(self.pipe, "_encode_prompt"):
            out = self.pipe._encode_prompt(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=do_cfg,
                negative_prompt=negative_prompt,
            )
            if isinstance(out, tuple):
                prompt_embeds, negative_prompt_embeds = out
            else:
                prompt_embeds, negative_prompt_embeds = out, None
            return prompt_embeds, negative_prompt_embeds

        # Fallback for older diffusers
        tokenizer = self.pipe.tokenizer
        text_encoder = self.pipe.text_encoder
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
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
                max_length=tokenizer.model_max_length,
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
        )

        eeg_token = eeg_proj(eeg_img_emb)
        if eeg_token_scale != 1.0:
            eeg_token = eeg_token * float(eeg_token_scale)
        eeg_token = eeg_token.unsqueeze(1)  # [B, 1, D]

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
