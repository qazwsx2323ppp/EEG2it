import torch
from diffusers import StableDiffusionPipeline

class StableDiffusionPainter:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device="cuda", torch_dtype=torch.float16):
        self.device = device
        self.dtype = torch_dtype
        # Use a local path if internet is restricted, or standard model ID
        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, safety_checker=None)
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