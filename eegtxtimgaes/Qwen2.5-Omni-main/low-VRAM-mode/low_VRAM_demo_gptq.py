from modeling_qwen2_5_omni_low_VRAM_mode import Qwen2_5OmniForConditionalGeneration

from transformers import Qwen2_5OmniProcessor
from transformers.utils.hub import cached_file

from gptqmodel import GPTQModel
from gptqmodel.models.base import BaseGPTQModel
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models._const import CPU, SUPPORTED_MODELS
from huggingface_hub import snapshot_download

# from qwen_omni_utils import process_mm_info # No longer needed for pure EEG-Text
from typing import Any, Dict

import torch
import time
import sys
import os

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.clip_models import SpatialMoEEncoder
from painter_sd import StableDiffusionPainter

model_path = "Qwen/Qwen2.5-Omni-7B-GPTQ-Int4"
# model_path = snapshot_download(repo_id=model_path) # if you use local model file, delete this line

class Qwen25OmniThinkerGPTQ(BaseGPTQModel):
    loader = Qwen2_5OmniForConditionalGeneration
    base_modules = [
        "thinker.model.embed_tokens", 
        "thinker.model.norm", 
        # "token2wav", # Removed
        # "thinker.audio_tower", 
        "thinker.model.rotary_emb",
        "thinker.visual", 
        # "talker" # Removed
    ]
    pre_lm_head_norm_module = "thinker.model.norm"
    require_monkeypatch = False
    layers_node = "thinker.model.layers"
    layer_type = "Qwen2_5OmniDecoderLayer"
    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]
   
    def pre_quantize_generate_hook_start(self):
        self.thinker.visual = move_to(self.thinker.visual, device=self.quantize_config.device)
        # self.thinker.audio_tower = move_to(self.thinker.audio_tower, device=self.quantize_config.device)

    def pre_quantize_generate_hook_end(self):
        self.thinker.visual = move_to(self.thinker.visual, device=CPU)
        # self.thinker.audio_tower = move_to(self.thinker.audio_tower, device=CPU)

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample
    

MODEL_MAP["qwen2_5_omni"] = Qwen25OmniThinkerGPTQ
SUPPORTED_MODELS.extend(["qwen2_5_omni"])

@classmethod
def patched_from_config(cls, config, *args, **kwargs):
    kwargs.pop("trust_remote_code", None)

    model = cls._from_config(config, **kwargs)
    # Speaker loading removed as Talker is disabled
    # spk_path = cached_file(...)
    # model.load_speakers(spk_path)

    return model

Qwen2_5OmniForConditionalGeneration.from_config = patched_from_config

device_map = {
    "thinker.model": "cuda", 
    "thinker.lm_head": "cuda", 
    "thinker.visual": "cpu",  
    # "thinker.audio_tower": "cpu",  
    # "talker": "cuda",  # Removed
    # "token2wav": "cuda",  # Removed
}


# GPTQ MODEL
model = GPTQModel.load(
    model_path, 
    device_map=device_map, 
    torch_dtype=torch.float16,   
    attn_implementation="flash_attention_2"
)
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

# --- EEG Component Initialization ---
print("Initializing EEG Components...")
# Initialize EEG Encoder (SpatialMoEEncoder)
# Note: Adjust parameters (n_channels, n_samples) to match your EEG data
eeg_encoder = SpatialMoEEncoder(
    n_channels=128, 
    n_samples=512, 
    embedding_dim=512,
    # pretrained_path="path/to/dreamdiffusion/checkpoint.pt" # Optional
).to("cuda").half() # Use half precision to match model

# Initialize Projector (Linear: 512 -> Hidden Size)
# This aligns the EEG embedding dimension with the Thinker's hidden size
thinker_hidden_size = model.config.thinker_config.hidden_size
eeg_projector = torch.nn.Linear(512, thinker_hidden_size).to("cuda").half()

# Attach components to the model
model.model.eeg_encoder = eeg_encoder
model.model.eeg_projector = eeg_projector
# Ensure tokenizer is available for token addition if needed
# model.model.tokenizer = processor.tokenizer 

print("EEG Components Initialized.")

# Initialize Stable Diffusion Painter
print("Initializing Stable Diffusion Painter...")
painter = StableDiffusionPainter(
    model_id="runwayml/stable-diffusion-v1-5", # Or a local path
    device="cuda",
    torch_dtype=torch.float16,
)
print("Painter Initialized.")

def eeg_inference(eeg_data, prompt_text="Describe the image decoded from brain signals."):
    """
    Perform inference using EEG data to generate text.
    """
    print(f"Input EEG Data Shape: {eeg_data.shape}")
    
    # Ensure inputs are on the correct device and dtype
    eeg_data = eeg_data.to("cuda").half()
    
    # Use the generate_from_eeg method added to Qwen2_5OmniForConditionalGeneration
    # If using the GPTQ wrapper, access the underlying model via model.model
    # We pass the tokenizer so it can handle the <EEG> token automatically
    
    gen_ids = model.model.generate_from_eeg(
        eeg_input=eeg_data,
        tokenizer=processor.tokenizer,
        prompt_text=prompt_text,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    # Decode generated tokens to text
    response = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
    return response


# --- Demo Execution ---
# Create dummy EEG data [Batch, Channels, Time]
dummy_eeg = torch.randn(1, 128, 512) 

print("Starting EEG Inference Demo...")
torch.cuda.reset_peak_memory_stats()
start = time.time()

try:
    response = eeg_inference(dummy_eeg, prompt_text="Please describe what you see.")
    print("\nGenerated Response:")
    print(response)
    
    # Generate Image from Response
    print("\nGenerating Image from Response...")
    image = painter.generate(
        prompt=response,
        negative_prompt="blurry, low quality, distorted",
        num_inference_steps=25,
        guidance_scale=7.5,
        height=512,
        width=512,
        seed=42, # Fixed seed for reproducibility, remove for random
    )
    output_image_path = "./output_image_sd.png"
    image.save(output_image_path)
    print(f"Image saved to {output_image_path}")
    
except Exception as e:
    print(f"\nError during inference: {e}")
    import traceback
    traceback.print_exc()

end = time.time()
peak_memory = torch.cuda.max_memory_allocated()

print(f"\nTotal Inference Time: {end-start:.2f} s.")
print(f"Peak GPU Memory Used: {peak_memory / 1024 / 1024:.2f} MB")

