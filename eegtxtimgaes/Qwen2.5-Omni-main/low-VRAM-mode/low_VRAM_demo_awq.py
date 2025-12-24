import torch
import time
import sys
import importlib.util
import soundfile as sf
import os
from omegaconf import OmegaConf

from awq.models.base import BaseAWQForCausalLM
from transformers import AutoProcessor
from transformers import Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from huggingface_hub import hf_hub_download

from modeling_qwen2_5_omni_low_VRAM_mode import (
    Qwen2_5OmniDecoderLayer
)
from modeling_qwen2_5_omni_low_VRAM_mode import Qwen2_5OmniForConditionalGeneration

def replace_transformers_module():
    original_mod_name = 'transformers.models.qwen2_5_omni.modeling_qwen2_5_omni'
    
    new_mod_path = 'modeling_qwen2_5_omni_low_VRAM_mode.py'

    if original_mod_name in sys.modules:
        del sys.modules[original_mod_name]

    spec = importlib.util.spec_from_file_location(original_mod_name, new_mod_path)
    new_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(new_mod)

    sys.modules[original_mod_name] = new_mod

replace_transformers_module()

class Qwen2_5_OmniAWQForConditionalGeneration(BaseAWQForCausalLM):
    layer_type = "Qwen2_5OmniDecoderLayer"
    max_seq_len_key = "max_position_embeddings"
    modules_to_not_convert = ["visual"]
    @staticmethod
    def get_model_layers(model: "Qwen2_5OmniForConditionalGeneration"):
        return model.thinker.model.layers

    @staticmethod
    def get_act_for_scaling(module: "Qwen2_5OmniForConditionalGeneration"):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model: "Qwen2_5OmniForConditionalGeneration", device: str):
        model.thinker.model.embed_tokens = model.thinker.model.embed_tokens.to(device)
        model.thinker.visual = model.thinker.visual.to(device)
        model.thinker.audio_tower = model.thinker.audio_tower.to(device)
        
        model.thinker.visual.rotary_pos_emb = model.thinker.visual.rotary_pos_emb.to(device)
        model.thinker.model.rotary_emb = model.thinker.model.rotary_emb.to(device)
        
        for layer in model.thinker.model.layers:
            layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(device)
        
    @staticmethod
    def get_layers_for_scaling(
        module: "Qwen2_5OmniDecoderLayer", input_feat, module_kwargs
    ):
        layers = []

        # attention input
        layers.append(
            dict(
                prev_op=module.input_layernorm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )

        # attention out
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            layers.append(
                dict(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                )
            )

        # linear 1
        layers.append(
            dict(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                module2inspect=module.mlp,
            )
        )

        # linear 2
        layers.append(
            dict(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
            )
        )

        return layers

device_map = {
    "thinker.model": "cuda", 
    "thinker.lm_head": "cuda", 
    "thinker.visual": "cpu",  
    "thinker.audio_tower": "cpu",  
    "talker": "cuda",  
    "token2wav": "cuda",  
}
device = 'cuda'

model_path = "Qwen/Qwen2.5-Omni-7B-AWQ"

model = Qwen2_5_OmniAWQForConditionalGeneration.from_quantized(  
                                            model_path, 
                                            model_type="qwen2_5_omni",
                                            # device_map=device_map, 
                                            torch_dtype=torch.float16,   
                                            attn_implementation="flash_attention_2"
                                        )

# spk_path = model_path + '/spk_dict.pt' # use this line if you load model from local
spk_path = hf_hub_download(repo_id=model_path, filename='spk_dict.pt')

model.model.load_speakers(spk_path)

model.model.thinker.model.embed_tokens = model.model.thinker.model.embed_tokens.to(device)
model.model.thinker.visual = model.model.thinker.visual.to(device)
model.model.thinker.audio_tower = model.model.thinker.audio_tower.to(device)
model.model.thinker.visual.rotary_pos_emb = model.model.thinker.visual.rotary_pos_emb.to(device)
model.model.thinker.model.rotary_emb = model.model.thinker.model.rotary_emb.to(device)

for layer in model.model.thinker.model.layers:
    layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(device)


processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

def video_inference(video_path, prompt, sys_prompt):
    messages = [
        {"role": "system", "content": [
                {"type": "text", "text": sys_prompt},
            ]},
        {"role": "user", "content": [
                {"type": "video", "video": video_path},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True)
    inputs = inputs.to('cuda')
    

    output = model.generate(**inputs, use_audio_in_video=True, return_audio=True)
    text = processor.batch_decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    audio = output[2]
    return text, audio


video_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4"
system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

torch.cuda.reset_peak_memory_stats()
start = time.time()
response, audio  = video_inference(video_path, prompt=None, sys_prompt=system_prompt)
end = time.time()
peak_memory = torch.cuda.max_memory_allocated()

audio_file_path = "./output_audio_awq.wav"
sf.write(
    audio_file_path,
    audio.reshape(-1).detach().cpu().numpy(),
    samplerate=24000,
)

print(response[0])
print(f"Total Inference Time: {end-start:.2f} s.")
print(f"Peak GPU Memory Used: {peak_memory / 1024 / 1024:.2f} MB")
###############测试eeg
def test_eeg_pipeline():
    print("\n\n=== Testing EEG Pipeline ===")
    
    # Try to load real data configuration
    config_path = "../configs/triplet_config.yaml"
    real_data_loaded = False
    eeg_input = None
    
    if os.path.exists(config_path):
        try:
            print(f"Loading config from {config_path}...")
            cfg = OmegaConf.load(config_path)
            
            # Add parent dir to sys.path to import dataset_2o
            sys.path.append("..") 
            from dataset_2o import TripletDataset
            
            # Initialize Dataset
            # Note: This might fail if data paths in yaml are incorrect
            dataset = TripletDataset(cfg.data, mode='test', split_index=0)
            print(f"Dataset loaded successfully! Size: {len(dataset)}")
            
            if len(dataset) > 0:
                eeg_data, _, _ = dataset[0]
                eeg_input = eeg_data.unsqueeze(0).to(device)
                print(f"Loaded Real EEG Data: {eeg_input.shape}")
                real_data_loaded = True
                
        except Exception as e:
            print(f"Could not load real data (using dummy data instead): {e}")
    else:
        print(f"Config file not found at {config_path}")

    if not real_data_loaded:
        print("Using Dummy EEG Data...")
        # Shape: [Batch=1, Channels=128, Time=512]
        eeg_input = torch.randn(1, 128, 512).to(device)
    
    print(f"1. EEG Input Ready: {eeg_input.shape}")
    
    try:
        # 2. Get EEG Embeddings
        eeg_embeds = model.model.generate_from_eeg(eeg_input)
        print(f"2. EEG Embeddings Generated: {eeg_embeds.shape}")
        
        # 3. Construct Text Prompt
        prompt = "Describe the image decoded from brain signals."
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text_input, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(device)
        input_embeds = model.model.get_input_embeddings()(input_ids)
        
        # 4. Concatenate
        final_embeds = torch.cat([eeg_embeds, input_embeds], dim=1)
        attention_mask = torch.ones(final_embeds.shape[:2], device=device)
        
        print(f"3. Final Input Embeddings: {final_embeds.shape}")
        
        # 5. Generate
        print("4. Running Generation...")
        output_ids = model.generate(
            inputs_embeds=final_embeds, 
            attention_mask=attention_mask, 
            max_new_tokens=50
        )
        
        # 6. Decode
        generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        print(f"5. Generation Success!\nOutput: {generated_text}")
        print("=== EEG Pipeline Verified ===")
        
    except Exception as e:
        print(f"!!! Pipeline Failed: {e}")
        import traceback
        traceback.print_exc()

# Uncomment to run the test
test_eeg_pipeline()
###############