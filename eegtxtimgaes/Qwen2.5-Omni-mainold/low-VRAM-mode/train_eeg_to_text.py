import os
import sys
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

# 让 DreamDiffusion 的代码可导入
DREAM_CODE_PATH = r"d:\ASUS\eegtxtimgaes\DreamDiffusion-main\code"
if DREAM_CODE_PATH not in sys.path and os.path.isdir(DREAM_CODE_PATH):
    sys.path.insert(0, DREAM_CODE_PATH)

from dataset import create_EEG_dataset  # DreamDiffusion 数据集加载
from dc_ldm.ldm_for_eeg import eLDM_eval  # DreamDiffusion 的评估类（直接EEG→图像）

def add_eeg_token(processor, model, token="<eeg>"):
    tok = processor.tokenizer
    if token not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [token]})
        model.thinker.model.resize_token_embeddings(len(tok))
    return tok.convert_tokens_to_ids(token)

def build_prompt(num_queries, token="<eeg>"):
    return " ".join([token] * num_queries)

def run_once(
    model_name="Qwen/Qwen2.5-Omni-7B",
    eeg_signals_path=r"d:\ASUS\eegtxtimgaes\DreamDiffusion-main\datasets\eeg_5_95_std.pth",
    splits_path=r"d:\ASUS\eegtxtimgaes\DreamDiffusion-main\datasets\block_splits_by_image_single.pth",
    dream_config_path=r"d:\ASUS\eegtxtimgaes\DreamDiffusion-main\pretrains\models\config.yaml",
    imagenet_path=None,
    subject=4,
    sample_index=0,
    output_dir=r"d:\ASUS\eegtxtimgaes\outputs\unified",
    ddim_steps=50,
    num_samples=1
):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 加载 Omni 模型与处理器（文本分支）
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16 if device=="cuda" else torch.float32, device_map="auto"
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name)

    # 注册 <eeg> 特殊占位符，并构造提示模板
    eeg_token_index = add_eeg_token(processor, model, token="<eeg>")
    num_queries = getattr(model.thinker.config, "eeg_num_queries", 32)
    prompt_template = build_prompt(num_queries, token="<eeg>")

    # 2) 加载 DreamDiffusion 数据集（一致的数据与编码器）
    split_train, split_test = create_EEG_dataset(
        eeg_signals_path=eeg_signals_path,
        splits_path=splits_path,
        imagenet_path=imagenet_path,
        image_transform=lambda x: x,  # DreamDiffusion内部会做规范化
        subject=subject
    )
    item = split_test[sample_index]  # {'eeg': (T,C)或(C,T)，'image': HWC等}

    # 3) EEG→文本推理（Omni）
    # 取 EEG 原始张量并按 Omni 桥接要求转成 (B,C,T)
    eeg = item["eeg"]
    if eeg.ndim == 2:
        # DreamDiffusion的返回通常是 (T, C)，转成 (C, T)
        if eeg.shape[0] == 512:
            eeg_raw = eeg.t().unsqueeze(0)  # (1, C, T)
        else:
            # 若是 (C, T)，直接加 batch 维
            eeg_raw = eeg.unsqueeze(0)
    else:
        raise ValueError("Unexpected EEG shape, expected 2D tensor")

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are Qwen Omni with EEG input."}]},
        {"role": "user", "content": [{"type": "text", "text": f"Analyze EEG: {prompt_template}"}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    text_ids = model.generate(
        **inputs,
        eeg_values=eeg_raw.to(model.device).to(model.dtype),
        eeg_token_index=eeg_token_index,
        max_new_tokens=128,
    )
    response = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    with open(os.path.join(output_dir, "eeg_to_text.txt"), "w", encoding="utf-8") as f:
        f.write(response)

    # 4) EEG→图像推理（DreamDiffusion 原生）
    # 使用他们的评估类，直接以 EEG 为条件生成图像；会用到 item['image'] 作为GT拼图
    ldm = eLDM_eval(
        config_path=dream_config_path,
        num_voxels=440,  # 与其cond_stage_model默认一致
        device=torch.device(device),
        global_pool=True,
        clip_tune=True,
        cls_tune=False
    )
    grid, samples = ldm.generate(
        fmri_embedding=[{"eeg": item["eeg"], "image": item["image"]}],
        num_samples=num_samples,
        ddim_steps=ddim_steps,
        output_path=output_dir
    )
    # 图像已由 eLDM_eval.generate 在 output_dir 下按 test{idx}-{copy}.png 保存

    print("Unified inference done.")
    print(f"Text saved to: {os.path.join(output_dir,'eeg_to_text.txt')}")
    print(f"Images saved under: {output_dir}")

if __name__ == "__main__":
    run_once()