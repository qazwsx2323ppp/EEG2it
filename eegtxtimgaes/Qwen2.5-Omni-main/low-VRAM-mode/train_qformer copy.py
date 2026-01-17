import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os
from tqdm import tqdm
from omegaconf import OmegaConf



from dataset import TripletDataset
from transformers import Qwen2_5OmniProcessor
from modeling_qwen2_5_omni_low_VRAM_mode import Qwen2_5OmniForConditionalGeneration
from models.clip_models import SpatialMoEEncoder
from out_qformer import EEGQFormer

def train_qformer():
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4        # 显存允许的话调大
    epochs = 10           # Q-Former 可能需要多训练几轮
    lr = 5e-5             # Transformer 通常使用较小的 LR
    save_path = "eeg_qformer.pth"
    

    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "triplet_config.yaml")
    cfg = OmegaConf.load(config_path)
    cfg.data.root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    print(">>> 正在初始化模型...")
    

    
    # A. Qwen (LLM) - 冻结
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B",
        torch_dtype=torch.float32, # 临时改为 float32 避免精度问题
        device_map="auto"
    ).eval()
    
    for param in model.parameters():
        param.requires_grad = False
        
  
    eeg_encoder = SpatialMoEEncoder(
        n_channels=128,
        n_samples=512,
        embedding_dim=512,
        pretrained_path=r"best_12.8_change.pth"
    ).to(device).eval()
    
    for param in eeg_encoder.parameters():
        param.requires_grad = False
        
    # C. Q-Former (训练目标) - 激活
    thinker_hidden_size = model.config.thinker_config.text_config.hidden_size
    qformer = EEGQFormer(
        hidden_size=thinker_hidden_size, # 3584
        kv_dim=512,                      # EEG Encoder 输出维度
        num_queries=16,                  # 生成 16 个 Query Token
        num_layers=2,
        num_heads=8
    ).to(device).train()
    

    model.qformer = qformer
    model.use_qformer = True
    
    # 优化器只优化 Q-Former 的参数
    optimizer = optim.AdamW(qformer.parameters(), lr=lr, weight_decay=0.01)
    
    
    print(">>> 正在加载数据...")
    # 必须设置 return_text=True
    dataset = TripletDataset(cfg.data, mode='train', split_index=0, return_text=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    print(f">>> 开始训练 Q-Former, 共 {len(dataset)} 个样本...")
    
    
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            eeg_signal, img_vecs, txt_vecs, text_captions = batch
            
            # 过滤无效文本
            valid_indices = [i for i, t in enumerate(text_captions) if t]
            if not valid_indices:
                continue
                
            eeg_signal = eeg_signal[valid_indices].to(device)
            text_captions = [text_captions[i] for i in valid_indices]
            
            # 1. 获取 EEG 特征 (img_emb, txt_emb)
            with torch.no_grad():
                # Encoder 返回 (img_emb, txt_emb, weights)
                emb_img, emb_txt, _ = eeg_encoder(eeg_signal)
                
            # 2. 准备 Q-Former 输入
            # 将 img 和 txt 特征堆叠作为 KV
            # shape: (Batch, 2, 512)
            kv_tokens = torch.stack([emb_img, emb_txt], dim=1)
            
            # 3. Q-Former 前向传播
            # 返回: (Batch, num_queries, Hidden)
            eeg_embeds = qformer(kv_tokens, return_sequence=True) # 必须 return_sequence=True
            
            # 4. 构建 LLM 输入
            text_inputs = tokenizer(text_captions, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            text_token_embeds = model.get_input_embeddings()(text_inputs.input_ids)
            
            # 拼接: [EEG_Embeds, Text_Embeds]
            inputs_embeds = torch.cat([eeg_embeds, text_token_embeds], dim=1)
            
            # 5. 构建 Labels 和 Attention Mask
            # EEG 部分设为 -100
            batch_size_curr = inputs_embeds.shape[0]
            num_eeg_tokens = eeg_embeds.shape[1] # num_queries
            
            # Label 部分
            ignore_labels = torch.full((batch_size_curr, num_eeg_tokens), -100, dtype=torch.long, device=device)
            labels = torch.cat([ignore_labels, text_inputs.input_ids], dim=1)
            
            # Mask 部分
            # EEG tokens 都是有效的，所以 mask 是 1
            eeg_attention_mask = torch.ones((batch_size_curr, num_eeg_tokens), dtype=torch.long, device=device)
            # 拼接: [EEG_Mask, Text_Mask]
            attention_mask = torch.cat([eeg_attention_mask, text_inputs.attention_mask], dim=1)
            
            # 6. 计算 Loss
            # --- DEBUG START ---
            if batch_idx == 0:
                print(f"\n[DEBUG] Input Embeds Shape: {inputs_embeds.shape}")
                print(f"[DEBUG] Labels Shape: {labels.shape}")
                print(f"[DEBUG] Non-ignore labels count: {(labels != -100).sum().item()}")
                print(f"[DEBUG] Sample Label: {labels[0, -10:]}") # 打印最后10个 token 的 label
                
                # 检查 inputs_embeds 是否有 NaN
                if torch.isnan(inputs_embeds).any():
                    print("[ERROR] Input Embeds contains NaN!")
            # --- DEBUG END ---

            outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # --- DEBUG LOSS ---
            if loss.item() == 0.0:
                 print(f"[WARNING] Zero Loss detected at batch {batch_idx}!")
            # ------------------
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        # 保存
        torch.save(qformer.state_dict(), save_path)
        print(f">>> 权重已保存至 {save_path}")

if __name__ == "__main__":
    train_qformer()