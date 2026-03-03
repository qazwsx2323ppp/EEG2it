import os
import json
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from newmodel.clip_models import SpatialMoEEncoder
from newmodel.dataset import TripletDataset  # 你已有的三元组数据集
from omegaconf import OmegaConf


class EEGCaptionDataset(Dataset):
    def __init__(self, cfg_data, captions_path: str, tokenizer, max_len: int = 64, split: str = "train"):
        # 复用你现有的 TripletDataset（保证 EEG 预处理逻辑一致）
        self.triplet = TripletDataset(cfg_data, mode=split, split_index=cfg_data.get("split_index", 0))
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 加载 caption 映射：eeg_index -> caption
        self.caption_map = {}
        with open(captions_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                idx = int(item["index"])
                self.caption_map[idx] = item["caption"]

        # TripletDataset 里保存的是 eeg 原始 index 列表
        self.valid_indices = []
        for eeg_idx in self.triplet.indices:
            if eeg_idx in self.caption_map:
                self.valid_indices.append(eeg_idx)

        print(f"EEGCaptionDataset[{split}] usable EEG indices: {len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, i):
        eeg_idx = self.valid_indices[i]
        # TripletDataset 通过内部 indices 映射到真实索引
        # 这里简单暴力：找到它在 indices 里的位置
        inner_idx = self.triplet.indices.index(eeg_idx)
        eeg_signal, image_vec, text_vec = self.triplet[inner_idx]

        caption = self.caption_map[eeg_idx]
        tokenized = self.tokenizer(
            caption,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        # [1, L] -> [L]
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        return eeg_signal, input_ids, attention_mask


def collate_fn(batch):
    eegs, ids, masks = zip(*batch)
    eegs = torch.stack(eegs, dim=0)           # [B, C, T]
    ids = torch.stack(ids, dim=0)             # [B, L]
    masks = torch.stack(masks, dim=0)         # [B, L]
    return eegs, ids, masks


class EEGTextGenerator(nn.Module):
    """
    冻结你的 SpatialMoEEncoder，只训练一个小的文本解码头（基于预训练 LM）。
    """
    def __init__(self, encoder_cfg, pretrained_eeg_ckpt: str,
                 lm_name: str = "gpt2", z_dim: int = 512):
        super().__init__()
        # 1. EEG encoder: 只做特征抽取
        self.encoder = SpatialMoEEncoder(
            n_channels=encoder_cfg.n_channels,
            n_samples=encoder_cfg.n_samples,
            embedding_dim=encoder_cfg.embedding_dim,
            pretrained_path=encoder_cfg.pretrained_path,
        )
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        # 2. 预训练语言模型（decoder）
        self.lm = AutoModelForCausalLM.from_pretrained(lm_name)
        self.lm_emb_dim = self.lm.config.n_embd

        # 3. 把 z_sem 映射到一个“前置 token”的 embedding
        self.z_to_prefix = nn.Linear(z_dim, self.lm_emb_dim)

    @torch.no_grad()
    def encode_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        # eeg: [B, C, T]
        img_emb, txt_emb, _ = self.encoder(eeg)
        # 统一用 img_emb 作为 z_sem；如果你更信任 txt_emb，可以改这里
        z_sem = img_emb
        return z_sem

    def forward(self, eeg, input_ids, attention_mask):
        """
        把 z_sem 接到语言模型前面当成一个 prefix token。
        """
        B, L = input_ids.shape
        device = input_ids.device

        with torch.no_grad():
            z_sem = self.encode_eeg(eeg.to(device))   # [B, z_dim]

        prefix_emb = self.z_to_prefix(z_sem)          # [B, D]
        prefix_emb = prefix_emb.unsqueeze(1)          # [B, 1, D]

        # 原始 token embedding
        inputs_embeds = self.lm.transformer.wte(input_ids)  # [B, L, D]
        # 拼接 prefix
        inputs_with_prefix = torch.cat([prefix_emb, inputs_embeds], dim=1)  # [B, 1+L, D]

        # attention_mask 也要扩一位
        prefix_mask = torch.ones(B, 1, dtype=attention_mask.dtype, device=device)
        attn_with_prefix = torch.cat([prefix_mask, attention_mask], dim=1)  # [B, 1+L]

        # Shift labels 以避免 prefix 位置计算 loss
        # label 长度和 input_ids 一致，但让 prefix 的位置忽略（-100）
        labels = input_ids.clone()
        labels = torch.cat(
            [torch.full((B, 1), -100, dtype=labels.dtype, device=device), labels],
            dim=1
        )

        out = self.lm(
            inputs_embeds=inputs_with_prefix,
            attention_mask=attn_with_prefix,
            labels=labels,
        )
        return out.loss


def train_text_decoder(
    cfg_path: str,
    captions_path: str,
    eeg_encoder_ckpt: str,
    lm_name: str = "gpt2",
    batch_size: int = 8,
    lr: float = 1e-4,
    epochs: int = 3,
    device: str = "cuda"
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # 载入你在 Hydra / yaml 里的 encoder 超参（可以简单用一个 OmegaConf）
    cfg = OmegaConf.load(cfg_path)
    encoder_cfg = cfg.model  # 假设有 n_channels, n_samples, embedding_dim, pretrained_path 等字段

    tokenizer = AutoTokenizer.from_pretrained(lm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = EEGCaptionDataset(cfg.data, captions_path, tokenizer, split="train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn)

    model = EEGTextGenerator(encoder_cfg, eeg_encoder_ckpt,
                             lm_name=lm_name, z_dim=encoder_cfg.embedding_dim).to(device)
    optimizer = torch.optim.AdamW(model.z_to_prefix.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for eeg, input_ids, attention_mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            eeg = eeg.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            loss = model(eeg, input_ids, attention_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] avg loss = {avg_loss:.4f}")

    save_path = "text_decoder_from_eeg.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "tokenizer_name": lm_name,
        "cfg_path": cfg_path,
    }, save_path)
    print(f"Saved text decoder to {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, required=True,
                        help="路径：你的 newmodel 的 Hydra / yaml 配置，里面至少有 model/data 部分")
    parser.add_argument("--captions_path", type=str, required=True,
                        help="包含 (index, caption) 的 captions.jsonl")
    parser.add_argument("--eeg_encoder_ckpt", type=str, default="",
                        help="如果需要额外加载你在 newmodel/main.py 训练好的 best_eeg_encoder.pth，可在这里指定（可选）")
    parser.add_argument("--lm_name", type=str, default="gpt2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    train_text_decoder(
        cfg_path=args.cfg_path,
        captions_path=args.captions_path,
        eeg_encoder_ckpt=args.eeg_encoder_ckpt,
        lm_name=args.lm_name,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
    )