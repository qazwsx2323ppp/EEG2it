import os
import sys
import torch
import torch.nn as nn
from sc_mbm.mae_for_eeg import eeg_encoder

class Qwen2_5OmniEEGEncoder(nn.Module):
    def __init__(self, hidden_size: int, num_queries: int, backbone_kwargs: dict):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_queries = num_queries
        self.backbone = eeg_encoder(**backbone_kwargs)
        in_chans = int(backbone_kwargs.get("in_chans", 128))
        if self.backbone is not None:
            embed_dim = int(backbone_kwargs.get("embed_dim", 1024))
            self.proj = nn.Linear(embed_dim, hidden_size)
            self.raw_proj = None
        else:
            self.raw_proj = nn.Conv1d(in_chans, hidden_size, kernel_size=1, stride=1)
            self.proj = None
        self.query = nn.Parameter(torch.randn(num_queries, hidden_size))
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=int(backbone_kwargs.get("num_heads", 8)), batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, 4 * hidden_size), nn.GELU(), nn.Linear(4 * hidden_size, hidden_size))

    def forward(self, eeg_raw: torch.Tensor) -> torch.Tensor:
        if eeg_raw.dim() != 3:
            raise ValueError("eeg_raw must have shape (batch, channels, time)")
        if self.backbone is not None:
            feats = self.backbone(eeg_raw)
            feats = self.proj(feats)
        else:
            feats = self.raw_proj(eeg_raw).transpose(1, 2).contiguous()
        q = self.query.unsqueeze(0).expand(eeg_raw.size(0), -1, -1)
        out, _ = self.attn(q, feats, feats)
        out = self.norm(out + q)
        out = self.norm(self.mlp(out) + out)
        return out

def build_eeg_tower(hidden_size: int, num_queries: int, backbone_kwargs: dict):
    return Qwen2_5OmniEEGEncoder(hidden_size=hidden_size, num_queries=num_queries, backbone_kwargs=backbone_kwargs)