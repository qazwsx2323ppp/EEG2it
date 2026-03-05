import torch
from torch import nn

class EEGQFormer(nn.Module):
    def __init__(self, hidden_size, kv_dim=1024, num_queries=4, num_layers=2, num_heads=8, dropout=0.1, resid_scale=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.kv_proj = nn.Linear(kv_dim, hidden_size)
        self.kv_norm = nn.LayerNorm(hidden_size)
        # 每层结构：Self-Attn(q→q) → LN → Cross-Attn(q→kv) → LN → FFN → LN，2 层堆叠 out_qformer.py:L10-L15 。
        self.query_embed = nn.Parameter(torch.randn(num_queries, hidden_size))
        self.query_pos = nn.Parameter(torch.zeros(num_queries, hidden_size))
        self.self_attn = nn.ModuleList([nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True) for _ in range(num_layers)])
        self.cross_attn = nn.ModuleList([nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True) for _ in range(num_layers)])
        self.norm_q1 = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.norm_q2 = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.norm_q3 = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(hidden_size, 4*hidden_size), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*hidden_size, hidden_size), nn.Dropout(dropout)) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.resid_scale = resid_scale
        nn.init.normal_(self.query_embed, std=0.02)
        nn.init.normal_(self.query_pos, std=0.01)

    def forward(self, kv_tokens, return_sequence=False):
        b = kv_tokens.size(0)
        q = self.query_embed.unsqueeze(0).expand(b, -1, -1) + self.query_pos.unsqueeze(0).expand(b, -1, -1)
        kv = self.kv_norm(self.kv_proj(kv_tokens))
        for i in range(len(self.self_attn)):
            sa = self.self_attn[i](q, q, q, need_weights=False)[0]
            q = q + self.resid_scale * self.dropout(sa)
            q = self.norm_q1[i](q)
            ca = self.cross_attn[i](q, kv, kv, need_weights=False)[0]
            q = q + self.resid_scale * self.dropout(ca)
            q = self.norm_q2[i](q)
            ff = self.ffn[i](q)
            q = q + self.resid_scale * self.dropout(ff)
            q = self.norm_q3[i](q)
        
        if return_sequence:
            return self.out_norm(self.out_proj(q))
            
        pooled = q.mean(dim=1)
        return self.out_norm(self.out_proj(pooled))
        # 默认返回序列 q 的线性投影+层归一化，形状 [B, num_queries, hidden_size]，用于拼接入 Thinker out_qformer.py:L31-L35 。
        