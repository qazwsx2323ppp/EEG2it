import torch
from torch import nn

class EEGQFormer(nn.Module):
    def __init__(self, hidden_size, kv_dim=1024, num_queries=4, num_layers=2, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.kv_proj = nn.Linear(kv_dim, hidden_size)
        self.query_embed = nn.Parameter(torch.randn(num_queries, hidden_size))
        self.self_attn = nn.ModuleList([nn.MultiheadAttention(hidden_size, num_heads, batch_first=True) for _ in range(num_layers)])
        self.cross_attn = nn.ModuleList([nn.MultiheadAttention(hidden_size, num_heads, batch_first=True) for _ in range(num_layers)])
        self.norm_q1 = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.norm_q2 = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.norm_q3 = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(hidden_size, 4*hidden_size), nn.GELU(), nn.Linear(4*hidden_size, hidden_size)) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, kv_tokens, return_sequence=False):
        b = kv_tokens.size(0)
        q = self.query_embed.unsqueeze(0).expand(b, -1, -1)
        kv = self.kv_proj(kv_tokens)
        for i in range(len(self.self_attn)):
            q = q + self.self_attn[i](q, q, q)[0]
            q = self.norm_q1[i](q)
            q = q + self.cross_attn[i](q, kv, kv)[0]
            q = self.norm_q2[i](q)
            q = q + self.ffn[i](q)
            q = self.norm_q3[i](q)
        
        if return_sequence:
            return self.out_norm(self.out_proj(q))
            
        pooled = q.mean(dim=1)
        return self.out_norm(self.out_proj(pooled))