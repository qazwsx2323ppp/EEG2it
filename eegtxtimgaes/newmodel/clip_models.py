# models/clip_models.py

import torch
from torch import nn
import torch.nn.functional as F
from models.ddpt_model import MAEforEEG
import math
import sys
import types

# ======================= 基础功能模块 =======================

class TemporalAttnPool(nn.Module):
    """EEG 时序注意力池化"""
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, 1)

    def forward(self, x):
        # x: [B, N, D]
        w = self.q(x)                  # [B, N, 1]
        w = torch.softmax(w, dim=1)
        return (w * x).sum(dim=1)


class CLIPResidualHead(nn.Module):
    """CLIP 风格残差投影头"""
    def __init__(self, d_in, d_out):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.GELU(),
            nn.Linear(d_in, d_out)
        )
        self.res = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.proj(x) + self.res(x)


class Adapter(nn.Module):
    """轻量级 Adapter 微调模块"""
    def __init__(self, dim, bottleneck=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, dim)
        )

    def forward(self, x):
        return x + self.net(x)


class EEGTokenRouter(nn.Module):
    """Token-wise MoE Router"""
    def __init__(self, dim, num_experts=3):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_experts)
        )

    def forward(self, x):
        # x: [B, N, D]
        return torch.softmax(self.router(x), dim=-1)


# ======================= 主模型 =======================

class SpatialMoEEncoder(nn.Module):
    def __init__(
        self,
        n_channels,
        n_samples,
        embedding_dim=512,
        pretrained_path=None
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.embedding_dim = embedding_dim

        # ========== 1. DreamDiffusion Backbone ==========
        self.backbone = MAEforEEG(
            time_len=512,
            in_chans=128,
            patch_size=4,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            decoder_embed_dim=1024,
            mlp_ratio=1.0,
            decoder_depth=8,
            decoder_num_heads=16
        )

        if pretrained_path:
            print(f"Loading DreamDiffusion checkpoint from {pretrained_path}")

            class DummyConfig: pass
            dummy_config_module = types.ModuleType("config")
            dummy_config_module.Config_Generative_Model = DummyConfig
            sys.modules["config"] = dummy_config_module

            checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            # Stage B 存的是 model_state_dict；Stage A 存的是 model
            state_dict = checkpoint.get("model_state_dict") or checkpoint.get("model")
            if state_dict is None:
                state_dict = checkpoint

            new_state_dict = {}
            prefix = "cond_stage_model.mae."
            for k, v in state_dict.items():
                if k.startswith(prefix):
                    new_state_dict[k.replace(prefix, "", 1)] = v

            msg = self.backbone.load_state_dict(new_state_dict, strict=False)
            print("Pretrained load msg:", msg)

        # ========== 2. 冻结 Backbone + Adapter ==========
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.adapter = Adapter(1024)

        # ========== 3. Token-wise MoE ==========
        self.router = EEGTokenRouter(1024, num_experts=3)

        self.expert_visual = nn.Linear(1024, 1024)
        self.expert_semantic = nn.Linear(1024, 1024)
        self.expert_fusion = nn.Linear(1024, 1024)

        # ========== 4. 时序注意力池化 ==========
        self.temporal_pool = TemporalAttnPool(1024)

        # ========== 5. CLIP 投影头 ==========
        self.img_head = CLIPResidualHead(1024, embedding_dim)
        self.txt_head = CLIPResidualHead(1024, embedding_dim)

        # ========== 6. CLIP 温度参数 ==========
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    # ======================= Forward =======================

    def forward(self, x):
        # x: [B, C, T]

        latent, _, _ = self.backbone.forward_encoder(x, mask_ratio=0.0)
        tokens = latent[:, 1:, :]                      # [B, N, 1024]

        # Adapter 微调
        tokens = self.adapter(tokens)

        # ========== Token-wise MoE ==========
        gates = self.router(tokens)                     # [B, N, 3]

        vis_feat = self.expert_visual(tokens)
        sem_feat = self.expert_semantic(tokens)
        fus_feat = self.expert_fusion(tokens)

        moe_tokens = (
            gates[..., 0:1] * vis_feat +
            gates[..., 1:2] * sem_feat +
            gates[..., 2:3] * fus_feat
        )                                                # [B, N, 1024]

        # ========== EEG 时序注意力池化 ==========
        shared_features = self.temporal_pool(moe_tokens)

        # ========== CLIP Projection ==========
        img_emb = self.img_head(shared_features)
        txt_emb = self.txt_head(shared_features)

        img_emb = F.normalize(img_emb, dim=-1)
        txt_emb = F.normalize(txt_emb, dim=-1)

        return img_emb, txt_emb, {
            "logit_scale": self.logit_scale.exp().item(),
            "gate_vis": gates[..., 0].mean().item(),
            "gate_sem": gates[..., 1].mean().item(),
            "gate_fus": gates[..., 2].mean().item()
        }
