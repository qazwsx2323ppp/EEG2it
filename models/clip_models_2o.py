# models/clip_models.py

import torch
from torch import nn
#from braindecode.models import to_dense_prediction_model
from models.ddpt_model import MAEforEEG
import torch.nn.functional as F


class SpatialMoEEncoder(nn.Module):
    def __init__(
        self,
        n_channels,
        n_samples,
        # base_encoder_cls,  <-- 不再需要传入类，直接内部实例化 ViT
        # base_encoder_params, <-- 不再需要
        visual_indices,
        semantic_indices,
        embedding_dim=512,
        pretrained_path=None  # <-- 新增：预训练权重路径
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.embedding_dim = embedding_dim
        
        # 注册索引 (用于生成 mask)
        self.register_buffer('idx_vis', torch.tensor(visual_indices, dtype=torch.long))
        self.register_buffer('idx_sem', torch.tensor(semantic_indices, dtype=torch.long))

        # --- 1. 定义共享的主干 (DreamDiffusion ViT) ---
        # 这是一个 128 通道的“全能专家”，我们用它来提取特征
        self.backbone = MAEforEEG(
            time_len=512,      
            in_chans=128,       # 必须是 128，为了匹配预训练权重
            patch_size=4,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            decoder_embed_dim=1024, 
            mlp_ratio=1.0,            # 新增此行！原来默认是 4.0，必须改为 1.0 以匹配 fc 层权重
            decoder_depth=8,
            decoder_num_heads=16
        )

        # # 加载预训练权重
        # if pretrained_path:
        #     print(f"Loading DreamDiffusion checkpoint from {pretrained_path}")
        #     # --- 【新增】 开始：伪造缺失的 config 模块 ---
        #     import sys
        #     import types

        #     # 1. 定义一个空的伪造类
        #     class DummyConfig:
        #         pass

        #     # 2. 创建一个伪造的模块，名字叫 'config'
        #     dummy_config_module = types.ModuleType("config")
            
        #     # 3. 将伪造类挂载到伪造模块下 (类名必须完全匹配报错信息：Config_Generative_Model)
        #     dummy_config_module.Config_Generative_Model = DummyConfig
            
        #     # 4. 将伪造模块注入系统，骗过 torch.load
        #     sys.modules["config"] = dummy_config_module
        #     # --- 【新增】 结束 ---

        #     # 现在可以安全加载了 (weights_only=False 是必须的)
        #     checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            
        #     # 处理 checkpoint 字典结构
        #     state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        #     # 显式允许加载 pickle 对象（因为我们要加载的 checkpoint 包含旧版 config 类）
        #     checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        #     state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        #     # 移除 module. 前缀
        #     new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}


        # 加载预训练权重
        if pretrained_path:
            print(f"Loading DreamDiffusion checkpoint from {pretrained_path}")
            # --- 伪造 config 模块逻辑保持不变 ---
            import sys
            import types
            class DummyConfig: pass
            dummy_config_module = types.ModuleType("config")
            dummy_config_module.Config_Generative_Model = DummyConfig
            sys.modules["config"] = dummy_config_module
            # ----------------------------------

            # ... (前文的伪造 config 代码保持不变) ...

        # 1. 加载文件
        print(f"Loading DreamDiffusion checkpoint from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        
        # 2. 基础拆包 (先拿到大字典)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # 3. 【核心修正】智能筛选与重命名
        # 目标：从大模型中提取 'cond_stage_model.mae.' 开头的权重
        new_state_dict = {}
        target_prefix = "cond_stage_model.mae."
        
        print(f"正在从完整生成模型中提取 EEG Encoder 权重 (前缀: {target_prefix})...")
        
        for k, v in state_dict.items():
            # 情况 A: 权重带有完整前缀 (最可能的情况)
            if k.startswith(target_prefix):
                new_key = k.replace(target_prefix, "")
                new_state_dict[new_key] = v
            # 情况 B: 万一权重已经是 mae. 开头 (以防万一)
            elif k.startswith("mae."):
                new_key = k.replace("mae.", "")
                new_state_dict[new_key] = v
                
        # 4. 如果提取到了参数，进行加载
        if len(new_state_dict) > 0:
            print(f">>> 成功提取了 {len(new_state_dict)} 个 EEG Encoder 参数。")
            msg = self.backbone.load_state_dict(new_state_dict, strict=False)
            print(f">>> 权重加载详情: {msg}")
            
            ## --- 【修正后的检查逻辑】 ---
            # 过滤掉所有 decoder 相关的缺失键，因为我们不需要 Decoder
            missing_keys = [k for k in msg.missing_keys if not k.startswith('decoder_') and not k.startswith('mask_token')]
            
            # 打印结果
            if len(missing_keys) > 0:
                print(f">>> ⚠️ 注意：部分 Encoder 权重未加载 (Missing Keys): {missing_keys}")
                # 只有当核心 Encoder 层缺失时才报严重警告
                if any("blocks" in k for k in missing_keys) or any("patch_embed" in k for k in missing_keys):
                    print("!!! 严重警告：核心 Encoder Block 缺失！请检查前缀！")
                else:
                    print(">>> (这些缺失可能不影响 Encoder 功能，如 head 等)")
            else:
                print(">>> ✅ 完美！所有 Encoder 核心权重均已加载！")

            # 确认 Decoder 确实被忽略了
            decoder_missing = [k for k in msg.missing_keys if k.startswith('decoder_')]
            if len(decoder_missing) > 0:
                print(f">>> 已忽略 {len(decoder_missing)} 个 Decoder 参数 (这是正常的)。")

            
            # (可选) 冻结主干，只训练后面的 Router 和 Heads，节省显存
            # for param in self.backbone.parameters():
            #     param.requires_grad = False

        # --- 2. 定义轻量级专家头 (Adapter Experts) ---
        # 我们不再用 3 个大模型，而是用 3 个轻量级映射层作为“专家”
        # 它们接收 Backbone 的输出 (1024维)，映射到目标空间 (512维)
        
        self.backbone_dim = 1024
        
        # 这里的“专家”变成了专门负责转换特征的 Adapter
        self.expert_visual_head = nn.Sequential(
            nn.Linear(self.backbone_dim, self.backbone_dim),
            nn.GELU(),
            nn.Linear(self.backbone_dim, embedding_dim)
        )
        
        self.expert_semantic_head = nn.Sequential(
            nn.Linear(self.backbone_dim, self.backbone_dim),
            nn.GELU(),
            nn.Linear(self.backbone_dim, embedding_dim)
        )
        
        self.expert_fusion_head = nn.Sequential(
            nn.Linear(self.backbone_dim, self.backbone_dim),
            nn.GELU(),
            nn.Linear(self.backbone_dim, embedding_dim)
        )

        # --- 3. Router (保持不变) ---
        self.router_pool = nn.AdaptiveAvgPool1d(1)
        self.router_net = nn.Sequential(
            nn.Linear(n_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 4), 
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch, channels, samples) 
        # DreamDiffusion 需要 (batch, channels, 512)
        
        # 1. 预处理：Padding 到 512
        target_len = 512
        if x.shape[-1] < target_len:
            padding = torch.zeros(x.shape[0], x.shape[1], target_len - x.shape[-1], device=x.device)
            x_padded = torch.cat([x, padding], dim=-1)
        else:
            x_padded = x[..., :target_len]

        # 2. 通过共享 Backbone 提取全局特征
        # latent shape: [batch, num_patches, 1024]
        latent, _, _ = self.backbone.forward_encoder(x_padded, mask_ratio=0.0)
        
        # 聚合特征 (Global Average Pooling) -> [batch, 1024]
        shared_features = latent.mean(dim=1)

        # 3. 计算 Router 权重 (使用原始 x 计算，保持物理意义)
        # feat_for_router: [batch, channels, 1] -> [batch, channels]
        feat_for_router = self.router_pool(x).squeeze(-1) 
        gates = self.router_net(feat_for_router) # [batch, 4]
        
        g_vis_img = gates[:, 0:1]
        g_fus_img = gates[:, 1:2]
        g_sem_txt = gates[:, 2:3]
        g_fus_txt = gates[:, 3:4]

        # 归一化权重
        g_vis_img = g_vis_img / (g_vis_img + g_fus_img + 1e-6)
        g_fus_img = g_fus_img / (g_vis_img + g_fus_img + 1e-6)
        g_sem_txt = g_sem_txt / (g_sem_txt + g_fus_txt + 1e-6)
        g_fus_txt = g_fus_txt / (g_sem_txt + g_fus_txt + 1e-6)

        # 4. 专家前向传播 (现在是轻量级 Adapter)
        # 这里的“思想”是：虽然特征是共享的，但不同的 Head 负责提取不同的信息
        emb_vis = self.expert_visual_head(shared_features)
        emb_sem = self.expert_semantic_head(shared_features)
        emb_fus = self.expert_fusion_head(shared_features)

        # 5. 加权融合
        final_img_embedding = (g_vis_img * emb_vis) + (g_fus_img * emb_fus)
        final_text_embedding = (g_sem_txt * emb_sem) + (g_fus_txt * emb_fus)
    
        # =============== 【新增】 强制 L2 归一化 ===============
        # 这一步至关重要！将向量投射到单位球面上，这是对比学习的标准动作。
        final_img_embedding = F.normalize(final_img_embedding, p=2, dim=-1)
        final_text_embedding = F.normalize(final_text_embedding, p=2, dim=-1)
        # =======================================================
    
        return final_img_embedding, final_text_embedding, {"w_vis_img": g_vis_img.mean(), "w_sem_txt": g_sem_txt.mean()}

