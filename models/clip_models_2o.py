# models/clip_models.py

import torch
from torch import nn
#from braindecode.models import to_dense_prediction_model
from models.ddpt_model import MAEforEEG


# # ----------------------------------------------------
# # 1. BraindecodeShallow 类的完整定义
# # ----------------------------------------------------
# class BraindecodeShallow(nn.Module):
#     def __init__(
#             self,
#             n_channels,
#             n_samples,
#             n_filters_time=40,
#             filter_time_length=25,
#             n_filters_spat=40,
#             pool_time_length=75,
#             pool_time_stride=15,
#             n_linear_layers=1,
#             embedding_dim=128,
#             drop_prob=0.5,
#     ):
#         super().__init__()
#         self.n_channels = n_channels
#         self.n_samples = n_samples
#         self.n_filters_time = n_filters_time
#         self.filter_time_length = filter_time_length
#         self.n_filters_spat = n_filters_spat
#         self.pool_time_length = pool_time_length
#         self.pool_time_stride = pool_time_stride
#         self.n_linear_layers = n_linear_layers
#         self.embedding_dim = embedding_dim
#         self.drop_prob = drop_prob

#         self.temporal_conv = nn.Conv2d(
#             1, n_filters_time, (filter_time_length, 1), padding="same"
#         )
#         self.spat_conv = nn.Conv2d(
#             n_filters_time, n_filters_spat, (1, n_channels), bias=False
#         )
#         self.batch_norm = nn.BatchNorm2d(n_filters_spat)
#         self.avg_pool = nn.AvgPool2d(
#             kernel_size=(pool_time_length, 1), stride=(pool_time_stride, 1)
#         )
#         self.dropout = nn.Dropout(drop_prob)

#         # --- 修改开始 ---
#         # 1. 'out' 重命名为 'backbone'，并且不包含最后的 'lin_embedding'
#         self.backbone = nn.Sequential()
#         self.out_dim = self.calculate_out_dim()  # 卷积和池化后的特征维度

#         if n_linear_layers > 1:
#             self.backbone.add_module("lin_intermediate", nn.Linear(self.out_dim, self.out_dim))
#             self.backbone.add_module("lin_activation", nn.ELU())
#             self.backbone.add_module("lin_dropout", nn.Dropout(self.drop_prob))
#             self.final_input_dim = self.out_dim  # 中间层的输出维度
#         else:
#             self.final_input_dim = self.out_dim  # 如果没有中间层，则直接使用 'out_dim'

#         # 2. 创建两个独立的输出头
#         self.head_img = nn.Linear(self.final_input_dim, embedding_dim)
#         self.head_txt = nn.Linear(self.final_input_dim, embedding_dim)
#         # --- 修改结束 ---

#     def calculate_out_dim(self):
#         # 模拟一次前向传播以获取输出维度
#         dummy_input = torch.randn(1, 1, self.n_samples, self.n_channels)
#         x = self.temporal_conv(dummy_input)
#         x = self.spat_conv(x)
#         x = self.batch_norm(x)
#         x = torch.square(x)
#         x = self.avg_pool(x)
#         x = torch.log(torch.clamp(x, min=1e-6))
#         x = self.dropout(x)
#         return int(x.reshape(x.shape[0], -1).shape[1])

#     def forward(self, x):
#         # 确保输入形状为 (batch, 1, samples, channels)
#         if len(x.shape) == 3:
#             x = x.unsqueeze(1)
#         x = x.permute(0, 1, 3, 2)  # (batch, 1, channels, samples) -> (batch, 1, samples, channels)

#         x = self.temporal_conv(x)
#         x = self.spat_conv(x)
#         x = self.batch_norm(x)
#         x = torch.square(x)
#         print(f"DEBUG: Shape before pooling: {x.shape}")
#         print(f"DEBUG: Pooling layer config: {self.avg_pool}")
#         x = self.avg_pool(x)
#         x = torch.log(torch.clamp(x, min=1e-6))
#         x = self.dropout(x)
#         x = x.reshape(x.shape[0], -1)  # 展平

#         # --- 修改开始 ---
#         # 1. 通过共享的 'backbone' (可能包含中间层)
#         shared_features = self.backbone(x)

#         # 2. 将共享特征分别送入两个独立的头
#         out_img = self.head_img(shared_features)
#         out_txt = self.head_txt(shared_features)

#         # 3. 返回两个向量
#         return out_img, out_txt
#         # --- 修改结束 ---


# # ----------------------------------------------------
# # 2. BraindecodeDeep 类的修改 (逻辑同上)
# # ----------------------------------------------------
# class BraindecodeDeep(nn.Module):
#     def __init__(
#             self,
#             n_channels,
#             n_samples,
#             n_filters_time=25,
#             filter_time_length=10,
#             n_filters_spat=25,
#             pool_time_length=3,
#             pool_time_stride=3,
#             n_linear_layers=1,
#             embedding_dim=128,
#             drop_prob=0.5,
#     ):
#         super().__init__()
#         self.n_channels = n_channels
#         self.n_samples = n_samples
#         self.n_filters_time = n_filters_time
#         self.filter_time_length = filter_time_length
#         self.n_filters_spat = n_filters_spat
#         self.pool_time_length = pool_time_length
#         self.pool_time_stride = pool_time_stride
#         self.n_linear_layers = n_linear_layers
#         self.embedding_dim = embedding_dim
#         self.drop_prob = drop_prob

#         # ... (前面的卷积块 conv1, conv2, block2, block3, block4 保持不变) ...
#         # 第一个卷积块
#         self.conv1 = nn.Conv2d(1, n_filters_time, (filter_time_length, 1), stride=1)
#         self.conv2 = nn.Conv2d(n_filters_time, n_filters_spat, (1, n_channels), bias=False)
#         self.batch_norm1 = nn.BatchNorm2d(n_filters_spat)
#         self.act1 = nn.ELU()
#         self.pool1 = nn.MaxPool2d(
#             kernel_size=(pool_time_length, 1), stride=(pool_time_stride, 1)
#         )
#         self.dropout1 = nn.Dropout(drop_prob)

#         # 辅助函数来创建后续的卷积块
#         def _create_conv_block(in_filters, out_filters, kernel, pool_kernel, pool_stride):
#             return nn.Sequential(
#                 nn.Conv2d(in_filters, out_filters, (kernel, 1), stride=1, bias=False),
#                 nn.BatchNorm2d(out_filters),
#                 nn.ELU(),
#                 nn.MaxPool2d(kernel_size=(pool_kernel, 1), stride=(pool_stride, 1)),
#                 nn.Dropout(drop_prob),
#             )

#         # 后续的卷积块
#         self.block2 = _create_conv_block(n_filters_spat, 50, 10, 3, 3)
#         self.block3 = _create_conv_block(50, 100, 10, 3, 3)
#         self.block4 = _create_conv_block(100, 200, 10, 3, 3)

#         # --- 修改开始 ---
#         # 1. 'out' 重命名为 'backbone'
#         self.backbone = nn.Sequential()
#         self.out_dim = self.calculate_out_dim()

#         if n_linear_layers > 1:
#             self.backbone.add_module("lin_intermediate", nn.Linear(self.out_dim, self.out_dim))
#             self.backbone.add_module("lin_activation", nn.ELU())
#             self.backbone.add_module("lin_dropout", nn.Dropout(self.drop_prob))
#             self.final_input_dim = self.out_dim
#         else:
#             self.final_input_dim = self.out_dim

#         # 2. 创建两个独立的输出头
#         self.head_img = nn.Linear(self.final_input_dim, embedding_dim)
#         self.head_txt = nn.Linear(self.final_input_dim, embedding_dim)
#         # --- 修改结束 ---

#     def calculate_out_dim(self):
#         # 模拟一次前向传播以获取输出维度
#         dummy_input = torch.randn(1, 1, self.n_samples, self.n_channels)
#         x = self.conv1(dummy_input)
#         x = self.conv2(x)
#         x = self.batch_norm1(x)
#         x = self.act1(x)
#         x = self.pool1(x)
#         x = self.dropout1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.block4(x)
#         return int(x.reshape(x.shape[0], -1).shape[1])

#     def forward(self, x):
#         # 确保输入形状为 (batch, 1, samples, channels)
#         if len(x.shape) == 3:
#             x = x.unsqueeze(1)
#         x = x.permute(0, 1, 3, 2)  # (batch, 1, channels, samples) -> (batch, 1, samples, channels)

#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.batch_norm1(x)
#         x = self.act1(x)
#         x = self.pool1(x)
#         x = self.dropout1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.block4(x)
#         x = x.reshape(x.shape[0], -1)  # 展平

#         # --- 修改开始 ---
#         # 1. 通过共享的 'backbone'
#         shared_features = self.backbone(x)

#         # 2. 将共享特征分别送入两个独立的头
#         out_img = self.head_img(shared_features)
#         out_txt = self.head_txt(shared_features)

#         # 3. 返回两个向量
#         return out_img, out_txt
#         # --- 修改结束 ---


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
            decoder_embed_dim=512, 
            decoder_depth=8,
            decoder_num_heads=16
        )

        # 加载预训练权重
        if pretrained_path:
            print(f"Loading DreamDiffusion checkpoint from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            # 移除 module. 前缀
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.backbone.load_state_dict(new_state_dict, strict=False)
            
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

        # 返回两个向量，完美适配 main_2o.py
        return final_img_embedding, final_text_embedding
