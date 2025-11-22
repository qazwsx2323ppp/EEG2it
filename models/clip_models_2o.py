# models/clip_models.py

import torch
from torch import nn
#from braindecode.models import to_dense_prediction_model


# ----------------------------------------------------
# 1. BraindecodeShallow 类的完整定义
# ----------------------------------------------------
class BraindecodeShallow(nn.Module):
    def __init__(
            self,
            n_channels,
            n_samples,
            n_filters_time=40,
            filter_time_length=25,
            n_filters_spat=40,
            pool_time_length=75,
            pool_time_stride=15,
            n_linear_layers=1,
            embedding_dim=128,
            drop_prob=0.5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_filters_time = n_filters_time
        self.filter_time_length = filter_time_length
        self.n_filters_spat = n_filters_spat
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.n_linear_layers = n_linear_layers
        self.embedding_dim = embedding_dim
        self.drop_prob = drop_prob

        self.temporal_conv = nn.Conv2d(
            1, n_filters_time, (filter_time_length, 1), padding="same"
        )
        self.spat_conv = nn.Conv2d(
            n_filters_time, n_filters_spat, (1, n_channels), bias=False
        )
        self.batch_norm = nn.BatchNorm2d(n_filters_spat)
        self.avg_pool = nn.AvgPool2d(
            kernel_size=(pool_time_length, 1), stride=(pool_time_stride, 1)
        )
        self.dropout = nn.Dropout(drop_prob)

        # --- 修改开始 ---
        # 1. 'out' 重命名为 'backbone'，并且不包含最后的 'lin_embedding'
        self.backbone = nn.Sequential()
        self.out_dim = self.calculate_out_dim()  # 卷积和池化后的特征维度

        if n_linear_layers > 1:
            self.backbone.add_module("lin_intermediate", nn.Linear(self.out_dim, self.out_dim))
            self.backbone.add_module("lin_activation", nn.ELU())
            self.backbone.add_module("lin_dropout", nn.Dropout(self.drop_prob))
            self.final_input_dim = self.out_dim  # 中间层的输出维度
        else:
            self.final_input_dim = self.out_dim  # 如果没有中间层，则直接使用 'out_dim'

        # 2. 创建两个独立的输出头
        self.head_img = nn.Linear(self.final_input_dim, embedding_dim)
        self.head_txt = nn.Linear(self.final_input_dim, embedding_dim)
        # --- 修改结束 ---

    def calculate_out_dim(self):
        # 模拟一次前向传播以获取输出维度
        dummy_input = torch.randn(1, 1, self.n_samples, self.n_channels)
        x = self.temporal_conv(dummy_input)
        x = self.spat_conv(x)
        x = self.batch_norm(x)
        x = torch.square(x)
        x = self.avg_pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.dropout(x)
        return int(x.reshape(x.shape[0], -1).shape[1])

    def forward(self, x):
        # 确保输入形状为 (batch, 1, samples, channels)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 2)  # (batch, 1, channels, samples) -> (batch, 1, samples, channels)

        x = self.temporal_conv(x)
        x = self.spat_conv(x)
        x = self.batch_norm(x)
        x = torch.square(x)
        x = self.avg_pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.dropout(x)
        x = x.reshape(x.shape[0], -1)  # 展平

        # --- 修改开始 ---
        # 1. 通过共享的 'backbone' (可能包含中间层)
        shared_features = self.backbone(x)

        # 2. 将共享特征分别送入两个独立的头
        out_img = self.head_img(shared_features)
        out_txt = self.head_txt(shared_features)

        # 3. 返回两个向量
        return out_img, out_txt
        # --- 修改结束 ---


# ----------------------------------------------------
# 2. BraindecodeDeep 类的修改 (逻辑同上)
# ----------------------------------------------------
class BraindecodeDeep(nn.Module):
    def __init__(
            self,
            n_channels,
            n_samples,
            n_filters_time=25,
            filter_time_length=10,
            n_filters_spat=25,
            pool_time_length=3,
            pool_time_stride=3,
            n_linear_layers=1,
            embedding_dim=128,
            drop_prob=0.5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_filters_time = n_filters_time
        self.filter_time_length = filter_time_length
        self.n_filters_spat = n_filters_spat
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.n_linear_layers = n_linear_layers
        self.embedding_dim = embedding_dim
        self.drop_prob = drop_prob

        # ... (前面的卷积块 conv1, conv2, block2, block3, block4 保持不变) ...
        # 第一个卷积块
        self.conv1 = nn.Conv2d(1, n_filters_time, (filter_time_length, 1), stride=1)
        self.conv2 = nn.Conv2d(n_filters_time, n_filters_spat, (1, n_channels), bias=False)
        self.batch_norm1 = nn.BatchNorm2d(n_filters_spat)
        self.act1 = nn.ELU()
        self.pool1 = nn.MaxPool2d(
            kernel_size=(pool_time_length, 1), stride=(pool_time_stride, 1)
        )
        self.dropout1 = nn.Dropout(drop_prob)

        # 辅助函数来创建后续的卷积块
        def _create_conv_block(in_filters, out_filters, kernel, pool_kernel, pool_stride):
            return nn.Sequential(
                nn.Conv2d(in_filters, out_filters, (kernel, 1), stride=1, bias=False),
                nn.BatchNorm2d(out_filters),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(pool_kernel, 1), stride=(pool_stride, 1)),
                nn.Dropout(drop_prob),
            )

        # 后续的卷积块
        self.block2 = _create_conv_block(n_filters_spat, 50, 10, 3, 3)
        self.block3 = _create_conv_block(50, 100, 10, 3, 3)
        self.block4 = _create_conv_block(100, 200, 10, 3, 3)

        # --- 修改开始 ---
        # 1. 'out' 重命名为 'backbone'
        self.backbone = nn.Sequential()
        self.out_dim = self.calculate_out_dim()

        if n_linear_layers > 1:
            self.backbone.add_module("lin_intermediate", nn.Linear(self.out_dim, self.out_dim))
            self.backbone.add_module("lin_activation", nn.ELU())
            self.backbone.add_module("lin_dropout", nn.Dropout(self.drop_prob))
            self.final_input_dim = self.out_dim
        else:
            self.final_input_dim = self.out_dim

        # 2. 创建两个独立的输出头
        self.head_img = nn.Linear(self.final_input_dim, embedding_dim)
        self.head_txt = nn.Linear(self.final_input_dim, embedding_dim)
        # --- 修改结束 ---

    def calculate_out_dim(self):
        # 模拟一次前向传播以获取输出维度
        dummy_input = torch.randn(1, 1, self.n_samples, self.n_channels)
        x = self.conv1(dummy_input)
        x = self.conv2(x)
        x = self.batch_norm1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return int(x.reshape(x.shape[0], -1).shape[1])

    def forward(self, x):
        # 确保输入形状为 (batch, 1, samples, channels)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 2)  # (batch, 1, channels, samples) -> (batch, 1, samples, channels)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batch_norm1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.reshape(x.shape[0], -1)  # 展平

        # --- 修改开始 ---
        # 1. 通过共享的 'backbone'
        shared_features = self.backbone(x)

        # 2. 将共享特征分别送入两个独立的头
        out_img = self.head_img(shared_features)
        out_txt = self.head_txt(shared_features)

        # 3. 返回两个向量
        return out_img, out_txt
        # --- 修改结束 ---


# # ----------------------------------------------------
# # 3. 修复后的 EEGEncoder 类 (现在可以正确接收参数)
# # ----------------------------------------------------
# class EEGEncoder(nn.Module):
#     def __init__(
#             self,
#             n_channels,  # <-- 1. 我们在这里添加了 n_channels
#             n_samples,  # <-- 2. 我们在这里添加了 n_samples
#             encoder_name,
#             n_filters_time,
#             filter_time_length,
#             n_filters_spat,
#             pool_time_length,
#             pool_time_stride,
#             n_linear_layers,
#             embedding_dim,
#             drop_prob,
#             channel_merge=None,
#             n_heads=None,
#     ):
#         super().__init__()
#         self.n_channels = n_channels
#         self.n_samples = n_samples
#         self.encoder_name = encoder_name
#         #self.channel_merge = channel_merge

#         if self.encoder_name == "braindecode_shallow":
#             self.encoder = BraindecodeShallow(
#                 n_channels=self.n_channels,  # <-- 3. 我们将 n_channels 传递下去
#                 n_samples=self.n_samples,  # <-- 4. 我们将 n_samples 传递下去
#                 n_filters_time=n_filters_time,
#                 filter_time_length=filter_time_length,
#                 n_filters_spat=n_filters_spat,
#                 pool_time_length=pool_time_length,
#                 pool_time_stride=pool_time_stride,
#                 n_linear_layers=n_linear_layers,
#                 embedding_dim=embedding_dim,
#                 drop_prob=drop_prob,
#             )
#         elif self.encoder_name == "braindecode_deep":
#             self.encoder = BraindecodeDeep(
#                 n_channels=self.n_channels,  # <-- 5. 同样传递给 BraindecodeDeep
#                 n_samples=self.n_samples,  # <-- 6. 同样传递给 BraindecodeDeep
#                 n_filters_time=n_filters_time,
#                 filter_time_length=filter_time_length,
#                 n_filters_spat=n_filters_spat,
#                 pool_time_length=pool_time_length,
#                 pool_time_stride=pool_time_stride,
#                 n_linear_layers=n_linear_layers,
#                 embedding_dim=embedding_dim,
#                 drop_prob=drop_prob,
#             )

#         # if self.channel_merge == "attention":
#         #     self.attention_pool = nn.TransformerEncoderLayer(
#         #         d_model=self.encoder.embedding_dim,
#         #         nhead=n_heads,
#         #         dim_feedforward=self.encoder.embedding_dim * 4,
#         #         dropout=drop_prob,
#         #         activation="gelu",
#         #     )
#         #     self.merger = nn.Linear(
#         #         self.encoder.embedding_dim * self.n_channels, embedding_dim
#         #     )
#         # elif self.channel_merge == "linear":
#         #     self.merger = nn.Linear(
#         #         self.encoder.embedding_dim * self.n_channels, embedding_dim
#         #     )

#     def forward(self, x):
#         x = self.encoder(x)
#         # if self.channel_merge == "attention":
#         #     x = x.permute(2, 0, 1)  # (time, batch, channels)
#         #     x = self.attention_pool(x)
#         #     x = x.permute(1, 0, 2)  # (batch, time, channels)
#         #     x = x.reshape(x.shape[0], -1)  # (batch, time*channels)
#         #     x = self.merger(x)
#         # elif self.channel_merge == "linear":
#         #     x = x.reshape(x.shape[0], -1)  # (batch, channels*embedding_dim)
#         #     x = self.merger(x)
#         return x

class SpatialMoEEncoder(nn.Module):
    def __init__(
        self,
        n_channels,
        n_samples,
        base_encoder_cls, # 传入 BraindecodeShallow 类
        base_encoder_params, # 字典，包含 n_filters_time 等参数
        visual_indices,
        semantic_indices,
        embedding_dim=512
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.visual_indices = torch.tensor(visual_indices, dtype=torch.long)
        self.semantic_indices = torch.tensor(semantic_indices, dtype=torch.long)
        
        # 注册为 buffer，这样它们会自动跟随模型移动到 GPU，但不是可训练参数
        self.register_buffer('idx_vis', self.visual_indices)
        self.register_buffer('idx_sem', self.semantic_indices)

        # --- 1. 定义三个专家 ---
        
        # Visual Expert (只看视觉通道)
        # 注意：我们需要更新 n_channels 参数
        vis_params = base_encoder_params.copy()
        vis_params['n_channels'] = len(visual_indices)
        vis_params['embedding_dim'] = embedding_dim
        self.expert_visual = base_encoder_cls(**vis_params)

        # Semantic Expert (只看语义通道)
        sem_params = base_encoder_params.copy()
        sem_params['n_channels'] = len(semantic_indices)
        sem_params['embedding_dim'] = embedding_dim
        self.expert_semantic = base_encoder_cls(**sem_params)

        # Fusion Expert (看所有通道，处理跨脑区协同)
        fus_params = base_encoder_params.copy()
        fus_params['n_channels'] = n_channels
        fus_params['embedding_dim'] = embedding_dim
        self.expert_fusion = base_encoder_cls(**fus_params)

        # --- 2. 定义 Router (门控网络) ---
        # 接收全通道信号，输出 2 个权重：
        # w_vis: 控制 Visual Expert 在图像对齐中的权重
        # w_sem: 控制 Semantic Expert 在文本对齐中的权重
        # (Fusion Expert 的权重可以是 1-w 或者独立学习，这里我们采用独立学习 Softmax)
        
        # 简单的 Router: 先在时间维度平均池化，然后通过 MLP
        self.router_pool = nn.AdaptiveAvgPool1d(1) 
        self.router_net = nn.Sequential(
            nn.Linear(n_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 4), # 输出4个值: [w_vis_img, w_fus_img, w_sem_txt, w_fus_txt]
            nn.Sigmoid() # 归一化到 0-1 (或者使用 Softmax 分组归一化)
        )

    def forward(self, x):
        """
        x shape: (batch, channels, samples) 或 (batch, 1, samples, channels)
        注意：BraindecodeShallow 内部通常期望 (batch, 1, samples, channels) 并做 permute。
        但在 dataset.py 中我们看到加载的是 (channels, samples)。
        DataLoader 会将其堆叠为 (batch, channels, samples)。
        """
        
        # 1. 统一输入形状为 (batch, 1, samples, channels) 以方便切片
        # 假设输入 x 是 (batch, n_channels, n_samples)
        if x.dim() == 3:
             x = x.permute(0, 2, 1).unsqueeze(1) # -> (batch, 1, samples, channels)
        
        # 2. 物理切分 (Hard-coded Masks)
        # 使用 index_select 在最后一个维度(channels)进行切片
        x_visual = torch.index_select(x, 3, self.idx_vis)
        x_semantic = torch.index_select(x, 3, self.idx_sem)
        x_fusion = x # 全通道
        
        # 3. 专家前向传播
        # 注意：BraindecodeShallow 的 forward 内部会处理维度，我们直接传入切分后的数据
        # 假设 base_encoder 的 forward 返回的是 embedding (batch, dim)
        
        emb_vis = self.expert_visual(x_visual)
        emb_sem = self.expert_semantic(x_semantic)
        emb_fus = self.expert_fusion(x_fusion)
        
        # 4. Router 权重计算
        # 我们需要一个特征向量来计算路由。使用全通道信号的平均值。
        # x_fusion: (batch, 1, samples, channels)
        # permute back to (batch, channels, samples) for pooling
        feat_for_router = x_fusion.squeeze(1).permute(0, 2, 1) 
        feat_pooled = self.router_pool(feat_for_router).squeeze(-1) # (batch, channels)
        
        gates = self.router_net(feat_pooled) # (batch, 4)
        
        # 分割门控权重
        g_vis_img = gates[:, 0:1] # Visual Expert 对 Image 的贡献
        g_fus_img = gates[:, 1:2] # Fusion Expert 对 Image 的贡献
        
        g_sem_txt = gates[:, 2:3] # Semantic Expert 对 Text 的贡献
        g_fus_txt = gates[:, 3:4] # Fusion Expert 对 Text 的贡献
        
        # --- 归一化权重 (可选，但推荐，类似于 Softmax) ---
        # 让 g_vis_img + g_fus_img = 1
        w_sum_img = g_vis_img + g_fus_img + 1e-6
        g_vis_img = g_vis_img / w_sum_img
        g_fus_img = g_fus_img / w_sum_img
        
        w_sum_txt = g_sem_txt + g_fus_txt + 1e-6
        g_sem_txt = g_sem_txt / w_sum_txt
        g_fus_txt = g_fus_txt / w_sum_txt

        # 5. 特征融合 (Soft Routing)
        # 生成最终用于对比学习的两个向量
        
        # 图像对齐向量：由 Visual Expert 和 Fusion Expert 决定
        final_img_embedding = (g_vis_img * emb_vis) + (g_fus_img * emb_fus)
        
        # 文本对齐向量：由 Semantic Expert 和 Fusion Expert 决定
        final_text_embedding = (g_sem_txt * emb_sem) + (g_fus_txt * emb_fus)

        return final_img_embedding, final_text_embedding