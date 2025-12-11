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


# ----------------------------------------------------
# 3. 修复后的 EEGEncoder 类 (现在可以正确接收参数)
# ----------------------------------------------------
class EEGEncoder(nn.Module):
    def __init__(
            self,
            n_channels,  # <-- 1. 我们在这里添加了 n_channels
            n_samples,  # <-- 2. 我们在这里添加了 n_samples
            encoder_name,
            n_filters_time,
            filter_time_length,
            n_filters_spat,
            pool_time_length,
            pool_time_stride,
            n_linear_layers,
            embedding_dim,
            drop_prob,
            channel_merge=None,
            n_heads=None,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.encoder_name = encoder_name
        #self.channel_merge = channel_merge

        if self.encoder_name == "braindecode_shallow":
            self.encoder = BraindecodeShallow(
                n_channels=self.n_channels,  # <-- 3. 我们将 n_channels 传递下去
                n_samples=self.n_samples,  # <-- 4. 我们将 n_samples 传递下去
                n_filters_time=n_filters_time,
                filter_time_length=filter_time_length,
                n_filters_spat=n_filters_spat,
                pool_time_length=pool_time_length,
                pool_time_stride=pool_time_stride,
                n_linear_layers=n_linear_layers,
                embedding_dim=embedding_dim,
                drop_prob=drop_prob,
            )
        elif self.encoder_name == "braindecode_deep":
            self.encoder = BraindecodeDeep(
                n_channels=self.n_channels,  # <-- 5. 同样传递给 BraindecodeDeep
                n_samples=self.n_samples,  # <-- 6. 同样传递给 BraindecodeDeep
                n_filters_time=n_filters_time,
                filter_time_length=filter_time_length,
                n_filters_spat=n_filters_spat,
                pool_time_length=pool_time_length,
                pool_time_stride=pool_time_stride,
                n_linear_layers=n_linear_layers,
                embedding_dim=embedding_dim,
                drop_prob=drop_prob,
            )

        # if self.channel_merge == "attention":
        #     self.attention_pool = nn.TransformerEncoderLayer(
        #         d_model=self.encoder.embedding_dim,
        #         nhead=n_heads,
        #         dim_feedforward=self.encoder.embedding_dim * 4,
        #         dropout=drop_prob,
        #         activation="gelu",
        #     )
        #     self.merger = nn.Linear(
        #         self.encoder.embedding_dim * self.n_channels, embedding_dim
        #     )
        # elif self.channel_merge == "linear":
        #     self.merger = nn.Linear(
        #         self.encoder.embedding_dim * self.n_channels, embedding_dim
        #     )

    def forward(self, x):
        x = self.encoder(x)
        # if self.channel_merge == "attention":
        #     x = x.permute(2, 0, 1)  # (time, batch, channels)
        #     x = self.attention_pool(x)
        #     x = x.permute(1, 0, 2)  # (batch, time, channels)
        #     x = x.reshape(x.shape[0], -1)  # (batch, time*channels)
        #     x = self.merger(x)
        # elif self.channel_merge == "linear":
        #     x = x.reshape(x.shape[0], -1)  # (batch, channels*embedding_dim)
        #     x = self.merger(x)
        return x