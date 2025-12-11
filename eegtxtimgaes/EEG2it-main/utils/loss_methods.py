# utils/loss_methods.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCE(nn.Module):
    """
    一个标准的、可重用的 InfoNCE (对比损失) 类。
    它封装了您在 loss_calculation 中找到的逻辑，并修复了 device 报错。
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, features1, features2):
        """
        计算对比损失。
        features1: 第一个模态的特征 (例如, EEG 向量), 形状 [batch_size, embedding_dim]
        features2: 第二个模态的特征 (例如, 图像/文本向量), 形状 [batch_size, embedding_dim]
        """
        # 从输入的张量中动态获取 device，这是正确的做法
        device = features1.device

        # 计算相似度矩阵
        # 形状: [batch_size, batch_size]
        logits = (features1 @ features2.T) / self.temperature

        # 创建目标标签（一个单位矩阵）
        # 形状: [batch_size]
        targets = torch.arange(logits.shape[0], device=device, dtype=torch.long)

        # 计算双向损失
        # F.cross_entropy 内部会自动处理 one-hot 转换
        loss1 = self.cross_entropy(logits, targets)
        loss2 = self.cross_entropy(logits.T, targets)

        # 返回平均损失
        loss = (loss1 + loss2) / 2.0
        return loss


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343
    (这是原始文件中的另一个损失函数，我们保留它以备将来实验)
    """

    def __init__(
            self,
            cache_labels=False,
            bidir=True,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.bidir = bidir

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def forward(self, image_features, text_features, logit_scale, logit_bias=None, ):
        device = image_features.device
        dtype = image_features.dtype
        num_logits = image_features.shape[0]

        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)

        labels = self.get_ground_truth(device, dtype, num_logits)

        loss = -F.logsigmoid(labels * logits).sum() / num_logits

        if self.bidir:
            logits_t = logits.T
            labels_t = self.get_ground_truth(device, dtype, logits_t.shape[0])
            loss_t = -F.logsigmoid(labels_t * logits_t).sum() / logits_t.shape[0]
            loss = (loss + loss_t) / 2

        return loss