# main.py

#忽略兼容警告
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

# 导入您本地的代码
from models.clip_models_2o import EEGEncoder
from utils.loss_methods import InfoNCE
from dataset_2o import TripletDataset

# 设置 PyTorch 以获得更好的性能
torch.backends.cudnn.benchmark = True


def train_one_epoch(model, dataloader, optimizer, loss_fn_img, loss_fn_txt, device, alpha):
    """
    执行一个周期的训练
    """
    model.train()
    total_loss = 0.0
    total_loss_img = 0.0
    total_loss_txt = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        eeg_signals, image_vecs, text_vecs = batch

        # 将数据移动到GPU
        eeg_signals = eeg_signals.to(device)
        image_vecs = image_vecs.to(device)
        text_vecs = text_vecs.to(device)

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        eeg_img_embeddings, eeg_text_embeddings = model(eeg_signals)

        # 计算损失
        loss_img = loss_fn_img(eeg_img_embeddings, image_vecs)
        loss_txt = loss_fn_txt(eeg_text_embeddings, text_vecs)

        # 加权联合损失
        loss = (alpha * loss_img) + ((1 - alpha) * loss_txt)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_loss_img += loss_img.item()
        total_loss_txt += loss_txt.item()

    avg_loss = total_loss / len(dataloader)
    avg_loss_img = total_loss_img / len(dataloader)
    avg_loss_txt = total_loss_txt / len(dataloader)

    return avg_loss, avg_loss_img, avg_loss_txt


def validate(model, dataloader, loss_fn_img, loss_fn_txt, device, alpha):
    """
    在验证集上评估模型
    """
    model.eval()
    total_loss_val = 0.0
    total_loss_val_img = 0.0
    total_loss_val_txt = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            eeg_signals, image_vecs, text_vecs = batch

            eeg_signals = eeg_signals.to(device)
            image_vecs = image_vecs.to(device)
            text_vecs = text_vecs.to(device)

            eeg_img_embedding, eeg_txt_embedding = model(eeg_signals)

            loss_img = loss_fn_img(eeg_img_embedding, image_vecs)
            loss_txt = loss_fn_txt(eeg_txt_embedding, text_vecs)

            loss = (alpha * loss_img) + ((1 - alpha) * loss_txt)

            total_loss_val += loss.item()
            total_loss_val_img += loss_img.item()
            total_loss_val_txt += loss_txt.item()

    avg_loss_val = total_loss_val / len(dataloader)
    avg_loss_val_img = total_loss_val_img / len(dataloader)
    avg_loss_val_txt = total_loss_val_txt / len(dataloader)

    return avg_loss_val, avg_loss_val_img, avg_loss_val_txt


@hydra.main(version_base=None, config_path="configs", config_name="triplet_config")
def main(cfg: DictConfig):
    print("Hydra 配置:\n", OmegaConf.to_yaml(cfg))

    # 初始化 WandB
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name,
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    device = torch.device(cfg.training.device)

    # 1. 初始化模型
    # 我们只实例化 EEGEncoder，因为图像/文本向量已经提前计算好了
    model = EEGEncoder(
        n_channels=cfg.model.n_channels,
        n_samples=cfg.model.n_samples,
        n_filters_time=cfg.model.n_filters_time,
        filter_time_length=cfg.model.filter_time_length,
        n_filters_spat=cfg.model.n_filters_spat,
        pool_time_length=cfg.model.pool_time_length,
        pool_time_stride=cfg.model.pool_time_stride,
        n_linear_layers=cfg.model.n_linear_layers,
        embedding_dim=cfg.model.embedding_dim,
        drop_prob=cfg.model.drop_prob,
        encoder_name=cfg.model.encoder_name,
        channel_merge=cfg.model.channel_merge,
        n_heads=cfg.model.n_heads
    ).to(device)

    # 2. 准备数据
    split_index = cfg.data.get("split_index", 0)

    train_dataset = TripletDataset(cfg.data, mode='train', split_index=split_index)
    val_dataset = TripletDataset(cfg.data, mode='val', split_index=split_index)
    test_dataset = TripletDataset(cfg.data, mode='test', split_index=split_index)

    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True,
                              num_workers=cfg.training.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False,
                            num_workers=cfg.training.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False,
                             num_workers=cfg.training.num_workers, pin_memory=True)

    # 3. 初始化损失函数和优化器
    # 我们需要两个独立的损失函数实例
    loss_fn_img = InfoNCE(temperature=cfg.training.temperature).to(device)
    loss_fn_txt = InfoNCE(temperature=cfg.training.temperature).to(device)

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.get("weight_decay", 0.0) # <-- 使用 .get() 安全读取
    )

    # 4. 训练循环
    print("开始训练...")
    best_val_loss = float('inf')
    
    #早停计时器
    # (如果 patience 未在 config 中定义, 默认使用一个很大的数)
    patience = cfg.training.get("patience", cfg.training.epochs) 
    min_delta = cfg.training.get("min_delta", 0.0)
    epochs_no_improve = 0

    for epoch in range(cfg.training.epochs):
        avg_loss, avg_loss_img, avg_loss_txt = train_one_epoch(
            model, train_loader, optimizer, loss_fn_img, loss_fn_txt, device, cfg.training.alpha
        )

        avg_loss_val, avg_loss_val_img, avg_loss_val_txt = validate(
            model, val_loader, loss_fn_img, loss_fn_txt, device, cfg.training.alpha
        )

        print(f"Epoch {epoch + 1}/{cfg.training.epochs}")
        print(f"  Train Loss: {avg_loss:.4f} | Val Loss: {avg_loss_val:.4f}")
        print(f"  Train Img Loss: {avg_loss_img:.4f} | Val Img Loss: {avg_loss_val_img:.4f}")
        print(f"  Train Txt Loss: {avg_loss_txt:.4f} | Val Txt Loss: {avg_loss_val_txt:.4f}")

        # 记录到 WandB
        wandb.log({
            "epoch": epoch,
            "train_loss_total": avg_loss,
            "train_loss_image": avg_loss_img,
            "train_loss_text": avg_loss_txt,
            "val_loss_total": avg_loss_val,
            "val_loss_image": avg_loss_val_img,
            "val_loss_text": avg_loss_val_txt
        })

        # 保存最佳模型 + 早停计时器data11.8_1
        # 只有当 (best_val_loss - avg_loss_val) > min_delta 时，才算作一次有效的“改善”
        if (best_val_loss - avg_loss_val) > min_delta:
            best_val_loss = avg_loss_val
            model_path = os.path.join(wandb.run.dir, "best_eeg_encoder.pth")
            torch.save(model.state_dict(), model_path)
            print(f"模型已保存到: {model_path}")
            epochs_no_improve = 0 # 重置计数器
        else:
            epochs_no_improve += 1 # 增加计数器

        if epochs_no_improve >= patience:
            print(f"验证损失连续 {patience} 个 epoch 没有改善，触发 Early Stopping。")
            break # 退出训练循环

    print("训练完成。")
    # 训练完成后在测试集上评估
    print("【测试集评估】")
    avg_loss_test, avg_loss_test_img, avg_loss_test_txt = validate(
        model, test_loader, loss_fn_img, loss_fn_txt, device, cfg.training.alpha
    )

    print(f"Test Total Loss: {avg_loss_test:.4f}")
    print(f"Test Image Loss: {avg_loss_test_img:.4f}")
    print(f"Test Text  Loss: {avg_loss_test_txt:.4f}")

    wandb.log({
        "test_loss_total": avg_loss_test,
        "test_loss_image": avg_loss_test_img,
        "test_loss_text": avg_loss_test_txt
    })
    wandb.finish()


if __name__ == "__main__":
    main()