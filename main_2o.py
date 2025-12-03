# main.py

# #忽略兼容警告
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

# import os
# import hydra
# from omegaconf import DictConfig, OmegaConf
# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import wandb
# from tqdm import tqdm

# # 导入您本地的代码
# from models.clip_models_2o import SpatialMoEEncoder
# from utils.loss_methods import InfoNCE
# from dataset_2o import TripletDataset

# # 设置 PyTorch 以获得更好的性能
# torch.backends.cudnn.benchmark = True


# def train_one_epoch(model, dataloader, optimizer, loss_fn_img, loss_fn_txt, device, alpha):
#     """
#     执行一个周期的训练
#     """
#     model.train()
#     total_loss = 0.0
#     total_loss_img = 0.0
#     total_loss_txt = 0.0

#     # --- 【新增】 用于累计权重值 ---
#     total_weights = {
#         "w_vis_img": 0.0, "w_fus_img": 0.0, 
#         "w_sem_txt": 0.0, "w_fus_txt": 0.0
#     }

#     for batch in tqdm(dataloader, desc="Training"):
#         eeg_signals, image_vecs, text_vecs = batch

#         # 将数据移动到GPU
#         eeg_signals = eeg_signals.to(device)
#         image_vecs = image_vecs.to(device)
#         text_vecs = text_vecs.to(device)

#         # 梯度清零
#         optimizer.zero_grad()

#         # 前向传播
#         # --- 【修改】 接收三个返回值 ---
#         # 注意：这里需要兼容旧模型。如果 model 返回 2 个值，说明是旧模型；3 个值是新模型。
#         outputs = model(eeg_signals)
        
#         if len(outputs) == 3:
#             eeg_img_embeddings, eeg_text_embeddings, weights_info = outputs
#             # 累加权重以便后续求平均
#             for k, v in weights_info.items():
#                 total_weights[k] += v.item()
#         else:
#             eeg_img_embeddings, eeg_text_embeddings = outputs
#             weights_info = None

#         # 计算损失
#         loss_img = loss_fn_img(eeg_img_embeddings, image_vecs)
#         loss_txt = loss_fn_txt(eeg_text_embeddings, text_vecs)

#         # 加权联合损失
#         loss = (alpha * loss_img) + ((1 - alpha) * loss_txt)

#         # 反向传播
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         total_loss_img += loss_img.item()
#         total_loss_txt += loss_txt.item()

#     avg_loss = total_loss / len(dataloader)
#     avg_loss_img = total_loss_img / len(dataloader)
#     avg_loss_txt = total_loss_txt / len(dataloader)

#     # --- 【新增】 返回平均权重字典 ---
#     avg_weights = {}
#     if total_weights["w_vis_img"] > 0: # 确保有数据
#         for k in total_weights:
#             avg_weights[k] = total_weights[k] / len(dataloader)
            
#     return avg_loss, avg_loss_img, avg_loss_txt, avg_weights


# def validate(model, dataloader, loss_fn_img, loss_fn_txt, device, alpha):
#     """
#     在验证集上评估模型
#     """
#     model.eval()
#     total_loss_val = 0.0
#     total_loss_val_img = 0.0
#     total_loss_val_txt = 0.0

#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="Validation"):
#             eeg_signals, image_vecs, text_vecs = batch

#             eeg_signals = eeg_signals.to(device)
#             image_vecs = image_vecs.to(device)
#             text_vecs = text_vecs.to(device)

#             eeg_img_embedding, eeg_txt_embedding = model(eeg_signals)

#             loss_img = loss_fn_img(eeg_img_embedding, image_vecs)
#             loss_txt = loss_fn_txt(eeg_txt_embedding, text_vecs)

#             loss = (alpha * loss_img) + ((1 - alpha) * loss_txt)

#             total_loss_val += loss.item()
#             total_loss_val_img += loss_img.item()
#             total_loss_val_txt += loss_txt.item()

#     avg_loss_val = total_loss_val / len(dataloader)
#     avg_loss_val_img = total_loss_val_img / len(dataloader)
#     avg_loss_val_txt = total_loss_val_txt / len(dataloader)

#     return avg_loss_val, avg_loss_val_img, avg_loss_val_txt


# @hydra.main(version_base=None, config_path="configs", config_name="triplet_config")
# def main(cfg: DictConfig):
#     print("Hydra 配置:\n", OmegaConf.to_yaml(cfg))

#     # 初始化 WandB
#     wandb.init(
#         project=cfg.wandb.project,
#         entity=cfg.wandb.entity,
#         name=cfg.wandb.name,
#         config=OmegaConf.to_container(cfg, resolve=True)
#     )

#     device = torch.device(cfg.training.device)

#     # --- 修改 2: 初始化模型逻辑 ---
#     print(f"正在初始化模型: {cfg.model.get('model_name', 'default')}")



#     # 3. 实例化 SpatialMoEEncoder
#     model = SpatialMoEEncoder(
#         n_channels=cfg.model.n_channels,
#         n_samples=cfg.model.n_samples,
#         visual_indices=cfg.model.moe_config.visual_indices,
#         semantic_indices=cfg.model.moe_config.semantic_indices,
#         embedding_dim=cfg.model.embedding_dim
#     ).to(device)
        
#     print(">>> 成功初始化 Spatial MoE Encoder (分区专家模型)")

   
#     # 2. 准备数据
#     split_index = cfg.data.get("split_index", 0)

#     train_dataset = TripletDataset(cfg.data, mode='train', split_index=split_index)
#     val_dataset = TripletDataset(cfg.data, mode='val', split_index=split_index)
#     test_dataset = TripletDataset(cfg.data, mode='test', split_index=split_index)

#     train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True,
#                               num_workers=cfg.training.num_workers, pin_memory=True)
#     val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False,
#                             num_workers=cfg.training.num_workers, pin_memory=True)
#     test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False,
#                              num_workers=cfg.training.num_workers, pin_memory=True)

#     # 3. 初始化损失函数和优化器
#     # 我们需要两个独立的损失函数实例
#     loss_fn_img = InfoNCE(temperature=cfg.training.temperature).to(device)
#     loss_fn_txt = InfoNCE(temperature=cfg.training.temperature).to(device)

#     optimizer = optim.AdamW(
#         model.parameters(), 
#         lr=cfg.training.learning_rate,
#         weight_decay=cfg.training.get("weight_decay", 0.0) # <-- 使用 .get() 安全读取
#     )

#     # 4. 训练循环
#     print("开始训练...")
#     best_val_loss = float('inf')
    
#     #早停计时器
#     # (如果 patience 未在 config 中定义, 默认使用一个很大的数)
#     patience = cfg.training.get("patience", cfg.training.epochs) 
#     min_delta = cfg.training.get("min_delta", 0.0)
#     epochs_no_improve = 0

#     for epoch in range(cfg.training.epochs):
#         avg_loss, avg_loss_img, avg_loss_txt, avg_weights = train_one_epoch(
#             model, train_loader, optimizer, loss_fn_img, loss_fn_txt, device, cfg.training.alpha
#         )

#         avg_loss_val, avg_loss_val_img, avg_loss_val_txt = validate(
#             model, val_loader, loss_fn_img, loss_fn_txt, device, cfg.training.alpha
#         )

#         print(f"Epoch {epoch + 1}/{cfg.training.epochs}")
#         print(f"  Train Loss: {avg_loss:.4f} | Val Loss: {avg_loss_val:.4f}")
#         print(f"  Train Img Loss: {avg_loss_img:.4f} | Val Img Loss: {avg_loss_val_img:.4f}")
#         print(f"  Train Txt Loss: {avg_loss_txt:.4f} | Val Txt Loss: {avg_loss_val_txt:.4f}")

#         # 记录到 WandB
#         log_dict = {
#             "epoch": epoch,
#             "train_loss_total": avg_loss,
#             "train_loss_image": avg_loss_img,
#             "train_loss_text": avg_loss_txt,
#             "val_loss_total": avg_loss_val,
#             "val_loss_image": avg_loss_val_img,
#             "val_loss_text": avg_loss_val_txt
#         }
#         # 把权重也加进去
#         if avg_weights:
#             log_dict.update(avg_weights)

#         # 保存最佳模型 + 早停计时器data11.8_1

#         # 只有当 (best_val_loss - avg_loss_val) > min_delta 时，才算作一次有效的“改善”
#         if (best_val_loss - avg_loss_val) > min_delta:
#             best_val_loss = avg_loss_val
#             model_path = os.path.join(wandb.run.dir, "best_eeg_encoder.pth")
#             torch.save(model.state_dict(), model_path)
#             print(f"模型已保存到: {model_path}")
#             epochs_no_improve = 0 # 重置计数器
#         else:
#             epochs_no_improve += 1 # 增加计数器

#         if epochs_no_improve >= patience:
#             print(f"验证损失连续 {patience} 个 epoch 没有改善，触发 Early Stopping。")
#             break # 退出训练循环

#     print("训练完成。")
#     # 训练完成后在测试集上评估
#     print("【测试集评估】")
#     avg_loss_test, avg_loss_test_img, avg_loss_test_txt = validate(
#         model, test_loader, loss_fn_img, loss_fn_txt, device, cfg.training.alpha
#     )

#     print(f"Test Total Loss: {avg_loss_test:.4f}")
#     print(f"Test Image Loss: {avg_loss_test_img:.4f}")
#     print(f"Test Text  Loss: {avg_loss_test_txt:.4f}")

#     # 记录到 WandB
#     wandb.log(log_dict)
#     wandb.finish()


# if __name__ == "__main__":
#     main()

# main_2o.py

# 忽略兼容警告
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

# --- 【新增】 引入混合精度和调度器 ---
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup

# 导入您本地的代码
# 确保 models/clip_models_2o.py 中定义的是最新的 SpatialMoEEncoder
from models.clip_models_2o import SpatialMoEEncoder
from utils.loss_methods import InfoNCE
from dataset_2o import TripletDataset

# 设置 PyTorch 以获得更好的性能
torch.backends.cudnn.benchmark = True


def freeze_backbone(model, freeze=True):
    """
    冻结或解冻 Backbone 的辅助函数
    """
    # 假设你的 SpatialMoEEncoder 中有一个 self.backbone (即 MAEforEEG)
    if hasattr(model, 'backbone'):
        for param in model.backbone.parameters():
            param.requires_grad = not freeze
        state = "冻结" if freeze else "解冻"
        print(f">>> Backbone 已{state}。")
    else:
        print("警告: 模型中未找到 'backbone' 属性，无法执行冻结/解冻操作。")


def train_one_epoch(model, dataloader, optimizer, loss_fn_img, loss_fn_txt, device, alpha, scaler, scheduler):
    """
    执行一个周期的训练 (加入了 AMP 和 Scheduler)
    """
    model.train()
    total_loss = 0.0
    total_loss_img = 0.0
    total_loss_txt = 0.0

    # 用于累计权重值
    total_weights = {
        "w_vis_img": 0.0, "w_fus_img": 0.0, 
        "w_sem_txt": 0.0, "w_fus_txt": 0.0
    }

    for batch in tqdm(dataloader, desc="Training"):
        eeg_signals, image_vecs, text_vecs = batch

        # 将数据移动到GPU
        eeg_signals = eeg_signals.to(device)
        image_vecs = image_vecs.to(device)
        text_vecs = text_vecs.to(device)

        # 梯度清零
        optimizer.zero_grad()

        # --- 【修改】 混合精度前向传播 ---
        with autocast():
            # 前向传播
            outputs = model(eeg_signals)
            
            if len(outputs) == 3:
                eeg_img_embeddings, eeg_text_embeddings, weights_info = outputs
                # 累加权重
                if weights_info:
                    for k, v in weights_info.items():
                        total_weights[k] += v.item()
            else:
                # 兼容旧接口，防止报错
                eeg_img_embeddings, eeg_text_embeddings = outputs
                weights_info = None

            # 计算损失
            loss_img = loss_fn_img(eeg_img_embeddings, image_vecs)
            loss_txt = loss_fn_txt(eeg_text_embeddings, text_vecs)

            # 加权联合损失
            loss = (alpha * loss_img) + ((1 - alpha) * loss_txt)

        # --- 【修改】 混合精度反向传播 ---
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_loss_img += loss_img.item()
        total_loss_txt += loss_txt.item()

    avg_loss = total_loss / len(dataloader)
    avg_loss_img = total_loss_img / len(dataloader)
    avg_loss_txt = total_loss_txt / len(dataloader)

    # 返回平均权重字典
    avg_weights = {}
    if total_weights["w_vis_img"] > 0: 
        for k in total_weights:
            avg_weights[k] = total_weights[k] / len(dataloader)
            
    return avg_loss, avg_loss_img, avg_loss_txt, avg_weights


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

            # 验证集通常不需要 autocast，除非显存非常紧缺
            outputs = model(eeg_signals)
            
            # 处理返回值解包
            if len(outputs) == 3:
                eeg_img_embedding, eeg_txt_embedding, _ = outputs
            else:
                eeg_img_embedding, eeg_txt_embedding = outputs

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

    print(f"正在初始化模型: SpatialMoEEncoder (Backbone: DreamDiffusion)")

    # 1. 实例化 SpatialMoEEncoder
    # --- 【关键修正】 传入 pretrained_path ---
    model = SpatialMoEEncoder(
        n_channels=cfg.model.n_channels,
        n_samples=cfg.model.n_samples,
        visual_indices=cfg.model.moe_config.visual_indices,
        semantic_indices=cfg.model.moe_config.semantic_indices,
        embedding_dim=cfg.model.embedding_dim,
        pretrained_path=cfg.model.get("pretrained_path", None) # 确保从 Config 读取路径
    ).to(device)
        
    print(">>> 成功初始化 Spatial MoE Encoder")

    # 2. 准备数据
    split_index = cfg.data.get("split_index", 0)

    train_dataset = TripletDataset(cfg.data, mode='train', split_index=split_index)
    val_dataset = TripletDataset(cfg.data, mode='val', split_index=split_index)
    test_dataset = TripletDataset(cfg.data, mode='test', split_index=split_index)

    # --- 建议增大 Batch Size (得益于 AMP) ---
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True,
                              num_workers=cfg.training.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False,
                            num_workers=cfg.training.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False,
                             num_workers=cfg.training.num_workers, pin_memory=True)

    # 3. 初始化损失函数、优化器、调度器、Scaler
    loss_fn_img = InfoNCE(temperature=cfg.training.temperature).to(device)
    loss_fn_txt = InfoNCE(temperature=cfg.training.temperature).to(device)

    # # 使用 AdamW，通常 ViT 微调需要较小的 LR，但这里通过调度器控制
    # optimizer = optim.AdamW(
    #     model.parameters(), 
    #     lr=cfg.training.learning_rate,
    #     weight_decay=cfg.training.get("weight_decay", 0.05)
    # )

    # 1. 将参数分组
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            # 包括 router, expert_heads 等
            head_params.append(param)

    # # 2. 定义优化器，给 Backbone 一个更小的学习率 (通常是主学习率的 1/10 或 1/100)
    # optimizer = optim.AdamW(
    #     [
    #         # Backbone 使用非常小的学习率，小心翼翼地微调
    #         {"params": backbone_params, "lr": cfg.training.learning_rate * 0.01}, 
    #         # Heads 使用正常的学习率
    #         {"params": head_params, "lr": cfg.training.learning_rate}, 
    #     ],
    #     weight_decay=cfg.training.get("weight_decay", 0.05)
    # )

       # 核心修改：保护你的 DreamDiffusion Backbone
    optimizer = optim.AdamW(
        [
            # 给 Backbone 一个极小的学习率 (如 1e-6 或 5e-6)
            {"params": backbone_params, "lr": cfg.training.learning_rate * 0.05}, 
            # 给 Router 和 Heads 正常的学习率 (如 1e-4)
            {"params": head_params, "lr": cfg.training.learning_rate}, 
        ],
           weight_decay=cfg.training.get("weight_decay", 0.05)
    )

    # --- 【新增】 学习率调度器 ---
    # 总步数 = epoch * steps_per_epoch
    num_training_steps = cfg.training.epochs * len(train_loader)
    # 预热步数设为总步数的 10%
    num_warmup_steps = int(0.1 * num_training_steps)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # --- 【新增】 混合精度 Scaler ---
    scaler = GradScaler()

    # 4. 训练循环
    print("开始训练...")
    best_val_loss = float('inf')
    
    patience = cfg.training.get("patience", cfg.training.epochs) 
    min_delta = cfg.training.get("min_delta", 0.0)
    epochs_no_improve = 0
    
    # 设定解冻 Backbone 的 Epoch (例如第 10 个 Epoch)
    unfreeze_epoch = 10 
    
    # --- 【新增】 初始冻结 Backbone ---
    freeze_backbone(model, freeze=True)

    for epoch in range(cfg.training.epochs):
        
        # --- 【新增】 在指定 Epoch 解冻 ---
        if epoch == unfreeze_epoch:
            print(f">>> 达到第 {epoch} 轮，开始解冻 Backbone 进行全局微调...")
            freeze_backbone(model, freeze=False)
            # 可选：解冻后可以重置学习率或调整
            
        avg_loss, avg_loss_img, avg_loss_txt, avg_weights = train_one_epoch(
            model, train_loader, optimizer, loss_fn_img, loss_fn_txt, device, 
            cfg.training.alpha, scaler, scheduler
        )

        avg_loss_val, avg_loss_val_img, avg_loss_val_txt = validate(
            model, val_loader, loss_fn_img, loss_fn_txt, device, cfg.training.alpha
        )

        # 获取当前学习率用于记录
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch + 1}/{cfg.training.epochs} | LR: {current_lr:.6f}")
        print(f"  Train Loss: {avg_loss:.4f} | Val Loss: {avg_loss_val:.4f}")
        print(f"  Train Img Loss: {avg_loss_img:.4f} | Val Img Loss: {avg_loss_val_img:.4f}")

        # 记录到 WandB
        log_dict = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss_total": avg_loss,
            "train_loss_image": avg_loss_img,
            "train_loss_text": avg_loss_txt,
            "val_loss_total": avg_loss_val,
            "val_loss_image": avg_loss_val_img,
            "val_loss_text": avg_loss_val_txt
        }
        if avg_weights:
            log_dict.update(avg_weights)
        
        wandb.log(log_dict)

        # 保存最佳模型 & 早停逻辑
        if (best_val_loss - avg_loss_val) > min_delta:
            best_val_loss = avg_loss_val
            model_path = os.path.join(wandb.run.dir, "best_eeg_encoder.pth")
            torch.save(model.state_dict(), model_path)
            print(f"模型已保存到: {model_path}")
            epochs_no_improve = 0 
        else:
            epochs_no_improve += 1 

        if epochs_no_improve >= patience:
            print(f"验证损失连续 {patience} 个 epoch 没有改善，触发 Early Stopping。")
            break 

    print("训练完成。")
    # 测试集评估
    print("【测试集评估】")
    avg_loss_test, avg_loss_test_img, avg_loss_test_txt = validate(
        model, test_loader, loss_fn_img, loss_fn_txt, device, cfg.training.alpha
    )

    print(f"Test Total Loss: {avg_loss_test:.4f}")
    
    wandb.log({
        "test_loss_total": avg_loss_test,
        "test_loss_image": avg_loss_test_img,
        "test_loss_text": avg_loss_test_txt
    })
    wandb.finish()


if __name__ == "__main__":
    main()