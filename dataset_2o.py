# dataset_2o.py

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import warnings 

class TripletDataset(Dataset):
    """
    用于加载 (EEG, 图像向量, 文本向量) 三元组的数据集。
    """

    def __init__(self, cfg_data, mode='train', split_index=0):
        print(f"正在加载 {mode} 数据（split_index={split_index}）...")

        # 1. 加载所有数据到内存
        try:
            eeg_loaded_data = torch.load(cfg_data.eeg_path)
            self.all_eeg_items = eeg_loaded_data['dataset']
            print(f"成功从 {cfg_data.eeg_path} 加载了 'dataset' 列表，包含 {len(self.all_eeg_items)} 个条目。")
        except KeyError:
            print(f"错误：在 {cfg_data.eeg_path} 中找不到 'dataset' 键。请检查文件结构。")
            raise
        except Exception as e:
            print(f"加载 EEG 数据时出错: {e}")
            raise

        self.all_image_vectors = torch.from_numpy(np.load(cfg_data.image_vec_path)).float()
        self.all_text_vectors = torch.from_numpy(np.load(cfg_data.text_vec_path)).float()
        num_available_img_vectors = len(self.all_image_vectors)
        num_available_txt_vectors = len(self.all_text_vectors)

        # 记录实际可用的向量数量
        self.num_available_vectors = min(num_available_img_vectors, num_available_txt_vectors)

        # 2. 加载数据划分索引
        splits_data = torch.load(cfg_data.splits_path)
        try:
            raw_indices = splits_data['splits'][split_index][mode]
            print(f"使用 split {split_index}，{mode} 模式原始索引数量：{len(raw_indices)}")
        except (KeyError, IndexError) as e:
            print(f"加载 split {split_index} 的 {mode} 划分失败：{e}")
            raise

        # 3. 过滤索引
        self.indices = [] 
        skipped_count = 0
        for eeg_idx in raw_indices:
            try:
                if eeg_idx >= len(self.all_eeg_items):
                    skipped_count += 1
                    continue

                image_idx = self.all_eeg_items[eeg_idx]['image']

                if image_idx < self.num_available_vectors:
                    self.indices.append(eeg_idx)
                else:
                    skipped_count += 1

            except KeyError:
                skipped_count += 1
            except Exception as e:
                skipped_count += 1

        if skipped_count > 0:
            warnings.warn(f"在 {mode} 模式下，跳过了 {skipped_count} 个无效条目。")

        print(f"过滤后，{mode} 模式实际使用 {len(self.indices)} 个 EEG 条目。")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 1. 获取原始 EEG 数据
        eeg_original_index = self.indices[idx]
        eeg_item_dict = self.all_eeg_items[eeg_original_index]
        eeg_signal = eeg_item_dict['eeg'].float()

        # 2. 长度处理：裁剪或填充到 440
        target_length = 440
        current_length = eeg_signal.shape[-1]
        
        if current_length > target_length:
            eeg_signal = eeg_signal[..., :target_length]
        elif current_length < target_length:
            # 填充 0
            padding_shape = list(eeg_signal.shape)
            padding_shape[-1] = target_length - current_length
            padding = torch.zeros(padding_shape, dtype=eeg_signal.dtype, device=eeg_signal.device)
            eeg_signal = torch.cat((eeg_signal, padding), dim=-1)

        # =======================================================
        # 【核心修正】 强制 Z-Score 归一化 (Per-sample Normalization)
        # =======================================================
        # 目的：将信号强度(Std)从 ~0.4 拉升到 1.0，匹配 DreamDiffusion 预训练分布
        # dim=-1 表示在时间维度上计算均值和标准差，对每个通道独立/或整体归一化均可
        # 这里我们对每个通道的时间序列做标准化
        mean = eeg_signal.mean(dim=-1, keepdim=True)
        std = eeg_signal.std(dim=-1, keepdim=True)
        # 加上 1e-6 是为了防止纯平信号(std=0)导致除以 0
        eeg_signal = (eeg_signal - mean) / (std + 1e-6)
        # =======================================================

        # 3. 获取对应的图像/文本向量
        main_image_index = eeg_item_dict['image']
        image_vector = self.all_image_vectors[main_image_index]
        text_vector = self.all_text_vectors[main_image_index]

        return eeg_signal, image_vector, text_vector