# dataset_2o.py

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import warnings
# 【新增】 必须导入 scipy 的插值函数，否则会报 NameError
from scipy.interpolate import interp1d 

class TripletDataset(Dataset):
    """
    用于加载 (EEG, 图像向量, 文本向量) 三元组的数据集。
    严格复刻 DreamDiffusion 的预处理逻辑 (Slice + Interpolate)。
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
        
        # 2. 读取数据并转为 numpy (为了使用 scipy 插值)
        # 假设原始数据形状是 (Channels, Time) 例如 (128, 500)
        eeg_signal = eeg_item_dict['eeg'].float().numpy() 
        
        # --- 【核心修正 1】: 严格复刻官方的转置与切片逻辑 (20ms~460ms) ---
        # 官方逻辑先 .t() (转置为 Time, Channel)，切片，再转回来
        eeg_signal = eeg_signal.T  # (Time, Channel)
        eeg_signal = eeg_signal[20:460, :] # 丢弃前20ms伪迹，保留中间440ms
        eeg_signal = eeg_signal.T  # (Channel, Time) -> (128, 440)

        # --- 【核心修正 2】: 使用插值 (Interpolation) 拉伸到 512 ---
        target_length = 512
        current_length = eeg_signal.shape[-1]
        
        if current_length != target_length:
            x = np.linspace(0, 1, current_length)
            x2 = np.linspace(0, 1, target_length)
            # 对最后一个维度 (Time) 进行插值
            f = interp1d(x, eeg_signal, axis=-1) 
            eeg_signal = f(x2)
        
        # 转回 PyTorch Tensor
        eeg_signal = torch.from_numpy(eeg_signal).float()

        # --- 【保留】: Z-Score 归一化 ---
        # DreamDiffusion 官方虽然依赖预训练分布，但显式归一化通常能加速收敛
        #先注释掉，当前归一化
        # mean = eeg_signal.mean(dim=-1, keepdim=True)
        # std = eeg_signal.std(dim=-1, keepdim=True)
        # eeg_signal = (eeg_signal - mean) / (std + 1e-6)

        # 3. 获取图像/文本向量
        main_image_index = eeg_item_dict['image']
        image_vector = self.all_image_vectors[main_image_index]
        text_vector = self.all_text_vectors[main_image_index]

        return eeg_signal, image_vector, text_vector