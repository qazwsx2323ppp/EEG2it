# dataset.py (或者 my_dataset.py)

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import warnings  # 用于显示过滤警告


class TripletDataset(Dataset):
    """
    用于加载 (EEG, 图像向量, 文本向量) 三元组的数据集。

    在初始化时过滤掉指向不存在的图像/文本向量的索引。
    """

    def __init__(self, cfg_data, mode='train',split_index=0):
        print(f"正在加载 {mode} 数据（split_index={split_index}）...")

        # 1. 加载所有数据到内存
        try:
            eeg_loaded_data = torch.load(cfg_data.eeg_path)
            # self.all_eeg_items 是包含约12000个字典的列表
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

        # 记录实际可用的向量数量（取两者中较小者，以防万一）
        self.num_available_vectors = min(num_available_img_vectors, num_available_txt_vectors)

        print(f"加载了 {num_available_img_vectors} 个图像向量。")
        print(f"加载了 {num_available_txt_vectors} 个文本向量。")
        print(f"实际可用向量索引范围: 0 至 {self.num_available_vectors - 1}")

        # 2. 加载数据划分索引
        splits_data = torch.load(cfg_data.splits_path)
        try:
            raw_indices = splits_data['splits'][split_index][mode]
            print(f"使用 split {split_index}，{mode} 模式原始索引数量：{len(raw_indices)}")
        except (KeyError, IndexError) as e:
            print(f"加载 split {split_index} 的 {mode} 划分失败：{e}")
            raise

        # 3. 过滤索引 <--- 核心修改在这里
        self.indices = []  # 存储过滤后的、有效的 EEG 列表索引
        skipped_count = 0
        for eeg_idx in raw_indices:
            try:
                # 检查 EEG 索引本身是否有效
                if eeg_idx >= len(self.all_eeg_items):
                    # print(f"警告：原始 EEG 索引 {eeg_idx} 超出 EEG 列表边界 ({len(self.all_eeg_items)})，跳过。")
                    skipped_count += 1
                    continue

                # 获取对应的图像索引
                image_idx = self.all_eeg_items[eeg_idx]['image']

                # 检查图像索引是否在可用向量范围内
                if image_idx < self.num_available_vectors:
                    self.indices.append(eeg_idx)  # 只保留有效的 EEG 列表索引
                else:
                    # print(f"警告：图像索引 {image_idx} 超出可用向量范围 (0-{self.num_available_vectors-1})，跳过 EEG 索引 {eeg_idx}。")
                    skipped_count += 1

            except KeyError:
                # print(f"警告：EEG 索引 {eeg_idx} 对应的条目缺少 'image' 键，跳过。")
                skipped_count += 1
            except Exception as e:
                # print(f"处理 EEG 索引 {eeg_idx} 时发生未知错误: {e}，跳过。")
                skipped_count += 1

        if skipped_count > 0:
            warnings.warn(f"在 {mode} 模式下，由于图像/文本向量缺失或数据错误，跳过了 {skipped_count} 个 EEG 条目。")

        print(f"过滤后，{mode} 模式实际使用 {len(self.indices)} 个 EEG 条目。")

    def __len__(self):
        # 返回过滤后的 EEG 条目数量
        return len(self.indices)

    def __getitem__(self, idx):
        # 1. 根据传入的idx，从 *过滤后* 的索引列表中获取“EEG列表索引”
        # 这里的索引现在保证是有效的
        eeg_original_index = self.indices[idx]

        # 2. 使用这个索引从原始EEG列表中获取对应的字典
        # 由于已经在 __init__ 中检查过 eeg_idx 的有效性，这里理论上不会再越界
        eeg_item_dict = self.all_eeg_items[eeg_original_index]

        # 3. 从字典中提取 EEG 信号
        eeg_signal = eeg_item_dict['eeg'].float()

        # (可选) 检查并裁剪/填充 EEG 信号长度到 440
        target_length = 440
        current_length = eeg_signal.shape[-1]  # EEG 形状可能是 (channels, samples)
        if current_length > target_length:
            eeg_signal = eeg_signal[..., :target_length]  # 使用 ... 来处理可能的通道维度
        elif current_length < target_length:
            padding_shape = list(eeg_signal.shape)
            padding_shape[-1] = target_length - current_length
            padding = torch.zeros(padding_shape, dtype=eeg_signal.dtype, device=eeg_signal.device)
            eeg_signal = torch.cat((eeg_signal, padding), dim=-1)  # 沿时间维度填充

        # 4. 从字典中提取 *正确的* 图像索引 (0-1999 or potentially higher)
        main_image_index = eeg_item_dict['image']

        # 5. 使用正确的图像索引获取对应的图像和文本向量
        # 由于已经在 __init__ 中检查过 image_idx 的有效性，这里理论上不会再越界
        image_vector = self.all_image_vectors[main_image_index]
        text_vector = self.all_text_vectors[main_image_index]

        return eeg_signal, image_vector, text_vector