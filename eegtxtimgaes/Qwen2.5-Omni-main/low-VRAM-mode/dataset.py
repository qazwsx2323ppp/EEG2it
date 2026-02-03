# dataset_2o.py

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import warnings
# 【新增】 必须导入 scipy 的插值函数，否则会报 NameError
from scipy.interpolate import interp1d 

_DS003825_INDEX_CACHE = {}

def _is_rank0() -> bool:
    try:
        return int(os.environ.get("RANK", "0")) == 0
    except Exception:
        return True

def _is_bids_root(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    return os.path.isfile(os.path.join(path, "dataset_description.json"))

def _safe_getattr(obj, name, default=None):
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _zscore_channelwise(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return (x - mean) / (std + eps)

def _ds003825_build_or_get_index(
    bids_root: str,
    include_teststim: bool,
    include_targets: bool,
    trial_stride: int,
    allowed_subjects: list[str] | None,
    exclude_subjects: set[str] | None,
    drop_exclude_flagged: bool,
):
    key = (
        os.path.abspath(bids_root),
        bool(include_teststim),
        bool(include_targets),
        int(trial_stride),
        tuple(allowed_subjects) if allowed_subjects else None,
        tuple(sorted(exclude_subjects)) if exclude_subjects else None,
        bool(drop_exclude_flagged),
    )
    if key in _DS003825_INDEX_CACHE:
        return _DS003825_INDEX_CACHE[key]

    import pandas as pd
    import mne

    bids_root = os.path.abspath(bids_root)

    participants_tsv = os.path.join(bids_root, "participants.tsv")
    flagged_exclude = set()
    if drop_exclude_flagged and os.path.isfile(participants_tsv):
        dfp = pd.read_csv(participants_tsv, sep="\t")
        if "participant_id" in dfp.columns and "exclude" in dfp.columns:
            flagged_exclude = set(dfp.loc[dfp["exclude"].astype(str) == "1", "participant_id"].astype(str).tolist())

    all_sub_dirs = sorted([d for d in os.listdir(bids_root) if d.startswith("sub-") and os.path.isdir(os.path.join(bids_root, d))])
    if allowed_subjects:
        allowed = set(allowed_subjects)
        all_sub_dirs = [s for s in all_sub_dirs if s in allowed]
    if exclude_subjects:
        all_sub_dirs = [s for s in all_sub_dirs if s not in exclude_subjects]
    if flagged_exclude:
        all_sub_dirs = [s for s in all_sub_dirs if s not in flagged_exclude]

    if not all_sub_dirs:
        raise ValueError(f"No subjects found under bids_root={bids_root}")

    subjects = all_sub_dirs
    subject_to_id = {s: i for i, s in enumerate(subjects)}

    subject_ids: list[int] = []
    onsets: list[int] = []
    concept_ids: list[int] = []

    for sub in subjects:
        eeg_dir = os.path.join(bids_root, sub, "eeg")
        events_path = os.path.join(eeg_dir, f"{sub}_task-rsvp_events.tsv")
        vhdr_path = os.path.join(eeg_dir, f"{sub}_task-rsvp_eeg.vhdr")
        if not (os.path.isfile(events_path) and os.path.isfile(vhdr_path)):
            continue

        raw = mne.io.read_raw_brainvision(vhdr_path, preload=False, verbose="ERROR")
        n_times = int(raw.n_times)
        raw.close()

        dfe = pd.read_csv(events_path, sep="\t")
        if "objectnumber" not in dfe.columns or "onset" not in dfe.columns:
            continue

        if not include_teststim and "isteststim" in dfe.columns:
            dfe = dfe[dfe["isteststim"].astype(str) == "0"]

        dfe = dfe[dfe["objectnumber"].astype(int) >= 0]

        if not include_targets and "istarget" in dfe.columns:
            dfe = dfe[dfe["istarget"].astype(str) == "0"]

        if trial_stride and trial_stride > 1:
            dfe = dfe.iloc[::trial_stride, :]

        onset_arr = dfe["onset"].astype(int).to_numpy()
        ok = (onset_arr >= 0) & ((onset_arr + 460) <= n_times)
        dfe = dfe.loc[ok]

        if len(dfe) == 0:
            continue

        sid = subject_to_id[sub]
        subject_ids.extend([sid] * len(dfe))
        onsets.extend(dfe["onset"].astype(int).tolist())
        concept_ids.extend(dfe["objectnumber"].astype(int).tolist())

    if not subject_ids:
        raise ValueError(f"No usable trials found under bids_root={bids_root}")

    index = {
        "subjects": subjects,
        "subject_ids": torch.tensor(subject_ids, dtype=torch.uint8),
        "onsets": torch.tensor(onsets, dtype=torch.int32),
        "concept_ids": torch.tensor(concept_ids, dtype=torch.int16),
    }
    _DS003825_INDEX_CACHE[key] = index
    return index

class TripletDataset(Dataset):
    """
    用于加载 (EEG, 图像向量, 文本向量) 三元组的数据集。
    严格复刻 DreamDiffusion 的预处理逻辑 (Slice + Interpolate)。
    """

    def __init__(self, cfg_data, mode='train', split_index=0, return_text=False):
        self.mode = mode
        self.return_text = return_text
        if _is_rank0():
            print(f"正在加载 {mode} 数据（split_index={split_index}）...")

        if _is_bids_root(cfg_data.eeg_path) or _safe_getattr(cfg_data, "backend", "").lower() in {"bids", "ds003825"}:
            self.backend = "ds003825"
            self._raw_cache = {}

            bids_root = os.path.abspath(cfg_data.eeg_path)
            include_teststim = bool(_safe_getattr(cfg_data, "include_teststim", False))
            include_targets = bool(_safe_getattr(cfg_data, "include_targets", False))
            trial_stride = int(_safe_getattr(cfg_data, "trial_stride", 1) or 1)
            target_channels = int(_safe_getattr(cfg_data, "target_channels", 128) or 128)
            zscore = bool(_safe_getattr(cfg_data, "zscore", False))
            zscore_eps = float(_safe_getattr(cfg_data, "zscore_eps", 1e-6))

            allowed_subjects = _safe_getattr(cfg_data, "subjects", None)
            if isinstance(allowed_subjects, str):
                allowed_subjects = [s.strip() for s in allowed_subjects.split(",") if s.strip()]

            exclude_subjects = _safe_getattr(cfg_data, "exclude_subjects", None)
            exclude_subjects_set = set()
            if isinstance(exclude_subjects, str):
                exclude_subjects_set = {s.strip() for s in exclude_subjects.split(",") if s.strip()}
            elif isinstance(exclude_subjects, (list, tuple, set)):
                exclude_subjects_set = set(exclude_subjects)

            drop_exclude_flagged = bool(_safe_getattr(cfg_data, "drop_exclude_flagged", True))

            index = _ds003825_build_or_get_index(
                bids_root=bids_root,
                include_teststim=include_teststim,
                include_targets=include_targets,
                trial_stride=trial_stride,
                allowed_subjects=allowed_subjects,
                exclude_subjects=exclude_subjects_set,
                drop_exclude_flagged=drop_exclude_flagged,
            )

            self.ds_subjects = index["subjects"]
            self.ds_subject_ids = index["subject_ids"]
            self.ds_onsets = index["onsets"]
            self.ds_concept_ids = index["concept_ids"]
            self.target_channels = target_channels
            self.bids_root = bids_root
            self.zscore = zscore
            self.zscore_eps = zscore_eps

            self.all_image_vectors = torch.from_numpy(np.load(cfg_data.image_vec_path)).float()
            self.all_text_vectors = torch.from_numpy(np.load(cfg_data.text_vec_path)).float()
            self.num_available_vectors = min(len(self.all_image_vectors), len(self.all_text_vectors))

            splits_path = _safe_getattr(cfg_data, "splits_path", None)
            if splits_path and os.path.isfile(splits_path):
                splits_data = torch.load(splits_path)
                raw_indices = splits_data["splits"][split_index][mode]
                self.indices = [int(i) for i in raw_indices]
            else:
                subjects = list(self.ds_subjects)
                rot = int(split_index) % len(subjects)
                subjects = subjects[rot:] + subjects[:rot]
                n_test = max(1, len(subjects) // 10)
                n_val = max(1, len(subjects) // 10)
                test_subs = set(subjects[:n_test])
                val_subs = set(subjects[n_test:n_test + n_val])
                train_subs = set(subjects[n_test + n_val:])

                if mode == "train":
                    keep = train_subs
                elif mode == "val":
                    keep = val_subs
                elif mode == "test":
                    keep = test_subs
                else:
                    keep = train_subs

                self.indices = []
                for i in range(int(self.ds_concept_ids.shape[0])):
                    concept = int(self.ds_concept_ids[i])
                    sub = self.ds_subjects[int(self.ds_subject_ids[i])]
                    if sub in keep and 0 <= concept < self.num_available_vectors:
                        self.indices.append(i)

            if _is_rank0():
                print(f"ds003825 backend: {mode} 使用 {len(self.indices)} 条 trial（subjects={len(self.ds_subjects)}）")
            return

        # 1. 加载所有数据到内存
        try:
            eeg_loaded_data = torch.load(cfg_data.eeg_path)
            self.all_eeg_items = eeg_loaded_data['dataset']
            if _is_rank0():
                print(f"成功从 {cfg_data.eeg_path} 加载了 'dataset' 列表，包含 {len(self.all_eeg_items)} 个条目。")
        except KeyError:
            if _is_rank0():
                print(f"错误：在 {cfg_data.eeg_path} 中找不到 'dataset' 键。请检查文件结构。")
            raise
        except Exception as e:
            if _is_rank0():
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
            if _is_rank0():
                print(f"使用 split {split_index}，{mode} 模式原始索引数量：{len(raw_indices)}")
        except (KeyError, IndexError) as e:
            if _is_rank0():
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
            if _is_rank0():
                warnings.warn(f"在 {mode} 模式下，跳过了 {skipped_count} 个无效条目。")

        if _is_rank0():
            print(f"过滤后，{mode} 模式实际使用 {len(self.indices)} 个 EEG 条目。")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if getattr(self, "backend", "") == "ds003825":
            import mne

            trial_index = int(self.indices[idx])
            sub = self.ds_subjects[int(self.ds_subject_ids[trial_index])]
            onset = int(self.ds_onsets[trial_index])
            concept = int(self.ds_concept_ids[trial_index])

            raw = self._raw_cache.get(sub)
            if raw is None:
                vhdr_path = os.path.join(self.bids_root, sub, "eeg", f"{sub}_task-rsvp_eeg.vhdr")
                raw = mne.io.read_raw_brainvision(vhdr_path, preload=False, verbose="ERROR")
                self._raw_cache[sub] = raw

            seg = raw.get_data(start=onset, stop=onset + 460, picks="eeg").astype(np.float32, copy=False)

            eeg_signal = seg.T
            eeg_signal = eeg_signal[20:460, :]
            eeg_signal = eeg_signal.T

            target_length = 512
            current_length = eeg_signal.shape[-1]
            if current_length != target_length:
                x = np.linspace(0, 1, current_length)
                x2 = np.linspace(0, 1, target_length)
                f = interp1d(x, eeg_signal, axis=-1)
                eeg_signal = f(x2)

            ch = eeg_signal.shape[0]
            if ch < self.target_channels:
                pad = np.zeros((self.target_channels - ch, eeg_signal.shape[1]), dtype=eeg_signal.dtype)
                eeg_signal = np.concatenate([eeg_signal, pad], axis=0)
            elif ch > self.target_channels:
                eeg_signal = eeg_signal[: self.target_channels, :]

            eeg_signal = torch.from_numpy(eeg_signal).float()

            if getattr(self, "zscore", False):
                eeg_signal = _zscore_channelwise(eeg_signal, eps=float(getattr(self, "zscore_eps", 1e-6)))

            if self.mode == "train":
                noise = torch.randn_like(eeg_signal) * 0.02
                eeg_signal = eeg_signal + noise
                if torch.rand(1) < 0.5:
                    shift = torch.randint(-5, 5, (1,)).item()
                    eeg_signal = torch.roll(eeg_signal, shifts=shift, dims=-1)

            image_vector = self.all_image_vectors[concept]
            text_vector = self.all_text_vectors[concept]

            if self.return_text:
                raw_text = f\"concept_{concept}\"
                return eeg_signal, image_vector, text_vector, raw_text

            return eeg_signal, image_vector, text_vector

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

        
        # === 【新增】 训练时数据增强 ===
        if self.mode == 'train':
            # 1. 高斯噪声注入 (Gaussian Noise)
            # 强度设为 0.01 ~ 0.05 (根据信号 Std≈1.0)
            noise = torch.randn_like(eeg_signal) * 0.02
            eeg_signal = eeg_signal + noise
            
            # 2. (可选) 随机时间偏移 (Temporal Shift)
            # 模拟脑电响应的微小时间差
            if torch.rand(1) < 0.5:
                shift = torch.randint(-5, 5, (1,)).item()
                eeg_signal = torch.roll(eeg_signal, shifts=shift, dims=-1)


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

        if self.return_text:
            # 尝试从 eeg_item_dict 中获取原始文本描述
            # 常见的键名可能是 'caption', 'text', 'prompt' 等
            raw_text = eeg_item_dict.get('caption') or eeg_item_dict.get('text') or ""
            return eeg_signal, image_vector, text_vector, raw_text

        return eeg_signal, image_vector, text_vector
