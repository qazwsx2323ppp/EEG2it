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
    # x: [C, T]
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
    """
    Build a compact trial index for ds003825 (THINGS RSVP EEG) and cache it in-memory.

    Returns dict with:
      - subjects: list[str]
      - subject_ids: torch.uint8 [N]
      - samples: torch.int32 [N]  (sample index at the EEG sampling rate)
      - concept_ids: torch.int16 [N] (objectnumber 0..1853)
    """
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
    samples: list[int] = []
    concept_ids: list[int] = []
    vhdr_paths: list[str] = [""] * len(subjects)

    # Stats for debugging "no usable trials" errors.
    stats = {
        "subjects_total": len(subjects),
        "subjects_with_files": 0,
        "subjects_raw_ok": 0,
        "subjects_events_ok": 0,
        "subjects_missing_columns": 0,
        "raw_errors": 0,
        "events_errors": 0,
        "trials_before_filter": 0,
        "trials_after_filter": 0,
        "small_eeg_files": 0,
    }
    missing_examples: list[dict] = []

    for sub in subjects:
        eeg_dir = os.path.join(bids_root, sub, "eeg")
        if not os.path.isdir(eeg_dir):
            continue

        import glob

        # Be tolerant to task naming differences by globbing.
        events_candidates = []
        events_candidates += glob.glob(os.path.join(eeg_dir, f"{sub}_task-*_events.tsv"))
        events_candidates += glob.glob(os.path.join(eeg_dir, f"{sub}_task-*_events.csv"))
        events_candidates += glob.glob(os.path.join(eeg_dir, f"{sub}_*_events.tsv"))
        events_candidates += glob.glob(os.path.join(eeg_dir, f"{sub}_*_events.csv"))
        events_candidates = sorted(set(events_candidates))

        vhdr_candidates = []
        vhdr_candidates += glob.glob(os.path.join(eeg_dir, f"{sub}_task-*_eeg.vhdr"))
        vhdr_candidates += glob.glob(os.path.join(eeg_dir, f"{sub}_*_eeg.vhdr"))
        vhdr_candidates += glob.glob(os.path.join(eeg_dir, "*.vhdr"))
        vhdr_candidates = sorted(set(vhdr_candidates))

        if not events_candidates or not vhdr_candidates:
            continue

        def _choose_best(paths: list[str], prefer_tsv: bool = True) -> str:
            def _score(p: str) -> tuple:
                base = os.path.basename(p).lower()
                is_rsvp = 0 if "task-rsvp" in base else 1
                if prefer_tsv:
                    ext_pri = 0 if base.endswith(".tsv") else 1
                else:
                    ext_pri = 0 if base.endswith(".vhdr") else 1
                return (is_rsvp, ext_pri, base)

            return sorted(paths, key=_score)[0]

        # Prefer RSVP + TSV for events; prefer RSVP for VHDR.
        events_path = _choose_best(events_candidates, prefer_tsv=True)
        vhdr_path = _choose_best(vhdr_candidates, prefer_tsv=False)

        stats["subjects_with_files"] += 1

        # Quick pointer-file heuristic: a real BrainVision .eeg is typically large.
        eeg_bin = ""
        for cand in glob.glob(os.path.join(eeg_dir, "*.eeg")) + glob.glob(os.path.join(eeg_dir, "*.EEG")):
            eeg_bin = cand
            break
        try:
            if os.path.isfile(eeg_bin) and os.path.getsize(eeg_bin) < 1_000_000:
                # A real BrainVision .eeg is typically tens/hundreds of MB; tiny files often indicate git-annex pointers.
                stats["small_eeg_files"] += 1
        except Exception:
            pass

        # Use raw length to drop out-of-range epochs early.
        try:
            raw = mne.io.read_raw_brainvision(vhdr_path, preload=False, verbose="ERROR")
            n_times = int(raw.n_times)
            sfreq = float(getattr(raw.info, "sfreq", raw.info.get("sfreq", 1000.0)))
            raw.close()
            stats["subjects_raw_ok"] += 1
            # Record the chosen vhdr path for this subject id (used later in __getitem__).
            vhdr_paths[subject_to_id[sub]] = vhdr_path
        except Exception:
            stats["raw_errors"] += 1
            continue

        try:
            # Auto-detect delimiter (tsv/csv) robustly.
            dfe = pd.read_csv(events_path, sep=None, engine="python")
            stats["subjects_events_ok"] += 1
        except Exception:
            stats["events_errors"] += 1
            continue

        if "objectnumber" not in dfe.columns or "onset" not in dfe.columns:
            stats["subjects_missing_columns"] += 1
            if len(missing_examples) < 3:
                missing_examples.append(
                    {
                        "subject": sub,
                        "events_path": events_path,
                        "columns": [str(c) for c in list(dfe.columns)[:30]],
                    }
                )
            continue

        if not include_teststim and "isteststim" in dfe.columns:
            dfe = dfe[dfe["isteststim"].astype(str) == "0"]

        # Keep only actual stimuli rows
        dfe = dfe[dfe["objectnumber"].astype(int) >= 0]

        if not include_targets and "istarget" in dfe.columns:
            dfe = dfe[dfe["istarget"].astype(str) == "0"]

        # Optional subsample to keep it manageable
        if trial_stride and trial_stride > 1:
            dfe = dfe.iloc[::trial_stride, :]

        stats["trials_before_filter"] += int(len(dfe))

        # Convert event timing to sample indices.
        # NOTE: ds003825 events.tsv stores 'onset' in seconds (float), and also provides a 'sample' column.
        if "sample" in dfe.columns:
            sample_arr = pd.to_numeric(dfe["sample"], errors="coerce").fillna(-1).astype(int).to_numpy()
        else:
            onset_sec = pd.to_numeric(dfe["onset"], errors="coerce").fillna(-1.0).to_numpy()
            sample_arr = np.rint(onset_sec * sfreq).astype(int)

        # Need 0..460ms window for later 20ms crop; require sample+win <= n_times.
        win = int(round(0.460 * sfreq))
        win = win if win > 0 else 460
        ok = (sample_arr >= 0) & ((sample_arr + win) <= n_times)
        dfe = dfe.loc[ok]
        sample_arr = sample_arr[ok]

        if len(dfe) == 0:
            continue

        stats["trials_after_filter"] += int(len(dfe))

        sid = subject_to_id[sub]
        subject_ids.extend([sid] * len(dfe))
        samples.extend(sample_arr.tolist())
        concept_ids.extend(dfe["objectnumber"].astype(int).tolist())

    if not subject_ids:
        raise ValueError(
            "No usable trials found under bids_root="
            f"{bids_root} "
            f"(subjects_total={stats['subjects_total']}, "
            f"subjects_with_files={stats['subjects_with_files']}, "
            f"subjects_raw_ok={stats['subjects_raw_ok']}, "
            f"subjects_events_ok={stats['subjects_events_ok']}, "
            f"subjects_missing_columns={stats['subjects_missing_columns']}, "
            f"raw_errors={stats['raw_errors']}, "
            f"events_errors={stats['events_errors']}, "
            f"trials_before_filter={stats['trials_before_filter']}, "
            f"trials_after_filter={stats['trials_after_filter']}, "
            f"small_eeg_files={stats['small_eeg_files']}). "
            f"missing_examples={missing_examples}. "
            "Check DS003825_ROOT points to the BIDS root with real BrainVision .eeg data (not git-annex pointer files)."
        )

    index = {
        "subjects": subjects,
        "subject_ids": torch.tensor(subject_ids, dtype=torch.uint8),
        "samples": torch.tensor(samples, dtype=torch.int32),
        "concept_ids": torch.tensor(concept_ids, dtype=torch.int16),
        "vhdr_paths": vhdr_paths,
    }
    _DS003825_INDEX_CACHE[key] = index
    return index

class TripletDataset(Dataset):
    """
    用于加载 (EEG, 图像向量, 文本向量) 三元组的数据集。
    严格复刻 DreamDiffusion 的预处理逻辑 (Slice + Interpolate)。
    """

    def __init__(self, cfg_data, mode='train', split_index=0):
        self.mode = mode
        if _is_rank0():
            print(f"正在加载 {mode} 数据（split_index={split_index}）...")

        # ---- ds003825 (BIDS) backend ----
        # 用法：把 cfg.data.eeg_path 指向 BIDS 根目录（包含 dataset_description.json），并提供 concept 级别向量：
        #   - cfg.data.image_vec_path / cfg.data.text_vec_path: shape [1854, D]
        # 训练集/验证/测试默认按 subject 划分（也可提供 splits_path 覆盖）。
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
            return_concept_id = bool(_safe_getattr(cfg_data, "return_concept_id", False))

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
            self.ds_samples = index["samples"]
            self.ds_concept_ids = index["concept_ids"]
            self.ds_vhdr_paths = index.get("vhdr_paths", [""] * len(self.ds_subjects))
            self.target_channels = target_channels
            self.bids_root = bids_root
            self.zscore = zscore
            self.zscore_eps = zscore_eps
            self.return_concept_id = return_concept_id

            # Concept-level targets: [num_concepts, D]
            # If stimuli images aren't available, text-only training can set image_vec_path to a missing/empty path.
            text_np = np.load(cfg_data.text_vec_path)
            image_path = _safe_getattr(cfg_data, "image_vec_path", "")
            if image_path and os.path.isfile(image_path):
                image_np = np.load(image_path)
            else:
                image_np = text_np

            self.all_image_vectors = torch.from_numpy(image_np).float()
            self.all_text_vectors = torch.from_numpy(text_np).float()
            self.num_available_vectors = min(len(self.all_image_vectors), len(self.all_text_vectors))

            # Splits: prefer explicit splits file; else split by subject deterministically.
            splits_path = _safe_getattr(cfg_data, "splits_path", None)
            if splits_path and os.path.isfile(splits_path):
                splits_data = torch.load(splits_path)
                raw_indices = splits_data["splits"][split_index][mode]
                self.indices = [int(i) for i in raw_indices]
            else:
                # Deterministic subject split (rotated by split_index):
                subjects = list(self.ds_subjects)
                if len(subjects) < 6:
                    raise ValueError("ds003825 subject split requires >= 6 subjects")
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

                # Filter trials by subject and vector availability (concept id used as index)
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
            sample = int(self.ds_samples[trial_index])
            concept = int(self.ds_concept_ids[trial_index])

            # Load/cached BrainVision raw per-subject per-worker
            raw = self._raw_cache.get(sub)
            if raw is None:
                vhdr_path = ""
                try:
                    sid = int(self.ds_subject_ids[trial_index])
                    vhdr_path = str(self.ds_vhdr_paths[sid]) if self.ds_vhdr_paths else ""
                except Exception:
                    vhdr_path = ""

                if not vhdr_path:
                    vhdr_path = os.path.join(self.bids_root, sub, "eeg", f"{sub}_task-rsvp_eeg.vhdr")
                raw = mne.io.read_raw_brainvision(vhdr_path, preload=False, verbose="ERROR")
                self._raw_cache[sub] = raw

            # Extract 0..460ms then apply the same 20ms crop (drop early artifact)
            sfreq = float(getattr(raw.info, "sfreq", raw.info.get("sfreq", 1000.0)))
            win = int(round(0.460 * sfreq))
            win = win if win > 0 else 460
            crop = int(round(0.020 * sfreq))
            crop = crop if crop >= 0 else 0

            seg = raw.get_data(start=sample, stop=sample + win, picks="eeg").astype(np.float32, copy=False)

            eeg_signal = seg.T  # (Time, Channel)
            eeg_signal = eeg_signal[crop:, :]
            eeg_signal = eeg_signal.T  # (Channel, Time) -> (63, 440)

            target_length = 512
            current_length = eeg_signal.shape[-1]
            if current_length != target_length:
                x = np.linspace(0, 1, current_length)
                x2 = np.linspace(0, 1, target_length)
                f = interp1d(x, eeg_signal, axis=-1)
                eeg_signal = f(x2)

            # Pad channels to target_channels (DreamDiffusion backbone expects 128 chans)
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
            if getattr(self, "return_concept_id", False):
                return eeg_signal, image_vector, text_vector, concept
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

        return eeg_signal, image_vector, text_vector
