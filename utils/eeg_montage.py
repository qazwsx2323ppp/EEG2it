# utils/eeg_montage.py
import torch
from typing import List, Tuple, Dict, Optional

def try_load_ch_names(eeg_path: str) -> Optional[List[str]]:
    """
    尝试从你保存的 eeg .pt 文件里读取通道名。
    你的 dataset 现在只取了 ['dataset']，但文件里可能还有其它 key。
    """
    obj = torch.load(eeg_path, map_location="cpu")
    for k in ["ch_names", "channels", "channel_names"]:
        if isinstance(obj, dict) and k in obj and isinstance(obj[k], (list, tuple)) and len(obj[k]) == 128:
            return list(obj[k])
    return None

def get_montage_and_names(eeg_path: str) -> Tuple[List[str], "mne.channels.DigMontage", bool]:
    """
    返回 (ch_names, montage, is_template)
    is_template=True 表示用的是模板 montage（需要你在论文里声明近似）
    """
    import mne

    ch_names = try_load_ch_names(eeg_path)
    if ch_names is not None:
        # 如果你能读到真实通道名：优先尝试匹配标准 montage（若找不到 actiCAP，可先用 biosemi128 占位）
        std = mne.channels.make_standard_montage("biosemi128")
        # 仅保留共有通道名
        common = [ch for ch in ch_names if ch in std.ch_names]
        if len(common) >= 100:
            std = std.copy()
            std.rename_channels({ch: ch for ch in std.ch_names})  # no-op, for clarity
            return ch_names, std, False

    # 兜底：完全没有通道名时，直接用模板并按模板顺序假设你的数据通道顺序一致
    std = mne.channels.make_standard_montage("biosemi128")
    return std.ch_names, std, True

def rough_region_group(ch_name: str) -> str:
    """
    Fig.4 风格的粗分区：按通道名前缀粗略映射到皮层区域。
    这对应 Palazzo 文中“rough matching with brain cortices”的逻辑。
    """
    name = ch_name.upper()

    if name.startswith("FP") or name.startswith("AF") or name.startswith("F"):
        return "Frontal"
    if name.startswith("T") or name.startswith("FT") or name.startswith("TP"):
        return "Temporal"
    if name.startswith("C"):
        return "Central"
    if name.startswith("P") or name.startswith("CP"):
        return "Parietal"
    if name.startswith("O") or name.startswith("PO"):
        return "Occipital"
    return "Other"
