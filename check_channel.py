import torch
import numpy as np

# 1. 加载你的数据文件
data_path = 'data/EEG_data/eeg_55_95_std.pth'  # 请修改为你的实际路径
try:
    loaded_data = torch.load(data_path, map_location='cpu')
    print(">>> 数据加载成功！")
except Exception as e:
    print(f"XXX 加载失败: {e}")
    exit()

# 2. 检查是否存在通道名称元数据
if isinstance(loaded_data, dict):
    # 常见键名探测
    keys_to_check = ['ch_names', 'channel_names', 'info', 'channels']
    found_names = None
    
    for key in keys_to_check:
        if key in loaded_data:
            found_names = loaded_data[key]
            print(f">>> 在键 '{key}' 中找到通道列表！")
            break
            
    if found_names:
        print(f"通道总数: {len(found_names)}")
        print("前 5 个通道:", found_names[:5])
        print("后 5 个通道:", found_names[-5:])
        
        # 3. 验证是否符合 BrainVision 标准 (简单的启发式检查)
        standard_start = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz']
        
        # 处理可能的大小写或格式差异
        current_start = [str(n).strip() for n in found_names[:5]]
        
        is_match = True
        for i in range(3): # 只检查前3个
            if current_start[i].lower() != standard_start[i].lower():
                is_match = False
                break
        
        if is_match:
            print("\n✅ 通道顺序看起来符合标准 (Fp1, Fp2 开头)！")
            print("主要排查方向转为：数据归一化 (Data Scaling)")
        else:
            print(f"\n❌ 通道顺序可能错误！")
            print(f"期望开头: {standard_start}")
            print(f"实际开头: {current_start}")
            print(">>> 请务必重排数据列！")
            
    else:
        print("\n⚠️ 未在文件中找到通道名称列表。")
        print("建议：请找到数据集的原始说明文档 (channels.tsv) 或手动可视化验证。")
        print("如果数据集中第1个通道看起来像眼电 (Fp1)，第30个像视觉 (Oz)，则顺序可能是对的。")

# 4. 再次检查数据统计 (Double Check)
# 这一步是为了确认是否需要 Z-Score 归一化
if 'dataset' in loaded_data:
    sample_eeg = loaded_data['dataset'][0]['eeg']
    print(f"\n[数据统计] 样本 0:")
    print(f"Mean: {sample_eeg.mean():.4f}")
    print(f"Std : {sample_eeg.std():.4f} (如果远不为1，必须加归一化!)")
    print(f"Max : {sample_eeg.max():.4f}")
    print(f"Min : {sample_eeg.min():.4f}")