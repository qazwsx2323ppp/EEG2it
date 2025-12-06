import torch
import clip
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 你的 EEG 数据集路径
EEG_PATH = "data/EEG_data/eeg_55_95_std.pth" 

# 2. ImageNet 图片的根目录
IMAGE_ROOT = "data/image_data"

# 3. 输出路径 (建议改个名以示区别)
OUTPUT_IMG_PATH = "data/image_vectors_aligned.npy"
OUTPUT_TXT_PATH = "data/text_vectors_aligned.npy"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL = "ViT-B/32"
# ===========================================

def main():
    print(f"正在加载 EEG 数据集元数据: {EEG_PATH} ...")
    try:
        # 加载 .pth 文件
        data = torch.load(EEG_PATH, map_location='cpu')
        
        # 【关键步骤】直接获取官方定义的图片顺序
        # 参考 thought2text/datautils.py 的逻辑
        if 'images' in data:
            target_images = data['images']
        else:
            # 兼容某些数据集结构差异，有的可能在 'dataset' 外部
            print("警告：未在根目录找到 'images'，尝试在 dataset 内部查找（如果适用）...")
            target_images = data.get('images', [])
            
        if not target_images:
            raise KeyError("无法找到 'images' 列表，请检查 .pth 文件结构")

        print(f"✅ 成功获取图片列表，共 {len(target_images)} 张。")
        print(f"   Index 0 对应: {target_images[0]}")
        print(f"   Index 100 对应: {target_images[100]}")
        
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        return

    print(f"正在加载 CLIP 模型: {CLIP_MODEL} ...")
    # 注意：确保安装了正确的 clip (pip install git+https://github.com/openai/CLIP.git)
    model, preprocess = clip.load(CLIP_MODEL, device=DEVICE)
    
    img_vectors = []
    txt_vectors = []
    
    print("🚀 开始生成严格对齐的向量...")
    
    # 遍历列表，顺序绝对不能乱！
    for img_name in tqdm(target_images):
        # 1. 拼凑图片路径
        # img_name 通常是 'n02106662_123.JPEG'
        class_folder = img_name.split('_')[0] 
        
        # 优先尝试：IMAGE_ROOT/class_folder/img_name
        full_path = os.path.join(IMAGE_ROOT, class_folder, img_name)
        
        # 容错逻辑：有的文件名可能没后缀，或者路径结构不同
        if not os.path.exists(full_path):
             # 尝试加 .JPEG
             if not img_name.lower().endswith('.jpeg') and not img_name.lower().endswith('.jpg'):
                 test_path = os.path.join(IMAGE_ROOT, class_folder, img_name + '.JPEG')
                 if os.path.exists(test_path):
                     full_path = test_path
        
        # ---------------- 图像编码 ----------------
        try:
            image = Image.open(full_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                img_feat = model.encode_image(image_input)
                # 归一化 (CLIP 标准)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                img_vectors.append(img_feat.cpu().numpy())
                
        except Exception as e:
            print(f"\n[警告] 图片读取失败: {img_name}")
            print(f"       路径: {full_path}")
            # 填充零向量，绝不能跳过，否则后续索引会全部错位！
            img_vectors.append(np.zeros((1, 512), dtype=np.float32))

        # ---------------- 文本编码 ----------------
        # 简单 prompt: "a photo of [CLASS_ID]"
        text_prompt = f"a photo of {class_folder}" 
        text_input = clip.tokenize([text_prompt]).to(DEVICE)
        
        with torch.no_grad():
            txt_feat = model.encode_text(text_input)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            txt_vectors.append(txt_feat.cpu().numpy())

    # 合并保存
    final_img_vecs = np.concatenate(img_vectors, axis=0)
    final_txt_vecs = np.concatenate(txt_vectors, axis=0)
    
    print(f"\n💾 保存向量到硬盘...")
    np.save(OUTPUT_IMG_PATH, final_img_vecs)
    np.save(OUTPUT_TXT_PATH, final_txt_vecs)
    print(f"✅ 完成！生成了 {len(final_img_vecs)} 个对齐向量。")

if __name__ == "__main__":
    main()