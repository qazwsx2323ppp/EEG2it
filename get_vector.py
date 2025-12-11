import torch
import clip
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 你的 EEG 数据集路径 (必须是那个 .pth 文件)
EEG_PATH = "data/EEG_data/eeg_55_95_std.pth" 

# 2. ImageNet 图片的根目录 (里面应该是 n021xxx 这种类别文件夹)
# 确保这个路径下包含 dataset['images'] 里引用的图片
IMAGE_ROOT = "data/image_data"

# 3. 输出路径
OUTPUT_IMG_PATH = "data/image_vectors_fixed.npy"
OUTPUT_TXT_PATH = "data/text_vectors_fixed.npy"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL = "ViT-B/32" # 确保和你 Config 里的维度一致 (ViT-B/32 输出 512维)
# ===========================================

def main():
    print(f"正在加载 EEG 数据集元数据: {EEG_PATH} ...")
    try:
        data = torch.load(EEG_PATH, map_location='cpu')
        # 核心：获取数据集内部定义的图片列表
        # 这是唯一的真理标准，必须按这个顺序生成向量！
        target_images = data['images'] 
        print(f"成功获取图片列表，共 {len(target_images)} 张图片。")
        print(f"样例 [0]: {target_images[0]}")
        print(f"样例 [100]: {target_images[100]}")
    except KeyError:
        print("错误：无法在 .pth 文件中找到 'images' 键。")
        print("可用键名:", data.keys())
        return

    print(f"正在加载 CLIP 模型: {CLIP_MODEL} ...")
    model, preprocess = clip.load(CLIP_MODEL, device=DEVICE)
    
    img_vectors = []
    txt_vectors = []
    
    print("开始生成对齐向量...")
    
    # 遍历列表，确保生成的向量顺序与 target_images 严格一致
    for img_name in tqdm(target_images):
        # target_images 里的文件名通常是 'n01443537_22563.JPEG'
        # 我们需要找到它在磁盘上的真实路径
        
        # 解析类别文件夹 (文件名下划线前面部分)
        class_folder = img_name.split('_')[0] 
        
        # 拼凑可能的路径
        # 路径格式通常是: IMAGE_ROOT / class_folder / img_name
        full_path = os.path.join(IMAGE_ROOT, class_folder, img_name)
        
        # 容错：如果 dataset 里没后缀，加上 .JPEG 试试
        if not os.path.exists(full_path):
             if not img_name.lower().endswith('.jpeg'):
                 full_path = os.path.join(IMAGE_ROOT, class_folder, img_name + '.JPEG')

        # ---------------- 图像处理 ----------------
        try:
            image = Image.open(full_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                img_feat = model.encode_image(image_input)
                # 归一化 (CLIP 标准操作)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                img_vectors.append(img_feat.cpu().numpy())
        except Exception as e:
            print(f"\n[警告] 无法读取图片: {img_name}")
            print(f"尝试路径: {full_path}")
            print(f"错误信息: {e}")
            # 填充随机向量或零向量，保证索引不乱 (非常重要！)
            # 这样即使缺图，第 N 个向量依然对应第 N 个 EEG
            img_vectors.append(np.zeros((1, 512), dtype=np.float32))

        # ---------------- 文本处理 ----------------
        # 使用类别名作为文本: "a photo of [CLASS_ID]"
        # 如果你想更精确，可以加载 imagenet_classes.txt 将 ID 映射为单词
        text_prompt = f"a photo of {class_folder}" 
        text_input = clip.tokenize([text_prompt]).to(DEVICE)
        
        with torch.no_grad():
            txt_feat = model.encode_text(text_input)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            txt_vectors.append(txt_feat.cpu().numpy())

    # 合并并保存
    final_img_vecs = np.concatenate(img_vectors, axis=0)
    final_txt_vecs = np.concatenate(txt_vectors, axis=0)
    
    print(f"\n保存中... 图像向量形状: {final_img_vecs.shape}")
    np.save(OUTPUT_IMG_PATH, final_img_vecs)
    np.save(OUTPUT_TXT_PATH, final_txt_vecs)
    print("✅ 完成！现在向量与 .pth 文件的索引已严格对齐。")

if __name__ == "__main__":
    main()