import os
import torch
import clip
from PIL import Image
import numpy as np
from tqdm import tqdm

# ================= 配置区域 =================
# 图片数据集根目录 (请修改为你服务器上的真实路径)
# 结构应该是: IMAGE_ROOT / n02106662 / xxxx.JPEG
IMAGE_ROOT = "data/image_data" 

# 输出路径
OUTPUT_IMG_PATH = "data/image_vectors_fixed.npy"
OUTPUT_TXT_PATH = "data/text_vectors_fixed.npy"

# 模型选择 (你的 config 中 embedding_dim=512，所以必须用 ViT-B/32)
CLIP_MODEL_NAME = "ViT-B/32" 
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ===========================================

def main():
    print(f"正在加载 CLIP 模型: {CLIP_MODEL_NAME} 到 {DEVICE}...")
    model, preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE)
    
    # 1. 扫描并排序文件
    # ImageNet-EEG 的标准顺序通常是：按类别文件夹排序 -> 按文件名排序
    print("正在扫描文件目录...")
    class_folders = sorted([d for d in os.listdir(IMAGE_ROOT) if os.path.isdir(os.path.join(IMAGE_ROOT, d))])
    
    all_image_paths = []
    all_class_names = [] # 用于生成文本向量
    
    for class_folder in class_folders:
        folder_path = os.path.join(IMAGE_ROOT, class_folder)
        # 获取该类别下的所有图片并排序
        images = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        for img_name in images:
            all_image_paths.append(os.path.join(folder_path, img_name))
            # 简单的文本提示: "This is a photo of [class_id]" 或者是具体的类别名
            # 这里我们暂时用类别文件夹名作为 prompt，或者你可以替换为更具体的标签
            all_class_names.append(f"a photo of {class_folder}")

    total_images = len(all_image_paths)
    print(f"扫描完成，找到 {total_images} 张图片 (预期应为 2000)。")
    
    # 2. 提取图像特征
    print("开始提取图像特征...")
    image_features_list = []
    
    # 分批处理
    for i in tqdm(range(0, total_images, BATCH_SIZE)):
        batch_paths = all_image_paths[i : i + BATCH_SIZE]
        batch_inputs = []
        valid_indices = [] # 记录这个 batch 里哪些图是好的
        
        # 预处理图片
        for idx, p in enumerate(batch_paths):
            try:
                image = Image.open(p).convert("RGB")
                processed = preprocess(image)
                batch_inputs.append(processed)
                valid_indices.append(idx)
            except Exception as e:
                print(f"\n[警告] 图片损坏，已跳过并填充零向量: {p}")
                # 坏图不加入 batch_inputs，稍后在结果里补 0
                pass
        
        if len(batch_inputs) > 0:
            # 堆叠并送入模型
            batch_tensor = torch.stack(batch_inputs).to(DEVICE)
            with torch.no_grad():
                # Encode 并归一化
                features = model.encode_image(batch_tensor)
                features = features / features.norm(dim=1, keepdim=True)
                features_np = features.cpu().numpy()
        
        # 组装结果 (含坏图填充)
        batch_outputs = np.zeros((len(batch_paths), 512), dtype=np.float32)
        
        if len(batch_inputs) > 0:
            # 把算出来的特征填回对应的位置
            for real_idx, vec_idx in enumerate(valid_indices):
                batch_outputs[vec_idx] = features_np[real_idx]
                
        image_features_list.append(batch_outputs)

    # 合并保存
    final_img_vecs = np.concatenate(image_features_list, axis=0)
    print(f"图像向量生成完毕，形状: {final_img_vecs.shape}")
    np.save(OUTPUT_IMG_PATH, final_img_vecs)
    
    # 3. 提取文本特征 (可选，确保一一对应)
    print("开始提取文本特征...")
    text_features_list = []
    
    for i in tqdm(range(0, total_images, BATCH_SIZE)):
        batch_texts = all_class_names[i : i + BATCH_SIZE]
        tokens = clip.tokenize(batch_texts).to(DEVICE)
        
        with torch.no_grad():
            features = model.encode_text(tokens)
            features = features / features.norm(dim=1, keepdim=True)
            text_features_list.append(features.cpu().numpy())
            
    final_txt_vecs = np.concatenate(text_features_list, axis=0)
    print(f"文本向量生成完毕，形状: {final_txt_vecs.shape}")
    np.save(OUTPUT_TXT_PATH, final_txt_vecs)
    
    print("\n✅ 所有向量已重新生成并保存！")
    print(f"请在 configs/triplet_config.yaml 中更新路径指向: \n{OUTPUT_IMG_PATH}\n{OUTPUT_TXT_PATH}")

if __name__ == "__main__":
    main()