import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

def check_alignment():
    # 1. 加载向量
    print("正在加载向量...")
    try:
        # 替换为你的实际路径
        img_vecs = np.load("data/image_vectors.npy") 
        # 如果有 text_vectors 也可以加载来看看
        # text_vecs = np.load("data/text_vectors.npy")
    except Exception as e:
        print(f"加载失败: {e}")
        return

    print(f"向量形状: {img_vecs.shape}") # 预期 (1996, 512)

    # 2. 计算自相似矩阵 (Self-Similarity Matrix)
    # 取前 200 个向量做演示 (包含 4 个类别)
    # 如果错位发生在后面，你可能需要画出全部，或者分段画
    N = 300 
    vecs_tensor = torch.from_numpy(img_vecs[:N]).float()
    
    # 归一化
    vecs_tensor = F.normalize(vecs_tensor, p=2, dim=1)
    
    # 计算相似度 (Cosine Similarity)
    # Shape: (N, N)
    sim_matrix = torch.matmul(vecs_tensor, vecs_tensor.T)
    
    # 3. 绘图
    plt.figure(figsize=(10, 10))
    plt.imshow(sim_matrix.numpy(), cmap='viridis', vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.title(f"Image Vectors Self-Similarity (First {N} samples)")
    plt.xlabel("Index")
    plt.ylabel("Index")
    
    # 辅助线：每 50 个画一条线 (ImageNet-EEG 一个类别 50 张图)
    for i in range(0, N, 50):
        plt.axvline(x=i, color='red', linestyle='--', alpha=0.5)
        plt.axhline(y=i, color='red', linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    plt.savefig("alignment_check.png")
    print(">>>以此图判断：高亮方块是否严格对齐红色网格？")
    print(">>> 已保存可视化结果到 alignment_check.png")

if __name__ == "__main__":
    check_alignment()