import numpy as np
img_vecs = np.load("/data/image_vectors.npy")
txt_vecs = np.load("/data/text_vectors.npy")
print("Image vectors shape:", img_vecs.shape)
print("Text vectors shape:", txt_vecs.shape)