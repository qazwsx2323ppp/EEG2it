import numpy as np

v = np.load("saliency_outputs_expert/saliency_expert_visual.npy")
s = np.load("saliency_outputs_expert/saliency_expert_semantic.npy")

print("corr(v,s) =", np.corrcoef(v, s)[0,1])
print("mean |v-s| =", np.mean(np.abs(v - s)))
print("max  |v-s| =", np.max(np.abs(v - s)))