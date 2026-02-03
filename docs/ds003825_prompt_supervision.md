# ds003825：方案一（双 Prompt）监督

ds003825 常见情况是没有 `stimuli/` 图片，因此无法稳定用 CLIP image encoder 生成 `image_vec`。
“方案一”通过 **两种不同的文本 prompt** 用 CLIP text encoder 生成两套目标向量：

- `text_vec`：偏语义（概念）  
  例：`a {concept}`
- `image_vec`：偏视觉（照片语境）  
  例：`a photo of a {concept}`

两套向量不同但相关，从而让模型的两个 head 有真实的“解耦”学习目标。

## 生成向量

在项目根目录执行（Linux 示例）：

```bash
python utils/ds003825_embed_concepts.py \
  --bids-root /path/to/ds003825 \
  --out-text data/ds003825_concept_text.npy \
  --out-image data/ds003825_concept_image.npy \
  --prompt-text "a {}" \
  --prompt-image "a photo of a {}" \
  --allow-download
```

说明：
- 不想联网就去掉 `--allow-download`（前提是模型权重已缓存本地）。
- 如果你想退回 text-only，可加 `--image-same-as-text`（此时 `alpha` 建议设为 `0.0`）。

## 训练

使用 `main_ds.py` + `configs/ds003825_triplet_config.yaml`。
