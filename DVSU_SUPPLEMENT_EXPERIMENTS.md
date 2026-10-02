# DVSU 补充实验执行说明（供 Codex 直接执行）

## 1. 任务目标与边界

你正在 `EEG2it` 仓库的 `dvsu-revision` 分支上工作。目标是在尽量短的时间内补充能够支撑 DVSU 论文核心叙事的实验代码和结果，不重新设计整套模型，不为了得到理想结论而挑选样本或修改指标。

本轮只回答两个问题：

1. SERE 的 visual / shared / semantic experts 是否表现出任务相关的功能差异，输入相关路由是否真实工作？
2. 如果完整生成代码和全部 checkpoint 已经可用，EEG 感知条件与生成文本条件是否对图像生成具有互补作用？

优先级如下：

- **P0（必须完成）**：SERE 专家-目标相似度、逐样本路由权重、专家切除敏感性分析。只复用现有 Stage-1 checkpoint，不重新训练。
- **P1（条件满足才完成）**：DPG 三条件生成与评估。只有 EEG encoder、Qwen EEG projector、EEG-to-SD image projector、Qwen 本地模型和 Stable Diffusion 本地模型全部存在并能严格加载时才能执行。
- **P2（可选）**：根据 P1 的全量结果自动选取代表性样本并生成论文定性图。

不要在本轮进行新数据集训练、频段重训、大规模超参数搜索、专家数量搜索或完整模型从头训练。不要修改论文 `.tex`；本轮先产出可信实验结果。

## 2. 开始前必须检查

先在 VS Code 终端中执行：

```bash
git rev-parse --abbrev-ref HEAD
git status --short
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

验收要求：

- 当前分支必须是 `dvsu-revision`。如果不是，停止并切换分支；不要在 `main` 上修改。
- 不覆盖用户已有的未提交修改。
- 从仓库根目录运行所有命令。
- 数据、checkpoint 和本地大模型不提交到 Git。
- 不自动下载或替换 Qwen、Stable Diffusion、CLIP 权重。缺少时列出缺失路径并暂停对应实验。

确认并记录以下资源的绝对路径：

```text
EEG_DATA_PATH          EEG_data/eeg_55_95_std.pth
IMAGE_VEC_PATH         image_vectors_aligned.npy
TEXT_VEC_PATH          text_vectors_aligned.npy
SPLITS_PATH            block_splits_by_image_all.pth
EEG_ENCODER_CKPT       Stage-1 best_eeg_encoder.pth
QWEN_PROJECTOR_CKPT    Stage-2 best_eeg_projector.pth
EEG_IMG_PROJ_CKPT      EEG-to-SD projector checkpoint
QWEN_MODEL_DIR         本地 Qwen2.5-Omni 模型目录
SD_MODEL_DIR           本地 Stable Diffusion 1.5 diffusers 目录
CLIP_MODEL_DIR         本地 CLIP ViT-B/32 目录
CAPTIONS_DIR           图像关联描述目录
IMAGE_ROOT             真实刺激图像根目录（用于 LPIPS/FID/定性图）
```

先写一个只读的资源检查脚本或在实验脚本中提供 `--check-only`。检查 checkpoint 时输出 missing/unexpected keys；若核心 backbone、expert heads、router 或 projector 缺失，不得静默使用随机初始化继续生成。

## 3. 代码现状与注意事项

执行前先阅读以下文件：

```text
models/clip_models.py
main.py
dataset.py
train_eeg_qwen_projector.py
train_eeg_img_proj.py
Omni_generate/low-VRAM-mode/eeg_to_sd_full.py
Omni_generate/low-VRAM-mode/painter_sd.py
vis_expert_attribution_topk.py
utils/eval_moe.py
```

已知事实：

- `SpatialMoEEncoder` 已包含 visual、semantic、fusion(shared) 三个 expert heads 和输入相关 router。
- `forward(x, ablation=...)` 已支持 `kill_visual`、`kill_semantic`、`kill_fusion`。
- `router_mode` 已支持 `moe`、`uniform`、`fusion` 等模式。
- 当前 `forward` 只返回 batch 平均的 `w_vis_img` 和 `w_sem_txt`，不足以画逐样本分布。
- 当前 `utils/eval_moe.py` 使用了已经不存在的构造参数，并假设 dataset 只返回三个元素，属于过时代码。不要直接运行后把结果写进论文。
- 当前 `vis_expert_attribution_topk.py` 也有硬编码路径和 dataset tuple 长度问题，而且在缺少真实 channel names 时会回退到模板 montage。模板脑地形图不能作为主要论文证据。
- `Omni_generate/low-VRAM-mode/eeg_to_sd_full.py` 已能够先生成 prompt，再选择是否向 SD 注入 EEG visual token；这可以复用来做 DPG 条件移除实验。
- `Omni_generate/low-VRAM-mode/demo_min_eeg_to_sd.py` 使用固定 prompt，不是完整 EEG-to-text-to-image 流程，不可用于 DPG 主实验。

所有新增实验代码放在：

```text
experiments/
```

所有实验结果默认写入：

```text
artifacts/supplement_experiments/<experiment_name>/
```

如果 `.gitignore` 会忽略 PNG/JSON/CSV，至少提交代码和一个 Markdown 结果摘要；大型生成图像不要提交。不要为提交结果而放宽整个仓库的忽略规则。

## 4. P0-A：SERE 专家与路由功能分析

### 4.1 实现文件

新增：

```text
experiments/analyze_sere_experts.py
```

命令行参数至少包括：

```text
--config
--encoder-ckpt
--split {train,val,test}
--split-index
--output-dir
--batch-size
--num-workers
--device
--seed
--max-samples（0 表示全部）
--bootstrap-iters（默认 1000）
```

### 4.2 最小模型改动

在不改变默认训练行为和现有返回格式的前提下，为 `SpatialMoEEncoder.forward` 增加可选的详细输出，例如：

```python
forward(self, x, ablation=None, return_details=False)
```

`return_details=False` 时必须与当前行为完全一致。`return_details=True` 时，在第三个字典中额外返回逐样本 tensor：

```text
emb_visual
emb_semantic
emb_shared（代码中的 expert_fusion_head）
gate_visual_img
gate_shared_img
gate_semantic_txt
gate_shared_txt
```

不要 `.mean()` 后再保存。每个 gate 的第一维必须与 batch size 相同。检查：

```text
gate_visual_img + gate_shared_img ≈ 1
gate_semantic_txt + gate_shared_txt ≈ 1
```

### 4.3 专家-目标相似度

在 `test` split 上，分别计算三个专家输出与两个冻结目标的余弦相似度：

| Expert | CLIP image target | CLIP text target |
|---|---:|---:|
| Visual | mean + 95% CI | mean + 95% CI |
| Shared | mean + 95% CI | mean + 95% CI |
| Semantic | mean + 95% CI | mean + 95% CI |

要求：

- 专家输出和目标均先做 L2 normalization。
- 置信区间使用固定 seed 的 trial-level bootstrap，默认 1000 次。
- 同时保存每个样本的原始相似度，不只保存均值。
- 不预设结果一定符合命名。如果 visual expert 没有更偏向 image target，必须如实报告。

### 4.4 路由权重分布

保存每个测试 trial 的四个逐样本权重，并绘制：

1. image route：visual 与 shared 权重的小提琴图或箱线图；
2. text route：semantic 与 shared 权重的小提琴图或箱线图。

图中标注样本数、均值和标准差。禁止只画 batch mean。若权重标准差接近 0，结果应说明 router 近似常数路由，不能描述成强输入自适应。

### 4.5 输出文件

至少生成：

```text
expert_target_similarity_per_sample.csv
expert_target_similarity_summary.csv
router_weights_per_sample.csv
summary.json
expert_target_similarity_heatmap.png
expert_target_similarity_heatmap.pdf
router_weight_distribution.png
router_weight_distribution.pdf
run_config.yaml
run.log
```

图像使用适合论文的字号、白底和 300 dpi；同时导出矢量 PDF。

### 4.6 运行命令模板

```bash
python -m experiments.analyze_sere_experts \
  --config configs/triplet_config.yaml \
  --encoder-ckpt "$EEG_ENCODER_CKPT" \
  --split test \
  --split-index 0 \
  --output-dir artifacts/supplement_experiments/sere_expert_analysis \
  --batch-size 64 \
  --num-workers 4 \
  --device cuda \
  --seed 42 \
  --bootstrap-iters 1000
```

先使用 `--max-samples 32` 做 smoke test，确认无 NaN、维度正确和输出完整，再运行全量测试集。

## 5. P0-B：专家切除与路由敏感性

### 5.1 实现文件

新增：

```text
experiments/evaluate_sere_masking.py
```

使用同一个训练完成的 checkpoint，在推理阶段评估：

```text
full_moe
kill_visual
kill_semantic
kill_shared（对应现有 kill_fusion）
uniform_router
```

这是 **inference-time masking sensitivity analysis**，不是重新训练后的 architecture ablation。代码、图注和结果摘要都必须使用这个准确名称。

### 5.2 指标协议

分别评估 image branch 和 text branch 的检索：

```text
R@1
R@5
mean positive cosine similarity
mean negative cosine similarity
```

候选集合必须按 `target_id` 去重，并仅使用当前 test split 中的 unique targets。不要让同一图像的多个 EEG trial 在候选库中重复出现。保存每个 query 是否 Top-1/Top-5 命中的逐样本结果，并用 paired bootstrap 计算 full 与各切除条件差值的 95% CI。

论文预期关注的是选择性，而不只是整体下降：

- `kill_visual` 是否对 image branch 的影响大于对 text branch 的影响；
- `kill_semantic` 是否对 text branch 的影响大于对 image branch 的影响；
- `kill_shared` 是否同时影响两个分支；
- `uniform_router` 是否弱于 learned router。

如果数据不支持上述趋势，不要筛选 split、seed 或样本来制造趋势。

### 5.3 输出与命令

输出：

```text
masking_metrics.csv
masking_metrics.json
masking_per_sample.csv
masking_delta_with_ci.csv
run_config.yaml
run.log
```

命令模板：

```bash
python -m experiments.evaluate_sere_masking \
  --config configs/triplet_config.yaml \
  --encoder-ckpt "$EEG_ENCODER_CKPT" \
  --split test \
  --split-index 0 \
  --output-dir artifacts/supplement_experiments/sere_masking \
  --batch-size 64 \
  --device cuda \
  --seed 42 \
  --bootstrap-iters 1000
```

## 6. P1：DPG 条件移除实验（仅在资源齐全时）

### 6.1 强制门槛

只有以下全部通过才能执行：

- EEG encoder checkpoint 可加载；
- Qwen EEG projector checkpoint 可加载；
- EEG image projector checkpoint 可加载；
- Qwen 本地模型目录可加载；
- Stable Diffusion 本地模型目录可加载；
- dataset、caption 与 ground-truth image 可以通过同一 `target_id` 对齐；
- 不存在随机初始化 projector；
- 使用 3 个样本能够完成端到端 smoke test。

任何一项失败，就在 `BLOCKED_DPG.md` 中记录缺失项，完成 P0 后停止 P1。不要用固定 prompt、oracle label 或随机 projector 冒充完整 DPG。

### 6.2 实现方式

优先新增独立脚本：

```text
experiments/generate_dpg_conditions.py
```

可以复用 `eeg_to_sd_full.py` 的加载和推理函数，但不要让每个样本重复加载 Qwen 或 SD。模型只加载一次。

三种正式条件：

| Condition | Sample-specific generated text | EEG visual token |
|---|---:|---:|
| perceptual_only | No，使用统一中性 prompt `a photo` | Yes |
| semantic_only | Yes | No |
| joint | Yes | Yes |

可选增加：

```text
oracle_text_joint
```

它使用参考 caption + EEG token，只能作为 upper bound，不能与正式方法混为一谈。

实验控制要求：

- 每个 target 只选择一个确定的 EEG trial，按第一次出现选择，构成 333 个 unique test targets；实际数量以 split 为准并写入日志。
- 对同一 target 的三个条件使用完全相同的 seed、negative prompt、25 denoising steps、CFG=7.5、分辨率和 SD checkpoint。
- 同一个样本的 generated text 只生成一次并缓存，`semantic_only` 与 `joint` 必须复用完全相同的文本。
- 每个样本写 JSONL 元数据：dataset index、原始 EEG index、target_id、generated prompt、condition、seed、输出路径、所有 checkpoint 路径和 checkpoint SHA256。
- 如果 prompt 为空，不得偷偷使用 ground-truth 类别回退；记录为空并使用统一 fallback `a photo`，同时统计 fallback 比例。

### 6.3 生成命令模板

先运行 3 个样本：

```bash
python -m experiments.generate_dpg_conditions \
  --config configs/triplet_config.yaml \
  --encoder-ckpt "$EEG_ENCODER_CKPT" \
  --qwen-projector-ckpt "$QWEN_PROJECTOR_CKPT" \
  --eeg-img-proj-ckpt "$EEG_IMG_PROJ_CKPT" \
  --qwen-model-dir "$QWEN_MODEL_DIR" \
  --sd-model-dir "$SD_MODEL_DIR" \
  --captions-dir "$CAPTIONS_DIR" \
  --image-root "$IMAGE_ROOT" \
  --split test \
  --split-index 0 \
  --unique-targets \
  --max-targets 3 \
  --conditions perceptual_only semantic_only joint \
  --output-dir artifacts/supplement_experiments/dpg_conditions_smoke \
  --seed 42
```

检查三组图像、prompt 与元数据无误后，将 `--max-targets 3` 改为 `--max-targets 0` 运行全量。

## 7. P1 评估：图像、文本与图文一致性

新增：

```text
experiments/evaluate_dpg_conditions.py
```

至少输出以下指标：

1. 与论文当前协议一致的 SA@1 / SA@5。必须复用原论文相同的分类器和类别映射；若仓库中找不到，先报告缺失，不能静默换成其他分类器后仍称为 SA@1 / SA@5。
2. Generated image 与 generated text 的 CLIP cosine similarity（跨输出一致性）。
3. Generated image 与 reference caption 的 CLIP cosine similarity（刺激语义正确性）。
4. Generated text 与 reference caption 的 CLIP cosine similarity。
5. 若真实刺激图像路径可靠：LPIPS；FID 作为补充指标，并明确 N 较小时的不稳定性。

对 per-sample 指标报告 mean、standard deviation 和 95% bootstrap CI。对 `joint - semantic_only`、`joint - perceptual_only` 使用 paired bootstrap。FID 不做逐样本 bootstrap，除非实现和计算资源允许。

输出：

```text
dpg_metrics_summary.csv
dpg_metrics_summary.json
dpg_metrics_per_sample.csv
dpg_pairwise_delta_with_ci.csv
evaluation_protocol.md
run.log
```

评估命令模板：

```bash
python -m experiments.evaluate_dpg_conditions \
  --generation-dir artifacts/supplement_experiments/dpg_conditions \
  --clip-model-dir "$CLIP_MODEL_DIR" \
  --image-root "$IMAGE_ROOT" \
  --output-dir artifacts/supplement_experiments/dpg_evaluation \
  --device cuda \
  --seed 42 \
  --bootstrap-iters 1000
```

## 8. P2：定性对比图（可选）

只有在 P1 全量评估完成后再生成：

```text
experiments/make_dpg_qualitative_figure.py
```

不要人工挑选最好看的样本。根据预先定义的 `joint - best_single_condition` reference-caption CLIP improvement，自动选择：

- 2 个高分位样本；
- 2 个中位附近样本；
- 2 个低分位或失败样本。

图中每列是一个 target，每行依次为 GT、perceptual-only、semantic-only、joint，并在列下方显示 generated prompt。导出 300 dpi PNG 和矢量 PDF，同时保存 `selected_samples.csv` 和选择规则。

## 9. 结果验收与汇报格式

新增：

```text
artifacts/supplement_experiments/RESULTS_SUMMARY.md
```

该文件必须包含：

1. 运行环境：GPU、Python、PyTorch、CUDA、commit SHA；
2. 使用的数据 split、trial 数和 unique target 数；
3. 所有 checkpoint 路径及 SHA256；
4. 每个实验的完整命令；
5. P0 专家相似度与路由结果；
6. P0 专家切除结果和 paired CI；
7. P1 是否执行，若未执行则列出准确阻塞项；
8. 任何与论文现有数字不一致之处；
9. 明确区分“重新训练消融”和“推理时敏感性分析”；
10. 哪些图表可以进入正文、哪些只能进入补充材料。

完成后执行：

```bash
git status --short
git diff --check
python -m compileall experiments
```

只提交源代码、配置、说明文档和小型结果摘要。不要提交数据集、模型、缓存或批量生成图像。提交前向用户报告准备提交的文件和关键结果，未经用户明确要求不要自行改写论文数字。

## 10. 停止条件

遇到以下任一情况，保留日志并向用户汇报，不要猜测或伪造：

- checkpoint 与模型结构不匹配；
- test split 或 target_id 映射不确定；
- generated prompt 实际没有使用训练后的 EEG projector；
- EEG image projector 是随机初始化；
- ground-truth 图像与 target_id 无法可靠对齐；
- 原论文 SA 分类器或类别映射缺失；
- 路由权重退化为近似常数；
- 专家功能差异与论文命名不一致；
- 运行结果无法复现。

负结果本身可以用于收缩论文主张；不要通过改 seed、挑样本或更换指标隐藏负结果。
