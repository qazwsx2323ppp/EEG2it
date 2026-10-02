# DVSU P0 补充实验结果

资源检查、P0-A 全量分析和 P0-B 全量 **inference-time masking sensitivity analysis** 已完成。未重新训练，未修改论文数字，未执行 P1/P2，未下载或替换模型权重。

## 运行环境与数据

- 分支：`dvsu-revision`；基线 commit：`794a61eacd31fa782ce89829a186e0d88a7b0cde`（工作树含本次新增实验代码）。
- Python 3.13.5；PyTorch 2.6.0+cu124；CUDA runtime 12.4。
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU（8 GB）；NVIDIA 驱动 577.03，驱动显示支持 CUDA 12.9。
- 解释器：`D:\CODE\EEG\EEG2it\.venv\Scripts\python.exe`；默认系统 Python 未安装 PyTorch。
- 固定 test split 0：1,997 个 trial，333 个 unique target；无过滤、无目标跨 split 重叠。
- 预处理复用 TripletDataset：20–460 ms 裁剪，线性插值到 512；无推理时数据增强。
- Windows 使用 `num_workers=0`，避免将约 3.1 GB EEG 数据复制到各 spawn worker。
- 所有 CI：seed=42，1,000 次 trial-level percentile bootstrap；差值为配对 bootstrap。同一目标的重复 trial 并非独立目标，因此这些 CI 不代表跨目标/跨被试泛化置信区间。
- 已发现 image target 第 180 行全零（`n03452741_17620`），仅存在于 train 的 6 个 trial；val/test 均不含该目标。未修复、替换或删除任何数据。

## 资源路径与 checkpoint

| 资源 | 绝对路径 | 存在 |
| --- | --- | --- |
| EEG_DATA_PATH | `D:\CODE\EEG\EEG2it\data\EEG_data\eeg_55_95_std.pth` | True |
| IMAGE_VEC_PATH | `D:\CODE\EEG\EEG2it\data\image_vectors_aligned.npy` | True |
| TEXT_VEC_PATH | `D:\CODE\EEG\EEG2it\data\text_vectors_aligned.npy` | True |
| SPLITS_PATH | `D:\CODE\EEG\EEG2it\data\EEG_data\block_splits_by_image_all.pth` | True |
| EEG_ENCODER_CKPT | `D:\CODE\EEG\EEG2it\temp\best_12.8_change.pth` | True |
| QWEN_PROJECTOR_CKPT | `D:\CODE\EEG\EEG2it\temp\best_eeg_projector.pth` | False |
| EEG_IMG_PROJ_CKPT | `D:\CODE\EEG\EEG2it\temp\eeg_img_proj_ckpt.pth` | False |
| QWEN_MODEL_DIR | `D:\CODE\EEG\EEG2it\temp\Qwen2.5-Omni-3B` | False |
| SD_MODEL_DIR | `D:\CODE\EEG\EEG2it\temp\sd15-diffusers` | False |
| CLIP_MODEL_DIR | `C:\Users\HP\.cache\huggingface\hub\models--openai--clip-vit-base-patch32\snapshots\3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` | True |
| CAPTIONS_DIR | `D:\CODE\EEG\EEG2it\data\text_data` | True |
| IMAGE_ROOT | `D:\CODE\EEG\EEG2it\data\image_data` | True |

用于 P0 的 `temp/best_12.8_change.pth`：所有当前 backbone、expert heads、router 参数名称/形状完整。原始 missing keys 为 `[]`；unexpected keys 为 `[idx_sem, idx_vis]`。这两个旧通道索引 buffer 在当前 forward 中不使用，检查器明确记录并移除后，所有模型参数通过 `strict=True` 加载。未重新映射任何 learned key，也未使用随机参数。

该权重的独立训练配置/选择记录不完整，不能确认它属于 IDE 当前打开的 softmax text-only 配置，也不能宣称它是论文最终版本的 checkpoint。本报告只描述该本地权重在当前代码下的行为。

| 检查到的 encoder checkpoint | SHA256 | 可用于当前模型 |
| --- | --- | --- |
| `D:\CODE\EEG\EEG2it\temp\best_12.8_change.pth` | `bceac99e35a0b08a095985dfcc8bb72509e269e416e2978936876ffa7ed6461a` | True |
| `D:\CODE\EEG\EEG2it\wandb\run-20251206_150619-nh2xfiqk\files\best_eeg_encoder.pth` | `973c7d26b2f9a044931082ea90776b150e3b0795467d72351d886ab47b38dea6` | False |
| `D:\CODE\EEG\EEG2it\wandb\run-20251206_155546-ny01iyrx\files\best_eeg_encoder.pth` | `e200d1d4039280c3fdc2b52da8b8695e9279a293d20f2ad295ece41bc1d00255` | False |

另外两个历史 checkpoint 的专家末层为 `.2`，当前实现需要 `.3`，各缺少 6 个核心参数；已拒绝使用，未自动重命名。约 5.6 GB 的 DreamDiffusion 生成预训练 checkpoint 不参与本次加载。

test 中 333/333 个目标均可通过 `EEG images[target_id]` 对齐到现有描述文件及刺激图像。P0 使用仓库提供的 aligned CLIP 向量，不重新编码；没有独立的原始向量提取清单可核对其生成历史。

## P0-A 专家与路由

下表为余弦相似度 mean [95% CI]，专家和目标均 L2 归一化。

| Expert | CLIP image target | CLIP text target |
| --- | --- | --- |
| Visual | 0.1253 [0.1219, 0.1286] | 0.0114 [0.0097, 0.0132] |
| Shared | 0.1125 [0.1093, 0.1161] | 0.0065 [0.0056, 0.0073] |
| Semantic | -0.0010 [-0.0028, 0.0007] | 0.0600 [0.0583, 0.0616] |

| 配对目标偏好 | Mean [95% CI] |
| --- | --- |
| visual_image_minus_text | 0.1139 [0.1100, 0.1177] |
| semantic_text_minus_image | 0.0610 [0.0584, 0.0635] |

Visual 的图像偏好、Semantic 的文本偏好均与命名一致。Shared 的相似度主要偏向图像目标，仅凭这些结果不能称其为平衡的双任务专家。跨模态目标空间的难度/几何不同，相似度对比只是功能证据之一。

| 逐 trial gate | Mean | SD | Min | Max |
| --- | --- | --- | --- | --- |
| gate_visual_img | 0.494103 | 0.004120 | 0.478569 | 0.508844 |
| gate_shared_img | 0.505896 | 0.004120 | 0.491155 | 0.521430 |
| gate_semantic_txt | 0.926286 | 0.002414 | 0.916395 | 0.934245 |
| gate_shared_txt | 0.073713 | 0.002414 | 0.065753 | 0.083604 |

全量分布未触发预先固定的 SD < 0.001 近常数阈值，但波动仅为约 0.0024–0.0041；image 分支几乎均匀，text 分支主要采用 Semantic。不能将此描述为强输入自适应。首次 32 个 trial 的 SD < 0.001；该冒烟子集并不代表全量分布，结论使用固定 split 的所有 trial。

## P0-B 推理时敏感性分析

这是同一冻结 checkpoint 的推理时输出切除与路由替换，**不是重新训练的 architecture ablation**。`kill_shared` 对应 `kill_fusion`；切除后保留 learned gates；uniform 使用相同专家的等权混合。复用一次 backbone/head 输出的代数重组已逐条件与实际 forward 验证一致。

候选库严格限定为 test split 的 333 个 unique target_id；每个 trial 为一个 query。并列分数按升序 target_id 打破；negative cosine 为除正确目标外全部 332 个候选的平均值。下表 R@K 为百分比，cosine 为原始单位。

| Condition | Branch | R@1 (%) | R@5 (%) | Positive cosine | Negative cosine |
| --- | --- | --- | --- | --- | --- |
| full_moe | image | 2.7041 | 13.2699 | 0.143940 | 0.032142 |
| full_moe | text | 2.1032 | 11.6174 | 0.060101 | 0.009450 |
| kill_visual | image | 2.2033 | 11.7176 | 0.112517 | 0.021704 |
| kill_visual | text | 2.1032 | 11.6174 | 0.060101 | 0.009450 |
| kill_semantic | image | 2.7041 | 13.2699 | 0.143940 | 0.032142 |
| kill_semantic | text | 0.9514 | 4.2564 | 0.006467 | 0.003050 |
| kill_shared | image | 1.9029 | 12.3686 | 0.125277 | 0.030603 |
| kill_shared | text | 2.2033 | 11.6174 | 0.059989 | 0.009334 |
| uniform_router | image | 2.7041 | 13.2198 | 0.144035 | 0.032209 |
| uniform_router | text | 2.3035 | 11.2669 | 0.056644 | 0.009692 |

差值方向统一为 **full_moe − condition**，正值表示切除条件较差；R@K 差值的单位为百分点，cosine 差值为原始单位。全部列出 mean [paired 95% CI]。

| Condition | Branch | ΔR@1 (pp) | ΔR@5 (pp) | Δpositive cosine | Δnegative cosine |
| --- | --- | --- | --- | --- | --- |
| kill_visual | image | 0.5008 [-0.1002, 1.1517] | 1.5523 [0.3505, 2.6565] | 0.0314 [0.0299, 0.0329] | 0.0104 [0.0101, 0.0108] |
| kill_visual | text | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| kill_semantic | image | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| kill_semantic | text | 1.1517 [0.4507, 1.8528] | 7.3610 [5.9577, 8.9647] | 0.0536 [0.0519, 0.0555] | 0.0064 [0.0056, 0.0073] |
| kill_shared | image | 0.8012 [0.1502, 1.4522] | 0.9014 [-0.3518, 2.2033] | 0.0187 [0.0171, 0.0201] | 0.0015 [0.0013, 0.0018] |
| kill_shared | text | -0.1002 [-0.5008, 0.3005] | 0.0000 [-0.5020, 0.5508] | 0.0001 [0.0001, 0.0001] | 0.0001 [0.0001, 0.0001] |
| uniform_router | image | 0.0000 [-0.1502, 0.1502] | 0.0501 [0.0000, 0.1502] | -0.0001 [-0.0001, -0.0001] | -0.0001 [-0.0001, -0.0001] |
| uniform_router | text | -0.2003 [-0.6510, 0.2504] | 0.3505 [-0.4006, 1.0516] | 0.0035 [0.0031, 0.0039] | -0.0002 [-0.0006, 0.0001] |

切除 Visual 的 image R@5 下降 1.5523 个百分点，切除 Semantic 的 text R@5 下降 7.3610 个百分点。Visual 不直接进入 text 分支、Semantic 不直接进入 image 分支，因此相反分支的零变化是结构决定的，不能单独作为学习出任务选择性的证据。

切除 Shared 对 text R@1/R@5 的 CI 含零，不能支持 Shared 对两个分支均有明确检索贡献。learned router 相对 uniform 未表现出稳定的双分支检索优势，不能宣称输入相关路由带来普遍收益。

## 验证与可复现性

- 两项实验均先运行固定的前 32 个 trial；smoke 中候选仍为完整 test 的 333 个目标。
- P0-A 32-trial 重跑：逐样本相似度、bootstrap 汇总、逐样本 gates 三个 CSV 逐字节一致。
- 基线 forward 与当前默认 forward 逐元素相等；详细输出不改变两个最终 embedding。
- 手算检索 fixture、重复 query、负相似度及零配对差值 CI 检查通过。
- 全量保存 1997 行专家相似度、1997 行 gates、19970 行 masking 查询结果；汇总与原始行一致。
- 初次 smoke 因 TorchVersion 的 YAML 序列化错误在推理前中止；已转为普通字符串后重跑成功，失败日志保留在 smoke/run.log；未因此更换数据、seed 或 checkpoint。

## 完整命令

所有命令从仓库根目录执行；结果目录已有内容时拒绝覆盖。

```powershell
.\.venv\Scripts\python.exe -m experiments.check_resources --config configs/triplet_config.yaml --encoder-ckpt temp/best_12.8_change.pth
D:\CODE\EEG\EEG2it\.venv\Scripts\python.exe -m experiments.analyze_sere_experts --config configs/triplet_config.yaml --encoder-ckpt temp/best_12.8_change.pth --split test --split-index 0 --output-dir artifacts/supplement_experiments/sere_expert_analysis_smoke --batch-size 32 --num-workers 0 --device cuda --seed 42 --bootstrap-iters 1000 --max-samples 32
D:\CODE\EEG\EEG2it\.venv\Scripts\python.exe -m experiments.evaluate_sere_masking --config configs/triplet_config.yaml --encoder-ckpt temp/best_12.8_change.pth --split test --split-index 0 --output-dir artifacts/supplement_experiments/sere_masking_smoke --batch-size 32 --num-workers 0 --device cuda --seed 42 --bootstrap-iters 1000 --max-samples 32
D:\CODE\EEG\EEG2it\.venv\Scripts\python.exe -m experiments.analyze_sere_experts --config configs/triplet_config.yaml --encoder-ckpt temp/best_12.8_change.pth --split test --split-index 0 --output-dir artifacts/supplement_experiments/sere_expert_analysis --batch-size 64 --num-workers 0 --device cuda --seed 42 --bootstrap-iters 1000
D:\CODE\EEG\EEG2it\.venv\Scripts\python.exe -m experiments.evaluate_sere_masking --config configs/triplet_config.yaml --encoder-ckpt temp/best_12.8_change.pth --split test --split-index 0 --output-dir artifacts/supplement_experiments/sere_masking --batch-size 64 --num-workers 0 --device cuda --seed 42 --bootstrap-iters 1000
D:\CODE\EEG\EEG2it\.venv\Scripts\python.exe -m experiments.analyze_sere_experts --config configs/triplet_config.yaml --encoder-ckpt temp/best_12.8_change.pth --split test --split-index 0 --output-dir artifacts/supplement_experiments/sere_expert_analysis_reproducibility --batch-size 32 --num-workers 0 --device cuda --seed 42 --bootstrap-iters 1000 --max-samples 32
.\.venv\Scripts\python.exe -m experiments.summarize_sere_results
```

## P1 状态、论文数字与图表建议

P1/P2 未执行；本次用户要求仅资源检查和 P0。P1 同时存在以下实际资源缺口：

- Missing QWEN_PROJECTOR_CKPT: D:\CODE\EEG\EEG2it\temp\best_eeg_projector.pth
- Missing EEG_IMG_PROJ_CKPT: D:\CODE\EEG\EEG2it\temp\eeg_img_proj_ckpt.pth
- Missing QWEN_MODEL_DIR: D:\CODE\EEG\EEG2it\temp\Qwen2.5-Omni-3B
- Missing SD_MODEL_DIR: D:\CODE\EEG\EEG2it\temp\sd15-diffusers
- Python dependency diffusers is not installed in the selected interpreter

这些缺口已阻止严格加载和端到端 smoke；没有使用随机 projector 或固定 prompt 冒充 DPG。CLIP 本地 snapshot 存在，但没有为本次 P0 加载 CLIP 模型；其完整加载能力及原论文 SA 分类器/类别映射尚未验证。

仓库未包含论文 `.tex` 或指定版本的论文指标表，无法逐项核对论文现有数字；本次 333-target 检索协议也不应与 batch 内检索或未去重候选的旧指标直接比较。不得据此自动替换论文数字。

专家相似度热图及准确标为推理时敏感性分析的表格可作为当前 checkpoint 的正文候选证据；正式采用前须确认其与论文 checkpoint 的身份一致。路由分布建议进入补充材料并保留低变异结论。这次结果不支持强输入自适应、Shared 稳定贡献双分支或 DPG 互补性等更强主张。

图表输出为白底 300 dpi PNG 与矢量 PDF，位于 `sere_expert_analysis/`；CSV、JSON、run_config 和日志均已保留。代码放在 `experiments/`；没有提交数据、权重、缓存、批量图像或修改论文。
