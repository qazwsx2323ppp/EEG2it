# DPG 资源缺口

本次仅执行资源检查与 P0；P1/P2 未运行。

- Missing QWEN_PROJECTOR_CKPT: D:\CODE\EEG\EEG2it\temp\best_eeg_projector.pth
- Missing EEG_IMG_PROJ_CKPT: D:\CODE\EEG\EEG2it\temp\eeg_img_proj_ckpt.pth
- Missing QWEN_MODEL_DIR: D:\CODE\EEG\EEG2it\temp\Qwen2.5-Omni-3B
- Missing SD_MODEL_DIR: D:\CODE\EEG\EEG2it\temp\sd15-diffusers
- Python dependency diffusers is not installed in the selected interpreter

以上检查只确认资源与路径；未完成 Qwen/SD/projector 严格加载或 3 样本端到端 smoke。原论文 SA 分类器和类别映射尚未验证。不得使用随机权重、oracle 标签或固定 prompt 替代完整 DPG。
