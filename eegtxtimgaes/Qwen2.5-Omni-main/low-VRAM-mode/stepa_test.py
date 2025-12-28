import torch
import sys
import os
from omegaconf import OmegaConf

# Add parent directory to sys.path to allow importing dataset.py from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset import TripletDataset

from transformers import Qwen2_5OmniProcessor
from modeling_qwen2_5_omni_low_VRAM_mode import Qwen2_5OmniForConditionalGeneration
from models.clip_models import SpatialMoEEncoder
from out_qformer import EEGQFormer


USE_QFORMER = True  

# 引入图像生成器
try:
    from painter_sd import StableDiffusionPainter
    has_painter = True
except ImportError:
    print("Warning: diffusers not installed or painter_sd.py missing. Image generation disabled.")
    has_painter = False

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    attn_implementation="sdpa",
    device_map="auto",
    low_cpu_mem_usage=True
).eval()

processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
tokenizer = processor.tokenizer
tokenizer.padding_side = "left"

eeg_encoder = SpatialMoEEncoder(
    n_channels=128,
    n_samples=512,
    embedding_dim=512,
    pretrained_path=r"best_12.8_change.pth"
).to(device).eval()

thinker_hidden = model.config.thinker_config.text_config.hidden_size

if USE_QFORMER:
    print(">>> 正在初始化 Q-Former 投影层...")
    # 初始化 Q-Former
    # hidden_size: Qwen 的隐藏层维度 (通常 3584)
    # kv_dim: EEG Encoder 输出维度 (512)
    qformer = EEGQFormer(
        hidden_size=thinker_hidden, 
        kv_dim=512, 
        num_queries=16, # 查询向量数量，可以调整
        num_layers=2
    ).to(device)
    

    model.qformer = qformer
    model.use_qformer = True
    print(">>> Q-Former 已挂载 (随机初始化/需加载权重)。")
    

    # qformer_path = "eeg_qformer.pth"
    # if os.path.exists(qformer_path):
    #     model.qformer.load_state_dict(torch.load(qformer_path))
    #     print(f">>> 已加载 Q-Former 权重: {qformer_path}")

else:
    # 随机初始化线性层！
    # 提示词产生“幻觉”文本（hallucination）。用来拉通主流程而已
    eeg_projector = torch.nn.Linear(512, thinker_hidden).to(device)
    model.eeg_projector = eeg_projector
    model.use_qformer = False
    print(">>> 警告：eeg_projector (Linear) 是随机初始化的。")

model.eeg_encoder = eeg_encoder

print(">>> 正在进行 EEG 推理...")


try:
    print(">>> 正在加载数据集配置...")
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "triplet_config.yaml")
    cfg = OmegaConf.load(config_path)
    
    # 手动设置 root 路径，因为没有使用 hydra 启动
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg.data.root = project_root
    
    # 实例化数据集 (使用 test 模式)
    # 注意：确保数据文件已按照 configs/triplet_config.yaml 中的路径放置
    dataset = TripletDataset(cfg.data, mode='test', split_index=0)
    
    if len(dataset) > 0:
        # 获取第一个样本
        sample_idx = 0
        eeg_signal, _, _ = dataset[sample_idx]
        eeg_input = eeg_signal.unsqueeze(0).to(device) # (1, 128, 512)
        print(f">>> 已加载样本 {sample_idx}, EEG shape: {eeg_input.shape}")
    else:
        print(">>> 警告：数据集为空，回退到随机噪声。")
        eeg_input = torch.randn(1, 128, 512, device=device)

except Exception as e:
    print(f">>> 加载真实数据失败: {e}")
    print(">>> 回退到随机噪声输入。")
    # 假设输入是随机生成的 EEG 数据，实际应用时替换为真实数据
    eeg_input = torch.randn(1, 128, 512, device=device)  

gen_ids = model.generate_from_eeg(
    eeg_input=eeg_input,
    tokenizer=tokenizer,
    prompt_text="请根据脑信号描述潜在图像场景。",  # Prompt 可以引导模型
    max_new_tokens=64,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)
text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(f"\n[生成的文本]: {text}")

# --- 下游：生成图像 ---
if has_painter:
    print("\n>>> 正在根据文本生成图像...")
    try:
        # 初始化 SD 画家 (第一次运行会自动下载模型)
        painter = StableDiffusionPainter(device=device)
        
        # 使用生成的文本作为 Prompt
        prompt_to_use = text 
        
        # 调试：输出 prompt 确保其有效
        print(f"Using prompt: {prompt_to_use}")

        # 生成图像
        image = painter.generate(prompt_to_use)
        
        # 调试：检查生成的图像是否有效
        if image is None:
            print("图像生成失败，返回的图像为空！")
        else:
            # 确保保存目录存在
            import os
            save_dir = r"external"
            os.makedirs(save_dir, exist_ok=True)
            
            # 保存图像
            save_path = os.path.join(save_dir, "output_eeg_image.png")
            image.save(save_path)
            print(f">>> 图像已保存至: {save_path}")
    except Exception as e:
        print(f"图像生成失败: {e}")
