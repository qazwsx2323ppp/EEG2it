import torch
from transformers import Qwen2_5OmniProcessor
from modeling_qwen2_5_omni_low_VRAM_mode import Qwen2_5OmniForConditionalGeneration
from models.clip_models import SpatialMoEEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    torch_dtype=torch.float16 if device=="cuda" else torch.float32,
    attn_implementation="sdpa"  # 如装了 flash-attn 可改为 "flash_attention_2"
).to(device).eval()

processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
tokenizer = processor.tokenizer
tokenizer.padding_side = "left"

# 初始化 EEG 编码器和线性投影
eeg_encoder = SpatialMoEEncoder(n_channels=128, n_samples=512, embedding_dim=512).to(device).eval()
thinker_hidden_size = model.config.thinker_config.hidden_size
eeg_projector = torch.nn.Linear(512, thinker_hidden_size).to(device)

# 挂到主模型
model.eeg_encoder = eeg_encoder
model.eeg_projector = eeg_projector

# 构造假 EEG 输入
eeg_input = torch.randn(1, 128, 512, device=device)

# 直接调用端到端接口
gen_ids = model.generate_from_eeg(
    eeg_input=eeg_input,
    tokenizer=tokenizer,
    prompt_text="请根据脑信号描述潜在图像场景。",
    max_new_tokens=32,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("生成文本：", text)
PY