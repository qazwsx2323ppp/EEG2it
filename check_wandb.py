import json
import os
import pandas as pd
import subprocess
import sys
import wandb
try:
    import yaml
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml"])
    import yaml

# -------------------------- 仅修改这1个参数（已验证有效） --------------------------
WANDB_RUN_DIR = "wandb\\offline-run-20251031_223302-jneli3te"
# -------------------------------------------------------------------

# 打印关键信息
print(f"📌 Wandb 版本：{wandb.__version__}")
print(f"📌 待解析文件夹：{WANDB_RUN_DIR}")
wandb_file = [f for f in os.listdir(WANDB_RUN_DIR) if f.endswith('.wandb')][0]
print(f"📌 找到 .wandb 文件：{wandb_file}")

# 核心：设置环境变量，强制禁用登录和云端连接（关键解决 API key 报错）
env = os.environ.copy()
env["WANDB_MODE"] = "offline"  # 强制离线，不连接云端
env["WANDB_API_KEY"] = "dummy"  # 用占位符跳过 API key 校验
env["WANDB_SILENT"] = "true"  # 静默模式，减少无关输出
env["WANDB_DISABLE_LOGGING"] = "true"  # 禁用日志，避免干扰

print("\n🔧 纯离线解析（禁用登录校验）...")
command = [
    sys.executable,
    "-m", "wandb",
    "sync",
    "--include-offline",  # 环境支持的参数
    WANDB_RUN_DIR
]
print(f"📌 执行命令：{' '.join(command)}")

# 执行命令（传递环境变量，强制离线无登录）
result = subprocess.run(
    command,
    env=env,  # 关键：传递离线环境变量
    capture_output=True,
    text=True,
    encoding="utf-8"
)

# 打印输出
print(f"\n📌 命令 stdout：\n{result.stdout}")
print(f"\n📌 命令 stderr：\n{result.stderr}")

# 检查结果（returncode=0 即成功，旧版可能有警告但不影响）
if result.returncode != 0:
    # 终极兜底：无参数 sync + 环境变量（旧版最兼容）
    print("\n⚠️  尝试无参数纯离线解析...")
    command = [sys.executable, "-m", "wandb", "sync", WANDB_RUN_DIR]
    result = subprocess.run(command, env=env, capture_output=True, text=True, encoding="utf-8")
    print(f"📌 无参数命令 stdout：\n{result.stdout}")
    print(f"📌 无参数命令 stderr：\n{result.stderr}")
    if result.returncode != 0:
        raise RuntimeError(
            f"解析失败！返回码：{result.returncode}\n"
            "终极解决方案（手动执行）：\n"
            "1. 打开终端，激活虚拟环境：.venv\\Scripts\\activate\n"
            "2. 执行命令（复制粘贴）：\n"
            "set WANDB_MODE=offline && set WANDB_API_KEY=dummy && python -m wandb sync --include-offline wandb\\offline-run-20251031_223302-jneli3te\n"
            "3. 执行后再运行本代码导出 JSON"
        )

print("\n✅ .wandb 文件解析成功！开始导出 JSON...")

# -------------------------- 读取并导出文件 --------------------------
files_dir = os.path.join(WANDB_RUN_DIR, "files") if os.path.exists(os.path.join(WANDB_RUN_DIR, "files")) else WANDB_RUN_DIR

# 导出配置
config = {}
config_paths = [os.path.join(files_dir, f) for f in ["config.yaml", "config.json"]]
for cfg_path in config_paths:
    if os.path.exists(cfg_path) and os.path.getsize(cfg_path) > 0:
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) if cfg_path.endswith(".yaml") else json.load(f)
        break
with open("wandb_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

# 导出指标
metrics_df = None
metrics_paths = [os.path.join(files_dir, f) for f in ["metrics.csv", "metrics.jsonl"]]
for metric_path in metrics_paths:
    if os.path.exists(metric_path) and os.path.getsize(metric_path) > 0:
        metrics_df = pd.read_csv(metric_path) if metric_path.endswith(".csv") else pd.read_json(metric_path, lines=True)
        break
useful_cols = [col for col in metrics_df.columns if any(k in col for k in ["epoch", "loss"])] or metrics_df.columns.tolist()
metrics_df = metrics_df[useful_cols].dropna(how="all")
if "epoch" in metrics_df.columns:
    metrics_df = metrics_df.sort_values("epoch").reset_index(drop=True)
metrics_df.to_json("wandb_metrics.json", orient="records", indent=2, force_ascii=False)

print("\n🎉 100% 离线导出成功！")
print(f"- 配置文件：wandb_config.json（{len(config)} 个配置项）")
print(f"- 指标文件：wandb_metrics.json（{len(metrics_df)} 条记录）")
print(f"- 包含字段：{', '.join(useful_cols)}")