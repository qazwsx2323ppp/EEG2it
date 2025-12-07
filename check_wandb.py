import json
import os
import pandas as pd
import subprocess
import sys
import wandb
import glob

# ==========================================
# 1. 配置你的路径 (请指向包含 .wandb 文件的文件夹，而不是文件本身)
# 例如: D:\CODE\EEG\EEG2it\temp\wandb
WANDB_RUN_DIR = r"D:\CODE\EEG\EEG2it\temp\wandb"
# ==========================================

print(f"📌 Wandb 版本：{wandb.__version__}")
print(f"📌 目标文件夹：{WANDB_RUN_DIR}")

# 自动寻找 .wandb 文件
wandb_files = glob.glob(os.path.join(WANDB_RUN_DIR, "*.wandb"))
if not wandb_files:
    print("❌ 错误：在该目录下没找到任何 .wandb 文件！请检查路径。")
    sys.exit(1)

target_file = wandb_files[0]
print(f"📌 锁定目标文件：{target_file}")

# 设置离线环境变量
env = os.environ.copy()
env["WANDB_MODE"] = "offline"
env["WANDB_SILENT"] = "true"

print("\n🔧 正在尝试解析数据 (wandb sync)...")
# 使用 sync 命令将数据导出到当前目录
command = [sys.executable, "-m", "wandb", "sync", "--include-offline", WANDB_RUN_DIR]

try:
    result = subprocess.run(command, env=env, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        print("⚠️ sync 命令返回了错误代码，但这可能不影响数据生成。")
        print(f"错误输出: {result.stderr}")
except Exception as e:
    print(f"❌ 执行 sync 命令时发生异常: {e}")

print("\n✅ 同步尝试结束，开始寻找生成的 CSV 数据...")

# WandB sync 通常会在 WANDB_RUN_DIR 或者当前目录下生成 metrics.csv
# 我们遍历查找一下
search_paths = [
    os.path.join(WANDB_RUN_DIR, "metrics.csv"),
    "metrics.csv",  # 当前脚本目录
]
# 有时候 wandb 会生成在子文件夹里，递归找一下
for root, dirs, files in os.walk(WANDB_RUN_DIR):
    if "metrics.csv" in files:
        search_paths.append(os.path.join(root, "metrics.csv"))

metrics_df = None
found_csv = None

for csv_path in search_paths:
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        print(f"🎉 找到了指标文件：{csv_path}")
        try:
            metrics_df = pd.read_csv(csv_path)
            found_csv = csv_path
            break
        except Exception as e:
            print(f"❌ 读取 {csv_path} 失败: {e}")

# ==========================================
# 核心修复：增加空值检查，防止 AttributeError
# ==========================================
if metrics_df is not None:
    # 过滤列名
    cols = metrics_df.columns.tolist()
    useful_cols = [c for c in cols if any(k in c for k in ["epoch", "loss", "sim", "acc", "lr"])]
    
    # 如果没找到特定的列，就保留所有列
    if not useful_cols:
        useful_cols = cols
        
    final_df = metrics_df[useful_cols].dropna(how="all")
    
    # 保存结果
    output_file = "wandb_metrics_fixed.json"
    final_df.to_json(output_file, orient="records", indent=2, force_ascii=False)
    
    print(f"\n✅✅✅ 成功！数据已导出到: {output_file}")
    print(f"📊 包含字段: {useful_cols}")
    print(f"📄 总行数: {len(final_df)}")
else:
    print("\n❌❌❌ 失败：未能生成或读取到 metrics.csv。")
    print("可能原因：")
    print("1. 文件权限依然被锁（请执行第一步 taskkill）")
    print("2. .wandb 文件本身已损坏（无法解析）")
    print("3. 文件只读属性未取消")