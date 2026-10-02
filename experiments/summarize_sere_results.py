"""Build the Markdown report for the audited local 2026-10-02 P0 run."""
import argparse
import csv
import json
from pathlib import Path

from omegaconf import OmegaConf

from experiments.sere_common import CONDITIONS, GATES, METRICS


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def interval(value, mean_key='mean', scale=1):
    return f"{value[mean_key]*scale:.4f} [{value['ci_low']*scale:.4f}, {value['ci_high']*scale:.4f}]"


def table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |',
            *['| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows], '']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root', default='artifacts/supplement_experiments')
    args = p.parse_args()
    root = Path(args.results_root).resolve()
    expert = read_json(root / 'sere_expert_analysis/summary.json')
    masking = read_json(root / 'sere_masking/masking_metrics.json')
    resources = read_json(root / 'resource_check/resources.json')
    n = expert['dataset']['evaluated_trials']
    # Narrative interpretation below belongs to this documented checkpoint/protocol.
    # Refuse to apply that interpretation to a different future experiment.
    assert expert['checkpoint']['sha256'] == 'bceac99e35a0b08a095985dfcc8bb72509e269e416e2978936876ffa7ed6461a'
    assert expert['dataset']['split'] == 'test' and expert['dataset']['split_index'] == 0 and n == 1997
    assert expert['bootstrap']['seed'] == 42 and expert['bootstrap']['iterations'] == 1000
    assert masking['dataset']['evaluated_trials'] == n == expert['dataset']['trials']
    assert expert['checkpoint']['sha256'] == masking['checkpoint']['sha256']
    for name in ('split', 'split_index', 'unique_targets'):
        assert expert['dataset'][name] == masking['dataset'][name]
    similarity_rows = read_csv(root / 'sere_expert_analysis/expert_target_similarity_per_sample.csv')
    gate_rows = read_csv(root / 'sere_expert_analysis/router_weights_per_sample.csv')
    query_rows = read_csv(root / 'sere_masking/masking_per_sample.csv')
    assert len(similarity_rows) == len(gate_rows) == n and len(query_rows) == n * 10
    identities = [(row['dataset_index'], row['eeg_index'], row['target_id']) for row in similarity_rows]
    assert len(set(identities)) == n
    assert identities == [(r['dataset_index'], r['eeg_index'], r['target_id']) for r in gate_rows]
    for condition in CONDITIONS:
        for branch in ('image', 'text'):
            subset = [r for r in query_rows if r['condition'] == condition and r['branch'] == branch]
            assert identities == [(r['dataset_index'], r['eeg_index'], r['target_id']) for r in subset]
            metric = next(m for m in masking['metrics'] if m['condition'] == condition and m['branch'] == branch)
            for key in METRICS:
                mean = sum(float(r[key]) for r in subset) / n
                assert abs(mean - metric[key]) < 1e-10
    # Independently rerun the 32-trial smoke analysis with identical settings.
    reproducibility = {}
    for name in ('expert_target_similarity_per_sample.csv', 'expert_target_similarity_summary.csv',
                 'router_weights_per_sample.csv'):
        original = root / 'sere_expert_analysis_smoke' / name
        repeat = root / 'sere_expert_analysis_reproducibility' / name
        reproducibility[name] = original.read_bytes() == repeat.read_bytes()
    assert all(reproducibility.values()), 'Repeated smoke results are not byte-identical'
    env = expert['environment']
    lines = ['# DVSU P0 补充实验结果', '',
             '资源检查、P0-A 全量分析和 P0-B 全量 **inference-time masking sensitivity analysis** 已完成。'
             '未重新训练，未修改论文数字，未执行 P1/P2，未下载或替换模型权重。', '',
             '## 运行环境与数据', '',
             f"- 分支：`{env['branch']}`；基线 commit：`{env['commit_sha']}`（工作树含本次新增实验代码）。",
             f"- Python {env['python']}；PyTorch {env['pytorch']}；CUDA runtime {env['cuda_runtime']}。",
             f"- GPU：{env['gpu']}（8 GB）；NVIDIA 驱动 577.03，驱动显示支持 CUDA 12.9。",
             f"- 解释器：`{env['executable']}`；默认系统 Python 未安装 PyTorch。",
             f"- 固定 test split {expert['dataset']['split_index']}：{n:,} 个 trial，"
             f"{expert['dataset']['unique_targets']} 个 unique target；无过滤、无目标跨 split 重叠。",
             '- 预处理复用 TripletDataset：20–460 ms 裁剪，线性插值到 512；无推理时数据增强。',
             '- Windows 使用 `num_workers=0`，避免将约 3.1 GB EEG 数据复制到各 spawn worker。',
             '- 所有 CI：seed=42，1,000 次 trial-level percentile bootstrap；差值为配对 bootstrap。'
             '同一目标的重复 trial 并非独立目标，因此这些 CI 不代表跨目标/跨被试泛化置信区间。',
             '- 已发现 image target 第 180 行全零（`n03452741_17620`），仅存在于 train 的 6 个 trial；'
             'val/test 均不含该目标。未修复、替换或删除任何数据。', '', '## 资源路径与 checkpoint', '']
    lines += table(['资源', '绝对路径', '存在'],
                   [(k, '`' + v['path'] + '`', str(v['exists'])) for k, v in resources['resources'].items()])
    lines += ['用于 P0 的 `temp/best_12.8_change.pth`：所有当前 backbone、expert heads、router 参数'
              '名称/形状完整。原始 missing keys 为 `[]`；unexpected keys 为 `[idx_sem, idx_vis]`。'
              '这两个旧通道索引 buffer 在当前 forward 中不使用，检查器明确记录并移除后，所有模型参数通过 `strict=True` 加载。'
              '未重新映射任何 learned key，也未使用随机参数。', '',
              '该权重的独立训练配置/选择记录不完整，不能确认它属于 IDE 当前打开的 softmax text-only 配置，'
              '也不能宣称它是论文最终版本的 checkpoint。本报告只描述该本地权重在当前代码下的行为。', '']
    lines += table(['检查到的 encoder checkpoint', 'SHA256', '可用于当前模型'],
                   [('`' + a['path'] + '`', '`' + a.get('sha256', 'not available') + '`',
                     str(a.get('compatible', False))) for a in resources['checkpoint_audits']])
    lines += ['另外两个历史 checkpoint 的专家末层为 `.2`，当前实现需要 `.3`，各缺少 6 个核心参数；'
              '已拒绝使用，未自动重命名。约 5.6 GB 的 DreamDiffusion 生成预训练 checkpoint 不参与本次加载。', '',
              'test 中 333/333 个目标均可通过 `EEG images[target_id]` 对齐到现有描述文件及刺激图像。'
              'P0 使用仓库提供的 aligned CLIP 向量，不重新编码；没有独立的原始向量提取清单可核对其生成历史。', '',
              '## P0-A 专家与路由', '', '下表为余弦相似度 mean [95% CI]，专家和目标均 L2 归一化。', '']
    similarity = {(r['expert'], r['target']): r for r in expert['similarities']}
    lines += table(['Expert', 'CLIP image target', 'CLIP text target'],
                   [(name.title(), interval(similarity[name, 'image']), interval(similarity[name, 'text']))
                    for name in ('visual', 'shared', 'semantic')])
    lines += table(['配对目标偏好', 'Mean [95% CI]'],
                   [(key, interval(value)) for key, value in expert['specialization'].items()])
    lines += ['Visual 的图像偏好、Semantic 的文本偏好均与命名一致。Shared 的相似度主要偏向图像目标，'
              '仅凭这些结果不能称其为平衡的双任务专家。跨模态目标空间的难度/几何不同，相似度对比只是功能证据之一。', '']
    lines += table(['逐 trial gate', 'Mean', 'SD', 'Min', 'Max'],
                   [(key, *(f"{value[field]:.6f}" for field in ('mean', 'std', 'min', 'max')))
                    for key, value in expert['router'].items()])
    lines += ['全量分布未触发预先固定的 SD < 0.001 近常数阈值，但波动仅为约 0.0024–0.0041；'
              'image 分支几乎均匀，text 分支主要采用 Semantic。不能将此描述为强输入自适应。'
              '首次 32 个 trial 的 SD < 0.001；该冒烟子集并不代表全量分布，结论使用固定 split 的所有 trial。', '',
              '## P0-B 推理时敏感性分析', '',
              '这是同一冻结 checkpoint 的推理时输出切除与路由替换，**不是重新训练的 architecture ablation**。'
              '`kill_shared` 对应 `kill_fusion`；切除后保留 learned gates；uniform 使用相同专家的等权混合。'
              '复用一次 backbone/head 输出的代数重组已逐条件与实际 forward 验证一致。', '',
              '候选库严格限定为 test split 的 333 个 unique target_id；每个 trial 为一个 query。'
              '并列分数按升序 target_id 打破；negative cosine 为除正确目标外全部 332 个候选的平均值。'
              '下表 R@K 为百分比，cosine 为原始单位。', '']
    lines += table(['Condition', 'Branch', 'R@1 (%)', 'R@5 (%)', 'Positive cosine', 'Negative cosine'],
                   [(m['condition'], m['branch'], f"{100*m['r_at_1']:.4f}", f"{100*m['r_at_5']:.4f}",
                     f"{m['positive_cosine']:.6f}", f"{m['negative_cosine']:.6f}") for m in masking['metrics']])
    lines += ['差值方向统一为 **full_moe − condition**，正值表示切除条件较差；'
              'R@K 差值的单位为百分点，cosine 差值为原始单位。全部列出 mean [paired 95% CI]。', '']
    delta = {(d['condition'], d['branch'], d['metric']): d for d in masking['paired_deltas']}
    lines += table(['Condition', 'Branch', 'ΔR@1 (pp)', 'ΔR@5 (pp)', 'Δpositive cosine', 'Δnegative cosine'],
                   [(condition, branch,
                     *[interval(delta[condition, branch, metric], 'mean_delta', 100 if metric.startswith('r_at') else 1)
                       for metric in METRICS]) for condition in CONDITIONS[1:] for branch in ('image', 'text')])
    visual = delta['kill_visual', 'image', 'r_at_5']
    semantic = delta['kill_semantic', 'text', 'r_at_5']
    lines += [f"切除 Visual 的 image R@5 下降 {100*visual['mean_delta']:.4f} 个百分点，"
              f"切除 Semantic 的 text R@5 下降 {100*semantic['mean_delta']:.4f} 个百分点。"
              'Visual 不直接进入 text 分支、Semantic 不直接进入 image 分支，因此相反分支的零变化是结构决定的，'
              '不能单独作为学习出任务选择性的证据。', '',
              '切除 Shared 对 text R@1/R@5 的 CI 含零，不能支持 Shared 对两个分支均有明确检索贡献。'
              'learned router 相对 uniform 未表现出稳定的双分支检索优势，不能宣称输入相关路由带来普遍收益。', '',
              '## 验证与可复现性', '',
              '- 两项实验均先运行固定的前 32 个 trial；smoke 中候选仍为完整 test 的 333 个目标。',
              '- P0-A 32-trial 重跑：逐样本相似度、bootstrap 汇总、逐样本 gates 三个 CSV 逐字节一致。',
              '- 基线 forward 与当前默认 forward 逐元素相等；详细输出不改变两个最终 embedding。',
              '- 手算检索 fixture、重复 query、负相似度及零配对差值 CI 检查通过。',
              f'- 全量保存 {n} 行专家相似度、{n} 行 gates、{n*10} 行 masking 查询结果；汇总与原始行一致。',
              '- 初次 smoke 因 TorchVersion 的 YAML 序列化错误在推理前中止；已转为普通字符串后重跑成功，'
              '失败日志保留在 smoke/run.log；未因此更换数据、seed 或 checkpoint。', '',
              '## 完整命令', '', '所有命令从仓库根目录执行；结果目录已有内容时拒绝覆盖。', '',
              '```powershell',
              r'.\.venv\Scripts\python.exe -m experiments.check_resources --config configs/triplet_config.yaml --encoder-ckpt temp/best_12.8_change.pth']
    for name in ('sere_expert_analysis_smoke', 'sere_masking_smoke', 'sere_expert_analysis',
                 'sere_masking', 'sere_expert_analysis_reproducibility'):
        lines.append(str(OmegaConf.load(root / name / 'run_config.yaml').command))
    lines += [r'.\.venv\Scripts\python.exe -m experiments.summarize_sere_results', '```', '',
              '## P1 状态、论文数字与图表建议', '',
              'P1/P2 未执行；本次用户要求仅资源检查和 P0。P1 同时存在以下实际资源缺口：', '']
    lines += ['- ' + blocker for blocker in resources['p1_blockers']]
    lines += ['', '这些缺口已阻止严格加载和端到端 smoke；没有使用随机 projector 或固定 prompt 冒充 DPG。'
              'CLIP 本地 snapshot 存在，但没有为本次 P0 加载 CLIP 模型；其完整加载能力及原论文 SA 分类器/类别映射尚未验证。', '',
              '仓库未包含论文 `.tex` 或指定版本的论文指标表，无法逐项核对论文现有数字；本次 333-target 检索协议'
              '也不应与 batch 内检索或未去重候选的旧指标直接比较。不得据此自动替换论文数字。', '',
              '专家相似度热图及准确标为推理时敏感性分析的表格可作为当前 checkpoint 的正文候选证据；'
              '正式采用前须确认其与论文 checkpoint 的身份一致。路由分布建议进入补充材料并保留低变异结论。'
              '这次结果不支持强输入自适应、Shared 稳定贡献双分支或 DPG 互补性等更强主张。', '',
              '图表输出为白底 300 dpi PNG 与矢量 PDF，位于 `sere_expert_analysis/`；CSV、JSON、run_config 和日志均已保留。'
              '代码放在 `experiments/`；没有提交数据、权重、缓存、批量图像或修改论文。', '']
    (root / 'RESULTS_SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')
    blocked = ['# DPG 资源缺口', '', '本次仅执行资源检查与 P0；P1/P2 未运行。', '']
    blocked += ['- ' + blocker for blocker in resources['p1_blockers']]
    blocked += ['', '以上检查只确认资源与路径；未完成 Qwen/SD/projector 严格加载或 3 样本端到端 smoke。'
                '原论文 SA 分类器和类别映射尚未验证。不得使用随机权重、oracle 标签或固定 prompt 替代完整 DPG。', '']
    (root / 'BLOCKED_DPG.md').write_text('\n'.join(blocked), encoding='utf-8')
    print(root / 'RESULTS_SUMMARY.md')


if __name__ == '__main__':
    main()
