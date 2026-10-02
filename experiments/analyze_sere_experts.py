"""P0-A: trial-level expert/target similarity and learned router distributions."""
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.sere_common import (
    EXPERTS, GATES, bootstrap, collect, inspect_checkpoint, parser, run_checked,
    sample_identity, start_run, validate_dataset, write_csv, write_json,
)


def main():
    args = parser(__doc__, 'sere_expert_analysis').parse_args()
    cfg, device, output, provenance = start_run(args)
    if args.check_only:
        _, audit = inspect_checkpoint(args.encoder_ckpt, cfg)
        _, _, dataset_report = validate_dataset(cfg, args.split, args.split_index)
        write_json(output / 'resource_check.json', {'checkpoint': audit, 'dataset': dataset_report})
        if not audit['compatible']:
            raise RuntimeError('Incompatible encoder checkpoint')
        return
    data = collect(args, cfg, device)
    similarities = {}
    for expert in EXPERTS:
        emb = data['experts'][expert]
        emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True)
        for branch in ('image', 'text'):
            similarities[f'{expert}_{branch}'] = np.einsum('ij,ij->i', emb, data['targets'][branch])
    names = list(similarities)
    values = np.column_stack(list(similarities.values()))
    ci = bootstrap(values, args.bootstrap_iters, args.seed)
    rows = [{**sample_identity(data, i), **{k: float(v[i]) for k, v in similarities.items()}}
            for i in range(len(values))]
    summaries = [{'expert': k.rsplit('_', 1)[0], 'target': k.rsplit('_', 1)[1], 'n': len(values),
                  'mean': float(values[:, j].mean()), 'std': float(values[:, j].std(ddof=0)),
                  'ci_low': float(ci[0, j]), 'ci_high': float(ci[1, j])} for j, k in enumerate(names)]
    gates = [{**sample_identity(data, i), **{k: float(v[i]) for k, v in data['gates'].items()}}
             for i in range(len(values))]
    router_summary = {k: {'n': len(v), 'mean': float(v.mean()), 'std': float(v.std(ddof=0)),
                          'min': float(v.min()), 'max': float(v.max()),
                          'near_constant': bool(v.std(ddof=0) < 1e-3)} for k, v in data['gates'].items()}
    # A predeclared reporting threshold; never choose it after looking at the result.
    warnings = [f'{k}: near-constant routing (std < 0.001); do not claim strong input adaptation'
                for k, v in router_summary.items() if v['near_constant']]
    visual_preference = similarities['visual_image'] - similarities['visual_text']
    semantic_preference = similarities['semantic_text'] - similarities['semantic_image']
    preferences = np.column_stack([visual_preference, semantic_preference])
    preference_ci = bootstrap(preferences, args.bootstrap_iters, args.seed)
    specialization = {name: {'mean': float(preferences[:, i].mean()),
                             'ci_low': float(preference_ci[0, i]), 'ci_high': float(preference_ci[1, i])}
                      for i, name in enumerate(('visual_image_minus_text', 'semantic_text_minus_image'))}
    for name, value in specialization.items():
        if value['mean'] <= 0:
            warnings.append(f'{name}: functional target preference does not match the expert name')
    write_csv(output / 'expert_target_similarity_per_sample.csv', rows)
    write_csv(output / 'expert_target_similarity_summary.csv', summaries)
    write_csv(output / 'router_weights_per_sample.csv', gates)
    summary = {'dataset': data['dataset_report'], 'checkpoint': data['checkpoint'],
               'environment': provenance['environment'], 'similarities': summaries,
               'router': router_summary, 'specialization': specialization, 'warnings': warnings,
               'bootstrap': {'unit': 'EEG trial', 'iterations': args.bootstrap_iters, 'seed': args.seed,
                             'ci': '95% percentile; repeated trials are not independent targets'},
               'near_constant_threshold': 1e-3}
    write_json(output / 'summary.json', summary)
    plot_results(output, values, ci, data['gates'])
    for warning in warnings:
        logging.warning(warning)
    logging.info('P0-A finished: %s', output)


def plot_results(output, values, ci, gates):
    """Render stored numerical results without repeating encoder inference."""
    plt.rcParams.update({'font.size': 11, 'figure.facecolor': 'white', 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    matrix = values.mean(axis=0).reshape(3, 2)
    low, high = ci[0].reshape(3, 2), ci[1].reshape(3, 2)
    im = ax.imshow(matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    for i in range(3):
        for j in range(2):
            ax.text(j, i, f'{matrix[i,j]:.3f}\n[{low[i,j]:.3f}, {high[i,j]:.3f}]',
                    ha='center', va='center', fontsize=11)
    ax.set_xticks([0, 1], ['CLIP image target', 'CLIP text target'])
    ax.set_yticks(range(3), [s.title() for s in EXPERTS])
    ax.set_title(f'Expert-target cosine similarity (n={len(values)} trials)\nMean and 95% trial bootstrap CI')
    fig.colorbar(im, ax=ax, label='Cosine similarity')
    fig.tight_layout()
    fig.savefig(output / 'expert_target_similarity_heatmap.png', dpi=300)
    fig.savefig(output / 'expert_target_similarity_heatmap.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6), sharey=True)
    for ax, keys, labels, title in (
        (axes[0], GATES[:2], ['Visual', 'Shared'], 'Image route'),
        (axes[1], GATES[2:], ['Semantic', 'Shared'], 'Text route')):
        arrays = [gates[k] for k in keys]
        ax.boxplot(arrays, tick_labels=labels, showmeans=True)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f'{title} (n={len(values)} trials)')
        for j, v in enumerate(arrays):
            ax.text(j + 1, 0.84, f'mean={v.mean():.4f}\nSD={v.std():.4f}', ha='center', va='top')
        ax.grid(axis='y', alpha=0.2)
    axes[0].set_ylabel('Per-trial normalized gate weight')
    fig.tight_layout()
    fig.savefig(output / 'router_weight_distribution.png', dpi=300)
    fig.savefig(output / 'router_weight_distribution.pdf')
    plt.close(fig)


if __name__ == '__main__':
    run_checked(main)
