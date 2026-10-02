"""P0-B: inference-time masking sensitivity analysis (no retraining)."""
import logging

import numpy as np

from experiments.sere_common import (
    CONDITIONS, GATES, METRICS, bootstrap, collect, inspect_checkpoint, parser,
    run_checked, sample_identity, start_run, validate_dataset, write_csv, write_json,
)


def conditioned_embeddings(data, condition):
    """Reuse the frozen backbone/heads; apply exactly the forward's masking algebra."""
    v, s, h = (data['experts'][key] for key in ('visual', 'semantic', 'shared'))
    if condition == 'kill_visual':
        v = np.zeros_like(v)
    elif condition == 'kill_semantic':
        s = np.zeros_like(s)
    elif condition == 'kill_shared':
        h = np.zeros_like(h)
    elif condition not in ('full_moe', 'uniform_router'):
        raise ValueError(condition)
    gates = [data['gates'][key][:, None] for key in GATES]
    if condition == 'uniform_router':
        gates = [np.full_like(g, 0.5 / (1.0 + 1e-6)) for g in gates]
    outputs = {'image': gates[0] * v + gates[1] * h, 'text': gates[2] * s + gates[3] * h}
    for branch, value in outputs.items():
        norm = np.linalg.norm(value, axis=-1, keepdims=True)
        if not np.isfinite(value).all() or (norm == 0).any():
            raise ValueError(f'Invalid {condition} {branch} embeddings')
        outputs[branch] = value / norm
    return outputs


def retrieval(queries, candidates, candidate_ids, query_ids):
    """Unique split-local candidates. Ties use ascending target_id, recorded in protocol."""
    if len(candidate_ids) < 2 or len(candidate_ids) != len(np.unique(candidate_ids)):
        raise ValueError('At least two unique candidate targets are required')
    positives = np.searchsorted(candidate_ids, query_ids)
    if not np.array_equal(candidate_ids[positives], query_ids):
        raise ValueError('Queries missing from target candidate bank')
    scores = queries @ candidates.T
    order = np.argsort(-scores, axis=1, kind='stable')
    positive = scores[np.arange(len(queries)), positives]
    return {'r_at_1': (order[:, 0] == positives).astype(np.float64),
            'r_at_5': (order[:, :min(5, len(candidate_ids))] == positives[:, None]).any(axis=1).astype(np.float64),
            'positive_cosine': positive.astype(np.float64),
            'negative_cosine': ((scores.astype(np.float64).sum(axis=1) - positive) / (len(candidate_ids) - 1))}


def main():
    args = parser(__doc__, 'sere_masking').parse_args()
    cfg, device, output, provenance = start_run(args)
    if args.check_only:
        _, audit = inspect_checkpoint(args.encoder_ckpt, cfg)
        _, _, dataset_report = validate_dataset(cfg, args.split, args.split_index)
        write_json(output / 'resource_check.json', {'checkpoint': audit, 'dataset': dataset_report})
        if not audit['compatible']:
            raise RuntimeError('Incompatible encoder checkpoint')
        return
    data = collect(args, cfg, device)
    rows, metrics, results = [], [], {}
    for condition in CONDITIONS:
        embeddings = conditioned_embeddings(data, condition)
        for branch in ('image', 'text'):
            if condition == 'full_moe' and not np.allclose(embeddings[branch], data['finals'][branch], atol=2e-6, rtol=1e-5):
                raise ValueError('Cached full mixture differs from model.forward')
            value = retrieval(embeddings[branch], data['candidate_vectors'][branch],
                              data['candidate_ids'], data['target_ids'])
            results[condition, branch] = value
            metrics.append({'condition': condition, 'branch': branch, 'n_queries': len(data['target_ids']),
                            'n_candidates': len(data['candidate_ids']),
                            **{key: float(array.mean()) for key, array in value.items()}})
            rows.extend({**sample_identity(data, i), 'condition': condition, 'branch': branch,
                         **{key: float(array[i]) for key, array in value.items()}}
                        for i in range(len(data['target_ids'])))
        logging.info('Evaluated %s against %d unique split targets', condition, len(data['candidate_ids']))
    deltas = []
    for condition in CONDITIONS[1:]:
        for branch in ('image', 'text'):
            # Same query IDs and resampling for full and each condition: paired differences.
            value = np.column_stack([results['full_moe', branch][k] - results[condition, branch][k] for k in METRICS])
            ci = bootstrap(value, args.bootstrap_iters, args.seed)
            deltas.extend({'condition': condition, 'branch': branch, 'metric': key,
                           'delta_definition': 'full_moe_minus_condition', 'mean_delta': float(value[:, j].mean()),
                           'ci_low': float(ci[0, j]), 'ci_high': float(ci[1, j])} for j, key in enumerate(METRICS))
    write_csv(output / 'masking_per_sample.csv', rows)
    write_csv(output / 'masking_metrics.csv', metrics)
    write_csv(output / 'masking_delta_with_ci.csv', deltas)
    write_json(output / 'masking_metrics.json', {
        'analysis_type': 'inference-time masking sensitivity analysis',
        'dataset': data['dataset_report'], 'checkpoint': data['checkpoint'],
        'environment': provenance['environment'], 'metrics': metrics, 'paired_deltas': deltas,
        'protocol': {'candidate_bank': 'All unique target_ids in selected split, including smoke runs',
                     'candidate_ids': data['candidate_ids'].tolist(),
                     'r_at_k_units': 'fraction (multiply by 100 for percent)',
                     'ties': 'Stable descending cosine, ascending target_id',
                     'negative_cosine': 'Mean over all candidates other than the positive target_id',
                     'masking': 'Zero expert output, keep learned gates without renormalizing gates',
                     'kill_shared': 'Equivalent to model ablation=kill_fusion',
                     'uniform_router': 'Same frozen model, equal within-branch gates',
                     'execution': 'Backbone/heads evaluated once; exact frozen mixture recomposition',
                     'ci': '95% percentile paired trial bootstrap; repeated trials are not independent targets',
                     'bootstrap_iters': args.bootstrap_iters, 'seed': args.seed}})
    logging.info('P0-B finished: %s', output)


if __name__ == '__main__':
    run_checked(main)
