"""Shared P0 protocol: strict loading, deterministic trials, bootstrap and audit."""
import argparse
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import random
import subprocess
import sys

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataset import TripletDataset
from models.clip_models import SpatialMoEEncoder

ROOT = Path(__file__).resolve().parents[1]
GATES = ('gate_visual_img', 'gate_shared_img', 'gate_semantic_txt', 'gate_shared_txt')
EXPERTS = ('visual', 'shared', 'semantic')
CONDITIONS = ('full_moe', 'kill_visual', 'kill_semantic', 'kill_shared', 'uniform_router')
METRICS = ('r_at_1', 'r_at_5', 'positive_cosine', 'negative_cosine')


def parser(description, output_name):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--config', default='configs/triplet_config.yaml')
    p.add_argument('--encoder-ckpt', required=True)
    p.add_argument('--split', choices=('train', 'val', 'test'), default='test')
    p.add_argument('--split-index', type=int, default=0)
    p.add_argument('--output-dir', default=f'artifacts/supplement_experiments/{output_name}')
    p.add_argument('--batch-size', type=int, default=64)
    # EEG is a large in-memory dataset; Windows spawn would copy it to each worker.
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-samples', type=int, default=0)
    p.add_argument('--bootstrap-iters', type=int, default=1000)
    p.add_argument('--check-only', action='store_true')
    return p


def read_config(path, seen=()):
    """Compose local YAML defaults without starting Hydra/SLURM/WandB."""
    path = Path(path).resolve()
    if path in seen:
        raise ValueError(f'Cyclic config defaults: {path}')
    own = OmegaConf.load(path)
    defaults = own.pop('defaults', [])
    result = OmegaConf.create({})
    inserted = False
    for entry in defaults:
        if entry == '_self_':
            result = OmegaConf.merge(result, own)
            inserted = True
        elif isinstance(entry, str):
            base = path.parent / (entry if entry.endswith('.yaml') else entry + '.yaml')
            result = OmegaConf.merge(result, read_config(base, (*seen, path)))
        else:
            # The base config only has a Hydra launcher override, irrelevant to inference.
            if not all(str(key).startswith(('override hydra/', 'hydra/')) for key in entry):
                raise ValueError(f'Unsupported experiment config default: {entry}')
    if not inserted:
        result = OmegaConf.merge(result, own)
    result.data.root = str(ROOT)
    result.data.return_target_id = True
    result.data.return_caption = False
    for key in ('eeg_path', 'image_vec_path', 'text_vec_path', 'splits_path'):
        result.data[key] = str(Path(str(result.data[key])).resolve())
    # Stage-1 weights supply the entire model. Do not load the generative pretraining file.
    result.model.pretrained_path = None
    result.pop('hydra', None)
    return result


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def git(*args):
    return subprocess.check_output(
        ['git', '-c', f'safe.directory={ROOT.as_posix()}', *args], cwd=ROOT, text=True, encoding='utf-8'
    ).strip()


def environment():
    return {
        'python': platform.python_version(), 'executable': sys.executable,
        'pytorch': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
        'cuda_available': torch.cuda.is_available(),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'commit_sha': git('rev-parse', 'HEAD'), 'branch': git('rev-parse', '--abbrev-ref', 'HEAD'),
    }


def seed_everything(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def model_from_config(cfg, device='cpu'):
    with torch.device(device):
        return SpatialMoEEncoder(
            n_channels=int(cfg.model.n_channels), n_samples=int(cfg.model.n_samples),
            embedding_dim=int(cfg.model.embedding_dim), pretrained_path=None,
            router_mode=str(cfg.model.get('router_mode', 'moe')),
            head_dropout=float(cfg.model.get('head_dropout', 0.5)),
        )


def inspect_checkpoint(path, cfg):
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    for wrapper in ('state_dict', 'model_state_dict', 'model'):
        if wrapper in state and isinstance(state[wrapper], dict):
            state = state[wrapper]
            break
    if all(k.startswith('module.') for k in state):
        state = {k[len('module.'):]: v for k, v in state.items()}
    reference = model_from_config(cfg, device='meta').state_dict()
    missing = sorted(reference.keys() - state.keys())
    unexpected = sorted(state.keys() - reference.keys())
    shapes = {k: {'expected': list(reference[k].shape), 'actual': list(state[k].shape)}
              for k in reference.keys() & state.keys() if reference[k].shape != state[k].shape}
    # These are obsolete index metadata, unused by the current forward. No learned key
    # may be dropped/remapped. Report the original unexpected keys before filtering.
    legacy = [k for k in unexpected if k in ('idx_vis', 'idx_sem')]
    report = {'path': str(Path(path).resolve()), 'sha256': sha256(path),
              'missing_keys': missing, 'unexpected_keys': unexpected,
              'shape_mismatches': shapes, 'ignored_legacy_buffers': legacy,
              'compatible': not missing and not shapes and set(unexpected) <= set(legacy)}
    logging.info('Checkpoint audit: %s', json.dumps(report))
    return {k: v for k, v in state.items() if k not in legacy}, report


def load_encoder(path, cfg, device):
    state, audit = inspect_checkpoint(path, cfg)
    if not audit['compatible']:
        raise RuntimeError(f'Checkpoint incompatible; stop without random initialization: {audit}')
    model = model_from_config(cfg, device='meta')
    model.load_state_dict(state, strict=True, assign=True)
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    if model.router_mode != 'moe':
        raise ValueError(f'P0 learned routing requires router_mode=moe, got {model.router_mode}')
    return model, audit


def validate_dataset(cfg, split, split_index):
    """Validate all split IDs before TripletDataset can silently skip invalid rows."""
    for key in ('eeg_path', 'image_vec_path', 'text_vec_path', 'splits_path'):
        if not Path(cfg.data[key]).is_file():
            raise FileNotFoundError(f'{key}: {cfg.data[key]}')
    splits = torch.load(cfg.data.splits_path, map_location='cpu', weights_only=True)['splits'][split_index]
    ds = TripletDataset(cfg.data, mode=split, split_index=split_index)
    if getattr(ds, 'backend', '') == 'ds003825':
        raise ValueError('This P0 protocol requires the original EEG/CLIP aligned dataset')
    raw = [int(i) for i in splits[split]]
    if not raw or len(raw) != len(set(raw)) or raw != [int(i) for i in ds.indices]:
        raise ValueError('Empty/duplicate/filtered trial IDs: split mapping is uncertain')
    if ds.eeg_images is None or len(ds.eeg_images) != len(ds.all_image_vectors):
        raise ValueError('EEG images list and target-vector row count must match')
    if ds.all_image_vectors.shape != ds.all_text_vectors.shape:
        raise ValueError('Image/text target matrix shape mismatch')
    if ds.all_image_vectors.shape[1] != int(cfg.model.embedding_dim):
        raise ValueError('Target dimension differs from model embedding dimension')
    vector_issues = {}
    selected_ids = sorted({int(ds.all_eeg_items[i]['image']) for i in ds.indices})
    for branch, vectors in (('image', ds.all_image_vectors), ('text', ds.all_text_vectors)):
        invalid = (~torch.isfinite(vectors).all(dim=-1)) | (vectors.norm(dim=-1) == 0)
        invalid_ids = torch.nonzero(invalid).flatten().tolist()
        vector_issues[branch] = invalid_ids
        if set(invalid_ids) & set(selected_ids):
            raise ValueError(f'Nonfinite/zero {branch} targets in selected split: {invalid_ids}')
        if invalid_ids:
            logging.warning('%s target defects outside selected split: %s', branch, invalid_ids)
    target_sets = {}
    for name in ('train', 'val', 'test'):
        indices = [int(i) for i in splits[name]]
        if len(indices) != len(set(indices)):
            raise ValueError(f'Duplicate trial IDs in {name}')
        ids = []
        for index in indices:
            if not 0 <= index < len(ds.all_eeg_items):
                raise ValueError(f'Invalid EEG index: {index}')
            tid = int(ds.all_eeg_items[index]['image'])
            if not 0 <= tid < ds.num_available_vectors:
                raise ValueError(f'Invalid target_id: {tid}')
            ids.append(tid)
        target_sets[name] = set(ids)
    for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if target_sets[left] & target_sets[right]:
            raise ValueError(f'Target leakage between {left} and {right}')
    # Disable train augmentation during inference without changing the selected indices.
    ds.mode = 'test'
    ids = np.array([int(ds.all_eeg_items[i]['image']) for i in ds.indices], dtype=np.int64)
    report = {'split': split, 'split_index': split_index, 'trials': len(ds),
              'unique_targets': len(np.unique(ids)), 'skipped_trials': 0,
              'target_id_mapping': "EEG dataset[index]['image'] -> aligned vector row -> images[target_id]",
              'target_alignment_basis': 'Repository aligned vectors and EEG images list; no CLIP re-encoding',
              'split_unique_targets': {k: len(v) for k, v in target_sets.items()},
              'invalid_target_rows_outside_selected_split': vector_issues,
              'preprocessing': 'TripletDataset: crop [20:460], scipy linear interpolation to 512; no augmentation'}
    return ds, ids, report


def normalize(tensor):
    if not torch.isfinite(tensor).all() or (tensor.norm(dim=-1) == 0).any():
        raise ValueError('Nonfinite/zero embedding; stop before computing metrics')
    return F.normalize(tensor.float(), dim=-1)


def bootstrap(values, iterations=1000, seed=42):
    """Trial-level percentile bootstrap; columns share the same resampled trials."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Invalid bootstrap values')
    rng = np.random.default_rng(seed)
    means = np.empty((iterations, values.shape[1]))
    for i in range(iterations):
        means[i] = values[rng.integers(0, len(values), len(values))].mean(axis=0)
    return np.quantile(means, [0.025, 0.975], axis=0)


def write_csv(path, rows):
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def start_run(args):
    if Path.cwd().resolve() != ROOT:
        raise ValueError(f'Run from repository root: {ROOT}')
    if git('rev-parse', '--abbrev-ref', 'HEAD') != 'dvsu-revision':
        raise ValueError('Experiments must run on dvsu-revision')
    if min(args.batch_size, args.bootstrap_iters) < 1 or min(args.max_samples, args.num_workers) < 0:
        raise ValueError('Invalid batch size / bootstrap count / sample limit / worker count')
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'run_config.yaml').exists():
        raise FileExistsError(f'Output already contains a run; choose a new --output-dir: {output}')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(output / 'run.log', encoding='utf-8'), logging.StreamHandler()])
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no silent CPU fallback')
    cfg = read_config(args.config)
    provenance = {'arguments': vars(args), 'resolved_config': OmegaConf.to_container(cfg, resolve=True),
                  'environment': environment(), 'command': subprocess.list2cmdline([sys.executable, '-m',
                  sys.argv[0].replace(str(ROOT), '').lstrip('\\/').replace('\\', '/').removesuffix('.py').replace('/', '.'),
                  *sys.argv[1:]])}
    OmegaConf.save(OmegaConf.create(provenance), output / 'run_config.yaml')
    return cfg, device, output, provenance


def collect(args, cfg, device):
    ds, target_ids, dataset_report = validate_dataset(cfg, args.split, args.split_index)
    model, audit = load_encoder(args.encoder_ckpt, cfg, device)
    count = min(args.max_samples, len(ds)) if args.max_samples else len(ds)
    loader = DataLoader(Subset(ds, range(count)), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    embeddings = {name: [] for name in EXPERTS}
    weights = {key: [] for key in GATES}
    targets = {'image': [], 'text': []}
    finals = {'image': [], 'text': []}
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            eeg, image, text, ids = batch[:4]
            image_out, text_out, details = model(eeg.to(device), return_details=True)
            for key in GATES:
                gate = details[key]
                if gate.shape != (len(eeg), 1) or not torch.isfinite(gate).all() or ((gate < 0) | (gate > 1)).any():
                    raise ValueError(f'Invalid per-trial gate: {key}')
                weights[key].append(gate.cpu().numpy().reshape(-1))
            for a, b in ((GATES[0], GATES[1]), (GATES[2], GATES[3])):
                if not torch.allclose(details[a] + details[b], torch.ones_like(details[a]), atol=1e-5, rtol=0):
                    raise ValueError('Gate pairs do not sum to one')
            for name in EXPERTS:
                value = details[f'emb_{name}']
                normalize(value)  # Validate before preserving magnitudes for masking.
                embeddings[name].append(value.float().cpu().numpy())
            for branch, target, final in (('image', image, image_out), ('text', text, text_out)):
                targets[branch].append(normalize(target).numpy())
                finals[branch].append(normalize(final).cpu().numpy())
            logging.info('Inference %d/%d batches (%d/%d trials)', index + 1, len(loader),
                         min((index + 1) * args.batch_size, count), count)
    result = {'experts': {k: np.concatenate(v) for k, v in embeddings.items()},
              'gates': {k: np.concatenate(v) for k, v in weights.items()},
              'targets': {k: np.concatenate(v) for k, v in targets.items()},
              'finals': {k: np.concatenate(v) for k, v in finals.items()},
              'target_ids': target_ids[:count], 'dataset_indices': np.arange(count),
              'eeg_indices': np.asarray(ds.indices[:count], dtype=np.int64),
              # Candidates always include all targets from the selected split, even in smoke runs.
              'candidate_ids': np.unique(target_ids),
              'candidate_vectors': {branch: getattr(ds, f'all_{branch}_vectors')[np.unique(target_ids)].numpy()
                                    for branch in ('image', 'text')},
              'dataset_report': {**dataset_report, 'evaluated_trials': count}, 'checkpoint': audit}
    return result


def sample_identity(data, i):
    return {'dataset_index': int(data['dataset_indices'][i]), 'eeg_index': int(data['eeg_indices'][i]),
            'target_id': int(data['target_ids'][i])}


def run_checked(main):
    try:
        main()
    except Exception:
        logging.exception('Experiment stopped; no substitute weights or fabricated results')
        raise
