"""Read-only offline resource audit. JSON is written only to stdout."""
import argparse
import importlib.util
import contextlib
import json
from pathlib import Path
import sys

from experiments.sere_common import ROOT, environment, inspect_checkpoint, read_config, validate_dataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/triplet_config.yaml')
    p.add_argument('--encoder-ckpt', default='temp/best_12.8_change.pth')
    p.add_argument('--split-index', type=int, default=0)
    p.add_argument('--qwen-projector-ckpt', default='temp/best_eeg_projector.pth')
    p.add_argument('--eeg-img-proj-ckpt', default='temp/eeg_img_proj_ckpt.pth')
    p.add_argument('--qwen-model-dir', default='temp/Qwen2.5-Omni-3B')
    p.add_argument('--sd-model-dir', default='temp/sd15-diffusers')
    p.add_argument('--clip-model-dir', default='')
    p.add_argument('--captions-dir', default='data/text_data')
    p.add_argument('--image-root', default='data/image_data')
    args = p.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise ValueError('Run from repository root')
    cfg = read_config(args.config)
    report = {'environment': environment(), 'resources': {}, 'checkpoint_audits': [],
              'p0_blockers': [], 'p1_blockers': [], 'config': str(Path(args.config).resolve())}
    paths = {name: cfg.data[key] for name, key in (
        ('EEG_DATA_PATH', 'eeg_path'), ('IMAGE_VEC_PATH', 'image_vec_path'),
        ('TEXT_VEC_PATH', 'text_vec_path'), ('SPLITS_PATH', 'splits_path'))}
    for name, key in (('EEG_ENCODER_CKPT', 'encoder_ckpt'), ('QWEN_PROJECTOR_CKPT', 'qwen_projector_ckpt'),
                      ('EEG_IMG_PROJ_CKPT', 'eeg_img_proj_ckpt'), ('QWEN_MODEL_DIR', 'qwen_model_dir'),
                      ('SD_MODEL_DIR', 'sd_model_dir'), ('CLIP_MODEL_DIR', 'clip_model_dir'),
                      ('CAPTIONS_DIR', 'captions_dir'), ('IMAGE_ROOT', 'image_root')):
        paths[name] = getattr(args, key)
    if not paths['CLIP_MODEL_DIR']:
        cache = Path('C:/Users/HP/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots')
        snapshots = sorted(cache.glob('*')) if cache.exists() else []
        # Only select an actual local snapshot with weights; no online fallback.
        complete = [s for s in snapshots if (s / 'config.json').is_file()
                    and any((s / w).is_file() for w in ('pytorch_model.bin', 'model.safetensors'))]
        paths['CLIP_MODEL_DIR'] = str(complete[0]) if complete else str(ROOT / 'temp/clip-vit-base-patch32')
    p0_names = {'EEG_DATA_PATH', 'IMAGE_VEC_PATH', 'TEXT_VEC_PATH', 'SPLITS_PATH', 'EEG_ENCODER_CKPT'}
    for name, path in paths.items():
        resource = Path(path).resolve()
        report['resources'][name] = {'path': str(resource), 'exists': resource.exists()}
        if not resource.exists():
            report['p0_blockers' if name in p0_names else 'p1_blockers'].append(f'Missing {name}: {resource}')
    candidates = [Path(args.encoder_ckpt), *sorted((ROOT / 'wandb').glob('*/files/best_eeg_encoder.pth'))]
    for path in dict.fromkeys(candidates):
        if path.is_file():
            try:
                _, audit = inspect_checkpoint(path, cfg)
                report['checkpoint_audits'].append(audit)
                if path.resolve() == Path(args.encoder_ckpt).resolve() and not audit['compatible']:
                    report['p0_blockers'].append(f'Encoder incompatible: {path.resolve()}')
            except Exception as exc:
                report['checkpoint_audits'].append({'path': str(path.resolve()), 'error': str(exc)})
                if path.resolve() == Path(args.encoder_ckpt).resolve():
                    report['p0_blockers'].append(str(exc))
    if not report['p0_blockers']:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                ds, ids, dataset_report = validate_dataset(cfg, 'test', args.split_index)
            report['dataset'] = dataset_report
            unique_ids = sorted(set(ids.tolist()))
            def match(tid, caption):
                stem = Path(str(ds.eeg_images[tid])).stem
                folder = Path(paths['CAPTIONS_DIR'] if caption else paths['IMAGE_ROOT']) / stem.split('_')[0]
                suffixes = ('_caption.txt',) if caption else ('.JPEG', '.jpg', '.jpeg', '.png', '.JPG')
                return any((folder / (stem + suffix)).is_file() for suffix in suffixes)
            report['alignment'] = {'unique_test_targets': len(unique_ids),
                                   'matched_captions': sum(match(tid, True) for tid in unique_ids),
                                   'matched_images': sum(match(tid, False) for tid in unique_ids)}
            for kind in ('captions', 'images'):
                if report['alignment'][f'matched_{kind}'] != len(unique_ids):
                    report['p1_blockers'].append(f'Incomplete test target {kind} alignment: {report["alignment"]}')
        except Exception as exc:
            report['p0_blockers'].append(str(exc))
    report['dependencies'] = {m: importlib.util.find_spec(m) is not None
                              for m in ('torch', 'timm', 'numpy', 'scipy', 'omegaconf', 'matplotlib', 'transformers', 'diffusers')}
    if not report['dependencies']['diffusers']:
        report['p1_blockers'].append('Python dependency diffusers is not installed in the selected interpreter')
    report['p0_ready'] = not report['p0_blockers']
    # Existence is not strict loading; never call these checks proof of DPG readiness.
    report['p1_strict_loading'] = 'Not performed; P1 outside this run scope and missing resources block it'
    report['p1_ready'] = False
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report['p0_ready'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
