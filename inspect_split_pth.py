from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError as exc:
    print("Error: PyTorch is required to load .pth files. Please run this script in an environment with torch installed.")
    raise SystemExit(1) from exc

try:
    import yaml
except ImportError:
    yaml = None


SPLIT_NAMES = ("train", "val", "test")
SEARCH_KEYWORDS = ("split", "train", "test", "block", "eegx")
FIELD_ALIASES = {
    "image": ("image", "image_id", "img", "img_id", "imageidx", "image_idx", "imageindex"),
    "category": ("label", "class_id", "classid", "category", "category_id", "categoryid", "class", "concept", "concept_id"),
    "subject": ("subject", "subj", "subject_id", "subjectid", "participant", "participant_id", "participantid"),
    "block": ("block", "block_id", "blockid", "block_idx", "blockidx", "session", "session_id", "run", "run_id"),
    "trial": ("trial", "trial_id", "trialid", "trial_idx", "trialidx"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a .pth split file and print split statistics for paper writing."
    )
    parser.add_argument(
        "--split-path",
        type=str,
        default=None,
        help="Path to the split .pth file. If omitted, the script tries a project-aware default and then auto-search.",
    )
    parser.add_argument(
        "--eeg-path",
        type=str,
        default=None,
        help="Optional EEG/sample metadata .pth file used to map split indices back to sample records.",
    )
    parser.add_argument(
        "--split-index",
        type=int,
        default=0,
        help="Split index to inspect when the .pth contains multiple predefined splits. Default: 0",
    )
    parser.add_argument(
        "--search-root",
        type=str,
        default=".",
        help="Root directory for auto-searching candidate split files and configs. Default: current directory",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=20,
        help="Maximum number of auto-discovered candidate split files to print. Default: 20",
    )
    return parser.parse_args()


def to_path(path_like: str | Path | None, base_dir: Path) -> Path | None:
    if path_like is None:
        return None
    path = Path(path_like)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path


def load_pth(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def safe_len(obj: Any) -> str:
    try:
        return str(len(obj))
    except Exception:
        return "unknown"


def tensor_shape(obj: Any) -> str | None:
    shape = getattr(obj, "shape", None)
    if shape is None:
        return None
    try:
        return str(tuple(shape))
    except Exception:
        return str(shape)


def object_type_name(obj: Any) -> str:
    return type(obj).__name__


def normalize_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if value.is_integer():
            return int(value)
    return value


def is_index_like(value: Any) -> bool:
    value = normalize_scalar(value)
    return isinstance(value, int) and not isinstance(value, bool)


def normalize_sequence(obj: Any) -> list[Any] | None:
    if obj is None:
        return None
    if torch.is_tensor(obj):
        try:
            return obj.detach().cpu().reshape(-1).tolist()
        except Exception:
            return None
    if isinstance(obj, (list, tuple)):
        return list(obj)
    return None


def discover_split_candidates(search_root: Path, limit: int) -> list[Path]:
    candidates: list[tuple[int, int, str, Path]] = []
    for path in search_root.rglob("*.pth"):
        lower_name = path.name.lower()
        score = sum(1 for keyword in SEARCH_KEYWORDS if keyword in lower_name)
        if score <= 0:
            continue
        size_bonus = 1 if path.stat().st_size < 100_000_000 else 0
        candidates.append((-score, -size_bonus, lower_name, path.resolve()))
    candidates.sort()
    return [item[-1] for item in candidates[:limit]]


def resolve_string_path(raw_value: Any, project_root: Path) -> Path | None:
    if raw_value is None:
        return None
    path_str = str(raw_value).strip().strip("'").strip('"')
    if not path_str:
        return None
    replacements = {
        "${data.root}": str(project_root),
        "${hydra:runtime.cwd}": str(project_root),
    }
    for old, new in replacements.items():
        path_str = path_str.replace(old, new)
    path = Path(path_str)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    else:
        path = path.resolve()
    return path


def infer_eeg_path_from_configs(split_path: Path, search_root: Path) -> tuple[Path | None, str]:
    if yaml is None:
        return None, "PyYAML not installed; config-based EEG path inference skipped"

    split_name = split_path.name
    config_dirs = [search_root / "configs", search_root]
    checked_files: list[Path] = []
    def config_priority(path: Path) -> tuple[int, str]:
        lower = path.name.lower()
        if lower == "triplet_config.yaml":
            return (0, lower)
        if "triplet" in lower:
            return (1, lower)
        return (2, lower)

    for config_dir in config_dirs:
        if not config_dir.exists():
            continue
        yaml_paths = sorted(config_dir.rglob("*.y*ml"), key=config_priority)
        for path in yaml_paths:
            checked_files.append(path)
            try:
                with path.open("r", encoding="utf-8") as handle:
                    config = yaml.safe_load(handle)
            except Exception:
                continue
            if not isinstance(config, dict):
                continue
            data_cfg = config.get("data")
            if not isinstance(data_cfg, dict):
                continue
            split_value = data_cfg.get("splits_path")
            if split_value is None:
                continue
            split_candidate = resolve_string_path(split_value, search_root)
            matches = False
            if split_candidate is not None and split_candidate.name == split_name:
                matches = True
            elif split_name in str(split_value):
                matches = True
            if not matches:
                continue
            eeg_value = data_cfg.get("eeg_path")
            eeg_path = resolve_string_path(eeg_value, search_root)
            if eeg_path is not None and eeg_path.exists():
                return eeg_path, f"inferred from config: {path}"
    return None, f"unable to infer automatically from {len(checked_files)} config files"


def summarize_object(name: str, obj: Any, indent: str = "  ") -> list[str]:
    lines: list[str] = []
    obj_type = object_type_name(obj)
    shape = tensor_shape(obj)
    if shape is not None:
        lines.append(f"{indent}{name}: {obj_type}, shape={shape}")
        return lines
    if isinstance(obj, dict):
        keys = list(obj.keys())
        preview = ", ".join(str(key) for key in keys[:8]) if keys else "none"
        lines.append(f"{indent}{name}: dict, len={len(obj)}, sample_keys=[{preview}]")
        return lines
    if isinstance(obj, (list, tuple)):
        seq = list(obj)
        lines.append(f"{indent}{name}: {obj_type}, len={len(seq)}")
        if seq:
            first = seq[0]
            if isinstance(first, dict):
                preview = ", ".join(str(key) for key in list(first.keys())[:8]) or "none"
                lines.append(f"{indent}  first_item: dict, sample_keys=[{preview}]")
            elif torch.is_tensor(first):
                lines.append(f"{indent}  first_item: tensor, shape={tensor_shape(first)}")
            else:
                lines.append(f"{indent}  first_item: {object_type_name(first)}, repr={repr(first)[:120]}")
        return lines
    lines.append(f"{indent}{name}: {obj_type}, repr={repr(obj)[:120]}")
    return lines


def split_key_map(container: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for key in container.keys():
        lower = str(key).lower()
        if lower == "train" or "train" in lower:
            mapping["train"] = key
        elif lower == "val" or "valid" in lower:
            mapping["val"] = key
        elif lower == "test" or "eval" in lower:
            mapping["test"] = key
    return mapping


def extract_split_container(obj: Any, split_index: int) -> tuple[dict[str, Any] | None, str, int | None]:
    if isinstance(obj, dict):
        key_map = split_key_map(obj)
        if key_map:
            return {name: obj[key] for name, key in key_map.items()}, "top-level dict", None
        if "splits" in obj:
            splits = obj["splits"]
            if isinstance(splits, dict):
                key_map = split_key_map(splits)
                if key_map:
                    return {name: splits[key] for name, key in key_map.items()}, "dict['splits']", None
            seq = normalize_sequence(splits)
            if seq is None:
                return None, "dict['splits'] is not a list/tuple/tensor", None
            if not seq:
                return None, "dict['splits'] is empty", None
            if split_index < 0 or split_index >= len(seq):
                return None, f"split_index={split_index} out of range for {len(seq)} predefined splits", len(seq)
            selected = seq[split_index]
            if not isinstance(selected, dict):
                return None, f"selected split at index {split_index} is not a dict", len(seq)
            key_map = split_key_map(selected)
            if not key_map:
                return None, f"selected split at index {split_index} has no train/val/test keys", len(seq)
            return {name: selected[key] for name, key in key_map.items()}, "dict['splits'][split_index]", len(seq)
    if isinstance(obj, (list, tuple)):
        seq = list(obj)
        if not seq:
            return None, "top-level list/tuple is empty", len(seq)
        if all(isinstance(item, dict) and split_key_map(item) for item in seq):
            if split_index < 0 or split_index >= len(seq):
                return None, f"split_index={split_index} out of range for {len(seq)} predefined splits", len(seq)
            selected = seq[split_index]
            key_map = split_key_map(selected)
            return {name: selected[key] for name, key in key_map.items()}, "top-level list/tuple", len(seq)
    return None, "unable to infer automatically", None


def extract_dataset_records(obj: Any) -> tuple[list[Any] | None, str]:
    if isinstance(obj, dict):
        if "dataset" in obj and isinstance(obj["dataset"], list):
            return obj["dataset"], "dict['dataset']"
        for candidate_key in ("data", "items", "samples"):
            if candidate_key in obj and isinstance(obj[candidate_key], list):
                return obj[candidate_key], f"dict['{candidate_key}']"
    if isinstance(obj, list):
        return obj, "top-level list"
    if isinstance(obj, tuple):
        return list(obj), "top-level tuple"
    return None, "unable to infer automatically"


def find_field_value(obj: Any, aliases: tuple[str, ...], depth: int = 0, max_depth: int = 4) -> tuple[Any, str] | tuple[None, None]:
    if depth > max_depth:
        return None, None
    if isinstance(obj, dict):
        lower_map = {str(key).lower(): key for key in obj.keys()}
        for alias in aliases:
            if alias in lower_map:
                key = lower_map[alias]
                return obj[key], str(key)
        for key, value in obj.items():
            found_value, found_path = find_field_value(value, aliases, depth + 1, max_depth)
            if found_path is not None:
                return found_value, f"{key}.{found_path}"
    elif isinstance(obj, (list, tuple)):
        for idx, value in enumerate(list(obj)[:3]):
            found_value, found_path = find_field_value(value, aliases, depth + 1, max_depth)
            if found_path is not None:
                return found_value, f"[{idx}].{found_path}"
    return None, None


def format_id_list(values: set[Any]) -> str:
    if not values:
        return "[]"
    normalized = [normalize_scalar(value) for value in values]
    if all(isinstance(value, int) for value in normalized):
        return "[" + ", ".join(str(value) for value in sorted(normalized)) + "]"
    return "[" + ", ".join(str(value) for value in sorted(normalized, key=lambda item: str(item))) + "]"


def sequence_from_split_value(value: Any) -> list[Any] | None:
    seq = normalize_sequence(value)
    if seq is not None:
        return seq
    if isinstance(value, dict):
        for key in ("indices", "idx", "items", "samples", "data"):
            if key in value:
                seq = normalize_sequence(value[key])
                if seq is not None:
                    return seq
    return None


def records_for_split(split_value: Any, dataset_records: list[Any] | None) -> tuple[list[Any] | None, str, int]:
    seq = sequence_from_split_value(split_value)
    if seq is None:
        return None, "unable to infer automatically", 0
    missing = 0
    if seq and all(is_index_like(item) for item in seq):
        if dataset_records is None:
            return None, "split entries look like indices, but metadata source is unavailable", 0
        records: list[Any] = []
        for raw_index in seq:
            index = int(normalize_scalar(raw_index))
            if 0 <= index < len(dataset_records):
                records.append(dataset_records[index])
            else:
                missing += 1
        return records, "mapped from split indices to external EEG/sample metadata", missing
    return seq, "split entries appear to contain samples directly", missing


def collect_field_stats(records: list[Any]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    values_map: dict[str, list[Any]] = {name: [] for name in FIELD_ALIASES}
    field_paths: dict[str, str] = {}
    for record in records:
        for field_name, aliases in FIELD_ALIASES.items():
            value, found_path = find_field_value(record, aliases)
            if found_path is None:
                continue
            if field_name not in field_paths:
                field_paths[field_name] = found_path
            normalized = normalize_scalar(value)
            values_map[field_name].append(normalized)

    stats: dict[str, dict[str, Any]] = {}
    for field_name in FIELD_ALIASES:
        values = values_map[field_name]
        if not values:
            stats[field_name] = {"found": False, "count": None, "values": set()}
        else:
            unique_values = {value for value in values}
            stats[field_name] = {"found": True, "count": len(unique_values), "values": unique_values}
    return stats, field_paths


def split_statistics(
    split_name: str,
    split_value: Any,
    dataset_records: list[Any] | None,
) -> dict[str, Any]:
    seq = sequence_from_split_value(split_value)
    sample_count = len(seq) if seq is not None else "unknown"
    records, record_source, missing_count = records_for_split(split_value, dataset_records)
    result: dict[str, Any] = {
        "split_name": split_name,
        "sample_count": sample_count,
        "record_source": record_source,
        "missing_record_count": missing_count,
        "field_paths": {},
        "field_stats": {},
    }
    if records is None:
        result["trial_count"] = "unable to infer automatically"
        return result

    field_stats, field_paths = collect_field_stats(records)
    result["field_stats"] = field_stats
    result["field_paths"] = field_paths

    if field_stats["trial"]["found"]:
        result["trial_count"] = field_stats["trial"]["count"]
        result["trial_note"] = f"derived from explicit field '{field_paths.get('trial', 'trial')}'"
    else:
        result["trial_count"] = len(records)
        result["trial_note"] = "explicit trial field not found; treated each split entry as one trial"
    return result


def overlap_summary(field_name: str, stats_a: dict[str, Any], stats_b: dict[str, Any]) -> tuple[str, set[Any] | None]:
    field_a = stats_a.get("field_stats", {}).get(field_name, {})
    field_b = stats_b.get("field_stats", {}).get(field_name, {})
    if not field_a.get("found") or not field_b.get("found"):
        return "field not found", None
    overlap = field_a["values"] & field_b["values"]
    if field_name == "image" and not overlap:
        return "no image overlap", overlap
    return f"overlap count: {len(overlap)}", overlap


def infer_granularity(
    split_path: Path,
    stats_by_split: dict[str, dict[str, Any]],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    train_stats = stats_by_split.get("train")
    test_stats = stats_by_split.get("test")
    if not train_stats or not test_stats:
        return "unable to infer automatically", ["train/test split pair is incomplete"]

    image_overlap_text, image_overlap = overlap_summary("image", train_stats, test_stats)
    subject_overlap_text, subject_overlap = overlap_summary("subject", train_stats, test_stats)
    block_overlap_text, block_overlap = overlap_summary("block", train_stats, test_stats)

    split_name = split_path.name.lower()
    if image_overlap == set():
        reasons.append("train/test images are disjoint")
        if subject_overlap is not None and len(subject_overlap) > 0:
            reasons.append("subjects overlap across train/test, so this is not a subject-level split")
        if "by_image" in split_name:
            reasons.append("filename contains 'by_image'")
        if "block" in split_name:
            reasons.append("filename also contains 'block', but block fields are not exposed in the loaded samples")
        return "likely image-level split", reasons

    if block_overlap == set():
        reasons.append("train/test blocks are disjoint")
        if "block" in split_name:
            reasons.append("filename contains 'block'")
        return "likely block-level split", reasons

    if subject_overlap == set():
        reasons.append("train/test subjects are disjoint")
        return "likely subject-level split", reasons

    if image_overlap is not None and len(image_overlap) > 0:
        reasons.append("train/test share images")
        if block_overlap_text == "field not found":
            reasons.append("no block field was found, so trial-level separation cannot be ruled out")
        return "likely trial-level split", reasons

    return "unable to infer automatically", ["available fields do not support a confident judgment"]


def format_split_statistics(split_name: str, stats: dict[str, Any]) -> list[str]:
    lines = [f"{split_name.capitalize()} statistics:"]
    lines.append(f"  sample count: {stats.get('sample_count', 'unknown')}")
    lines.append(f"  trial count: {stats.get('trial_count', 'unable to infer automatically')}")
    trial_note = stats.get("trial_note")
    if trial_note:
        lines.append(f"  trial note: {trial_note}")

    for field_name, label in (
        ("image", "unique image count"),
        ("category", "unique category count"),
        ("subject", "subject count"),
        ("block", "block count"),
    ):
        field_stats = stats.get("field_stats", {}).get(field_name, {})
        if field_stats.get("found"):
            lines.append(f"  {label}: {field_stats['count']}")
            if field_name == "subject":
                lines.append(f"  subject IDs: {format_id_list(field_stats['values'])}")
        else:
            lines.append(f"  {label}: field not found")

    field_paths = stats.get("field_paths", {})
    if field_paths:
        used_fields = ", ".join(f"{name}={path}" for name, path in sorted(field_paths.items()))
        lines.append(f"  detected fields: {used_fields}")
    else:
        lines.append("  detected fields: unable to infer automatically")

    lines.append(f"  metadata mapping: {stats.get('record_source', 'unknown')}")
    missing_record_count = stats.get("missing_record_count", 0)
    if missing_record_count:
        lines.append(f"  missing mapped records: {missing_record_count}")
    return lines


def paper_notes(
    split_path: Path,
    split_index: int,
    stats_by_split: dict[str, dict[str, Any]],
    inferred_granularity: str,
    overlap_results: dict[str, tuple[str, set[Any] | None]],
) -> list[str]:
    lines: list[str] = ["Notes for paper writing:"]
    lines.append(f"  predefined split file: {split_path.name} (split index {split_index})")
    lines.append(f"  inferred split granularity: {inferred_granularity}")

    train_stats = stats_by_split.get("train")
    val_stats = stats_by_split.get("val")
    test_stats = stats_by_split.get("test")
    if train_stats and test_stats:
        lines.append(
            f"  train/test trial counts: {train_stats.get('trial_count', 'NA')} / {test_stats.get('trial_count', 'NA')}"
        )
    if val_stats:
        lines.append(f"  validation trial count: {val_stats.get('trial_count', 'NA')}")

    image_text, _ = overlap_results.get("image", ("field not found", None))
    category_text, _ = overlap_results.get("category", ("field not found", None))
    subject_text, _ = overlap_results.get("subject", ("field not found", None))
    block_text, _ = overlap_results.get("block", ("field not found", None))
    lines.append(f"  image overlap: {image_text}")
    lines.append(f"  category overlap: {category_text}")
    lines.append(f"  subject overlap: {subject_text}")
    lines.append(f"  block overlap: {block_text}")
    return lines


def generate_english_template(
    split_path: Path,
    split_index: int,
    stats_by_split: dict[str, dict[str, Any]],
    inferred_granularity: str,
    overlap_results: dict[str, tuple[str, set[Any] | None]],
) -> list[str]:
    train_stats = stats_by_split.get("train")
    val_stats = stats_by_split.get("val")
    test_stats = stats_by_split.get("test")

    def get_trial_count(name: str) -> str:
        stats = stats_by_split.get(name)
        if not stats:
            return "N/A"
        return str(stats.get("trial_count", "N/A"))

    def get_unique_count(name: str, field: str) -> str:
        stats = stats_by_split.get(name)
        if not stats:
            return "N/A"
        field_stats = stats.get("field_stats", {}).get(field, {})
        if not field_stats.get("found"):
            return "field not found"
        return str(field_stats["count"])

    image_text, image_overlap = overlap_results.get("image", ("field not found", None))
    category_text, category_overlap = overlap_results.get("category", ("field not found", None))
    subject_text, subject_overlap = overlap_results.get("subject", ("field not found", None))
    block_text, block_overlap = overlap_results.get("block", ("field not found", None))

    lines = ["Paper-ready English template:"]
    lines.append(
        f"  A predefined split file ({split_path.name}, split index {split_index}) was used for all experiments."
    )

    granularity_map = {
        "likely image-level split": "image level",
        "likely block-level split": "block level",
        "likely subject-level split": "subject level",
        "likely trial-level split": "trial level",
    }
    granularity_phrase = granularity_map.get(inferred_granularity)
    if granularity_phrase is None:
        lines.append(
            "  The dataset split granularity could not be determined fully automatically and should be described as a predefined split."
        )
    else:
        lines.append(f"  The dataset split was performed at the {granularity_phrase}.")

    if train_stats and val_stats and test_stats:
        lines.append(
            "  The training, validation, and test sets contain "
            f"{get_trial_count('train')}, {get_trial_count('val')}, and {get_trial_count('test')} trials, respectively."
        )
    elif train_stats and test_stats:
        lines.append(
            f"  The training and test sets contain {get_trial_count('train')} and {get_trial_count('test')} trials, respectively."
        )
    else:
        lines.append("  Trial counts should be filled in manually because the required split statistics were incomplete.")

    if train_stats and test_stats:
        lines.append(
            "  They correspond to "
            f"{get_unique_count('train', 'image')} and {get_unique_count('test', 'image')} unique images, "
            f"and {get_unique_count('train', 'category')} and {get_unique_count('test', 'category')} unique categories, respectively."
        )

    if image_overlap == set():
        lines.append("  No image overlap was observed between the training and test sets.")
    elif image_overlap is not None:
        lines.append(
            f"  Image overlap was observed between the training and test sets ({len(image_overlap)} shared images)."
        )
    else:
        lines.append("  Image overlap could not be determined automatically.")

    if category_overlap is not None:
        lines.append(
            f"  Category overlap between the training and test sets was {len(category_overlap)} shared categories."
        )
    else:
        lines.append("  Category overlap could not be determined automatically.")

    if subject_overlap is not None:
        lines.append(
            f"  Subject overlap between the training and test sets was {len(subject_overlap)} shared subjects."
        )
    else:
        lines.append("  Subject overlap could not be determined automatically.")

    if block_overlap is not None:
        lines.append(f"  Block overlap between the training and test sets was {len(block_overlap)} shared blocks.")
    else:
        lines.append("  Block overlap could not be verified automatically because no block field was found.")

    lines.append("  All compared methods used the same predefined split.")
    return lines


def print_section(lines: list[str]) -> None:
    for line in lines:
        print(line)


def main() -> None:
    args = parse_args()
    search_root = to_path(args.search_root, Path.cwd())
    if search_root is None:
        raise RuntimeError("search_root resolved to None")

    default_split = search_root / "data" / "EEG_data" / "block_splits_by_image_all.pth"
    split_candidates = discover_split_candidates(search_root, args.max_candidates)

    split_path = to_path(args.split_path, search_root)
    if split_path is None:
        if default_split.exists():
            split_path = default_split.resolve()
        elif split_candidates:
            split_path = split_candidates[0]
        else:
            raise FileNotFoundError("No split-like .pth file was found under the search root.")

    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    eeg_path = to_path(args.eeg_path, search_root)
    eeg_path_source = "provided by --eeg-path"
    if eeg_path is None:
        inferred_eeg_path, eeg_path_source = infer_eeg_path_from_configs(split_path, search_root)
        eeg_path = inferred_eeg_path

    split_obj = load_pth(split_path)
    split_container, split_container_source, total_split_count = extract_split_container(split_obj, args.split_index)
    if split_container is None:
        raise ValueError(f"Unable to parse split container from {split_path}: {split_container_source}")

    dataset_records = None
    dataset_source = "not used"
    if eeg_path is not None and eeg_path.exists():
        eeg_obj = load_pth(eeg_path)
        dataset_records, dataset_source = extract_dataset_records(eeg_obj)
    elif eeg_path is not None:
        eeg_path_source = f"{eeg_path_source}; file does not exist"
        eeg_path = None

    print(f"Split file:")
    print(f"  path: {split_path}")
    print(f"  filename: {split_path.name}")
    print(f"  selected split index: {args.split_index}")
    print(f"  split container source: {split_container_source}")
    print(f"  total predefined splits: {total_split_count if total_split_count is not None else 'not applicable'}")
    print(f"  metadata EEG/source file: {eeg_path if eeg_path is not None else 'field not found'}")
    print(f"  metadata source resolution: {eeg_path_source}")
    print(f"  metadata extraction: {dataset_source}")
    if split_candidates:
        print(f"  auto-discovered split-like files ({len(split_candidates)} shown):")
        for candidate in split_candidates:
            print(f"    - {candidate}")
    else:
        print("  auto-discovered split-like files: none")
    print()

    print("Available splits:")
    for split_name in SPLIT_NAMES:
        if split_name in split_container:
            seq = sequence_from_split_value(split_container[split_name])
            count_text = len(seq) if seq is not None else "unknown"
            print(f"  {split_name}: present ({count_text} entries)")
        else:
            print(f"  {split_name}: missing")
    print()

    print("Split file structure summary:")
    if isinstance(split_obj, dict):
        print(f"  top-level type: dict")
        print(f"  top-level keys: {list(split_obj.keys())}")
        for key, value in split_obj.items():
            print_section(summarize_object(str(key), value))
    else:
        print(f"  top-level type: {object_type_name(split_obj)}")
        print_section(summarize_object("top_level", split_obj))
    print()

    stats_by_split: dict[str, dict[str, Any]] = {}
    for split_name in SPLIT_NAMES:
        if split_name not in split_container:
            continue
        stats = split_statistics(split_name, split_container[split_name], dataset_records)
        stats_by_split[split_name] = stats
        print_section(format_split_statistics(split_name, stats))
        print()

    print("Overlap checks:")
    overlap_results: dict[str, tuple[str, set[Any] | None]] = {}
    if "train" in stats_by_split and "test" in stats_by_split:
        for field_name, label in (
            ("image", "train/test image overlap"),
            ("category", "train/test category overlap"),
            ("subject", "train/test subject overlap"),
            ("block", "train/test block overlap"),
        ):
            text, overlap_values = overlap_summary(field_name, stats_by_split["train"], stats_by_split["test"])
            overlap_results[field_name] = (text, overlap_values)
            print(f"  {label}: {text}")
            if field_name == "subject" and overlap_values:
                print(f"  overlapping subject IDs: {format_id_list(overlap_values)}")
    else:
        print("  unable to infer automatically")
    print()

    inferred_granularity, granularity_reasons = infer_granularity(split_path, stats_by_split)
    print("Inferred split granularity:")
    print(f"  result: {inferred_granularity}")
    for reason in granularity_reasons:
        print(f"  basis: {reason}")
    print()

    print_section(paper_notes(split_path, args.split_index, stats_by_split, inferred_granularity, overlap_results))
    print()
    print_section(generate_english_template(split_path, args.split_index, stats_by_split, inferred_granularity, overlap_results))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
