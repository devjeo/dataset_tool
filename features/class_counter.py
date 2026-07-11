"""
Feature: Class Counter

Counts class *instances*, not images: for every label line in every
labels/*.txt file, the leading class index is counted once. Reports are
split by train/ and valid/, since a class can be common in one split and
rare (or missing) in the other.

- Dataset is picked with the same folder browser used elsewhere; a valid
  dataset needs a train/labels and/or valid/labels folder.
- Class names come from the dataset's data.yaml (via
  dataset_utils.find_yaml_file) when available. If no yaml is found, or
  it has no readable "names" list, classes fall back to "class <idx>".
- Results are shown in a scrollable table (cli_ui.table_menu) with
  columns: Class / Train / Valid / Total, plus a TOTAL row at the end.
"""
import os
from collections import Counter

from dataset_utils import find_yaml_file
from cli_ui import browse_directory_menu, table_menu


def _is_countable_dataset(path):
    return os.path.isdir(os.path.join(path, "train", "labels")) or os.path.isdir(
        os.path.join(path, "valid", "labels")
    )


def _load_class_names(dataset_dir):
    """Returns {class_index: class_name}. Empty dict if nothing usable found."""
    yaml_path = find_yaml_file(dataset_dir)
    if not yaml_path:
        return {}

    try:
        import yaml
    except ImportError:
        print("  [Warning] PyYAML not installed - class names will show as 'class <idx>'.")
        return {}

    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as e:
        print(f"  [Warning] Could not read {yaml_path}: {e}")
        return {}

    names = data.get("names")
    if isinstance(names, dict):
        try:
            return {int(k): str(v) for k, v in names.items()}
        except (TypeError, ValueError):
            return {}
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    return {}


def _count_labels_dir(labels_dir):
    """Returns {class_index: instance_count} for every .txt file in labels_dir."""
    counts = Counter()
    if not os.path.isdir(labels_dir):
        return counts

    for fname in os.listdir(labels_dir):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(labels_dir, fname)
        try:
            with open(fpath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    try:
                        cls_idx = int(float(parts[0]))
                    except (ValueError, IndexError):
                        continue
                    counts[cls_idx] += 1
        except OSError as e:
            print(f"  [Warning] Could not read {fpath}: {e}")

    return counts


def run():
    dataset_dir = browse_directory_menu(
        "Class Counter",
        start_dir=".",
        accept_predicate=_is_countable_dataset,
        accept_label="Use this folder as dataset (needs train/ and/or valid/)",
    )
    if dataset_dir is None:
        print("Cancelled.")
        return

    class_names = _load_class_names(dataset_dir)

    train_counts = _count_labels_dir(os.path.join(dataset_dir, "train", "labels"))
    valid_counts = _count_labels_dir(os.path.join(dataset_dir, "valid", "labels"))

    all_indices = sorted(set(train_counts) | set(valid_counts))
    if not all_indices:
        print("No label instances found in train/ or valid/ - nothing to count.")
        return

    rows = []
    total_train = total_valid = 0
    for idx in all_indices:
        name = class_names.get(idx, f"class {idx}")
        t = train_counts.get(idx, 0)
        v = valid_counts.get(idx, 0)
        rows.append((name, t, v, t + v))
        total_train += t
        total_valid += v

    rows.append(("TOTAL", total_train, total_valid, total_train + total_valid))

    print(f"\nDataset: {dataset_dir}")
    print(f"Found {len(all_indices)} class(es) across train/valid. Opening table...")

    table_menu(
        "Class Counter - instances per class",
        ["Class", "Train", "Valid", "Total"],
        rows,
        subtitle=dataset_dir,
    )