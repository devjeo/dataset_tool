"""
Helpers for discovering a YOLO dataset's structure:
- which splits exist (train/valid/test/...)
- what the class names are (from data.yaml, falling back to classes.txt)
"""
import os
import yaml

# Folders we never want to offer as a "dataset" choice - the toolkit's own
# code/output folders, and common noise directories.
_IGNORED_DIRS = {"features", "visualized_images", "__pycache__", ".git", ".idea", ".vscode", "venv", ".venv"}


def list_candidate_dataset_dirs(base_dir="."):
    """Return subfolders of base_dir that could plausibly be a dataset root:
    any directory that isn't hidden and isn't one of the toolkit's own folders."""
    candidates = []
    for entry in sorted(os.listdir(base_dir)):
        if entry.startswith("."):
            continue
        if entry in _IGNORED_DIRS:
            continue
        full = os.path.join(base_dir, entry)
        if os.path.isdir(full):
            candidates.append(entry)
    return candidates


def get_dataset_splits(root_dir):
    """Return the subfolders of root_dir that look like dataset splits,
    i.e. contain an 'images' subfolder (e.g. 'train', 'valid', 'test')."""
    splits = []
    if not os.path.isdir(root_dir):
        return splits
    for entry in sorted(os.listdir(root_dir)):
        full = os.path.join(root_dir, entry)
        if os.path.isdir(full) and os.path.isdir(os.path.join(full, "images")):
            splits.append(entry)
    return splits


def find_yaml_file(root_dir):
    """Return the path to the first .yaml/.yml file found directly under root_dir, or None."""
    if not os.path.isdir(root_dir):
        return None
    for f in sorted(os.listdir(root_dir)):
        if f.lower().endswith((".yaml", ".yml")):
            return os.path.join(root_dir, f)
    return None


def load_classes_from_yaml(yaml_path):
    """Parse a YOLO data.yaml file's 'names' field into {class_id(int): class_name(str)}."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    names = data.get("names") if data else None
    if names is None:
        raise ValueError(f"No 'names' field found in {yaml_path}")

    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    if isinstance(names, list):
        return dict(enumerate(names))
    raise ValueError(f"Unrecognized 'names' format in {yaml_path}")


def load_classes_from_txt(txt_path):
    """Fallback: parse a classes.txt file (one class name per line) into {id: name}."""
    with open(txt_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return dict(enumerate(lines))


def load_classes(root_dir, split):
    """Try data.yaml first (preferred), then fall back to <split>/classes.txt.
    Returns {class_id: class_name} or None if nothing could be found."""
    yaml_path = find_yaml_file(root_dir)
    if yaml_path:
        try:
            return load_classes_from_yaml(yaml_path)
        except ValueError as e:
            print(f"Warning: {e}")

    txt_path = os.path.join(root_dir, split, "classes.txt")
    if os.path.exists(txt_path):
        return load_classes_from_txt(txt_path)

    return None