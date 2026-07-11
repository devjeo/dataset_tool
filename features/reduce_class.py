"""
Feature: Reduce Class Instances

Smart undersampling for imbalanced YOLO datasets. Removes *individual
bounding boxes* of an overrepresented class instead of deleting whole
images, so co-occurring rare classes in the same image are never touched.

Boxes are ranked for removal using quality heuristics:
    1. Edge-touching / truncated boxes are removed first (usually the
       lowest-quality, most ambiguous annotations).
    2. Among remaining candidates, the smallest boxes are removed first
       (tiny/far-away/blurry instances).
    3. Ties are broken by preferring removal from images that already
       contain many instances of the target class (redundant, dense
       scenes), which protects sparser / more diverse images.

Like merge_classes.py, this walks candidate dataset folders via the arrow
key menus instead of asking you to type a path, and always backs up label
folders before writing anything.
"""
import os
import shutil
import time

from dataset_utils import (
    list_candidate_dataset_dirs,
    get_dataset_splits,
    load_classes,
)
from cli_ui import single_select_menu, table_menu

EDGE_MARGIN = 0.01  # normalized distance from 0/1 considered "touching the edge"


def run():
    dataset_options = list_candidate_dataset_dirs(".")
    if not dataset_options:
        print("No folders found in the current directory to use as a dataset.")
        return

    dataset_idx = single_select_menu(
        "Reduce Class Instances", dataset_options, subtitle="Select your dataset folder:"
    )
    if dataset_idx is None:
        print("Cancelled.")
        return
    dataset_root = dataset_options[dataset_idx]

    splits = get_dataset_splits(dataset_root)
    if not splits:
        print(f"No splits found under '{dataset_root}' "
              f"(expected e.g. train/ or valid/ containing an 'images' folder).")
        return

    class_names = load_classes(dataset_root, splits[0])
    if not class_names:
        print("Could not find class names (looked for a data.yaml and a classes.txt).")
        return
    class_ids = sorted(class_names.keys())

    class_idx = single_select_menu(
        "Reduce Class Instances",
        [f"[{cid}] {class_names[cid]}" for cid in class_ids],
        subtitle="Which class do you want to reduce?",
    )
    if class_idx is None:
        print("Cancelled.")
        return
    target_id = class_ids[class_idx]
    target_name = class_names[target_id]

    split_options = list(splits) + (["All splits"] if len(splits) > 1 else [])
    split_idx = single_select_menu(
        "Reduce Class Instances", split_options, subtitle=f"Reduce '{target_name}' in which split?"
    )
    if split_idx is None:
        print("Cancelled.")
        return
    chosen_splits = splits if split_options[split_idx] == "All splits" else [split_options[split_idx]]

    label_dirs = [os.path.join(dataset_root, split, "labels") for split in chosen_splits]
    label_dirs = [d for d in label_dirs if os.path.isdir(d)]
    if not label_dirs:
        print("No label folders found for the selected split(s).")
        return

    all_candidates = []
    file_caches = {}
    for label_dir in label_dirs:
        cands, cache = _collect_candidates(label_dir, target_id)
        all_candidates.extend(cands)
        file_caches[label_dir] = cache

    total_current = len(all_candidates)
    if total_current == 0:
        print(f"No instances of '{target_name}' found in the selected split(s).")
        return

    mode_idx = single_select_menu(
        "Reduce Class Instances",
        ["Remove an exact number", "Remove a percentage", "Reduce down to a target total"],
        subtitle=f"Current '{target_name}' count: {total_current}. How do you want to specify the reduction?",
    )
    if mode_idx is None:
        print("Cancelled.")
        return

    try:
        if mode_idx == 0:
            remove_n = int(input("Number of instances to remove: ").strip())
        elif mode_idx == 1:
            pct = float(input("Percentage to remove (e.g. 40): ").strip())
            remove_n = int(round(total_current * pct / 100))
        else:
            target_total = int(input("Target total instance count: ").strip())
            remove_n = max(0, total_current - target_total)
    except ValueError:
        print("Invalid number entered. Cancelled.")
        return

    remove_n = min(remove_n, total_current)
    if remove_n <= 0:
        print("Nothing to remove.")
        return

    ranked = _rank_for_removal(all_candidates)
    to_remove = ranked[:remove_n]
    edge_removed = sum(1 for c in to_remove if c["edge"])

    table_menu(
        "Reduce Class Instances - Review",
        ["Metric", "Value"],
        [
            ("Class", f"[{target_id}] {target_name}"),
            ("Current count", total_current),
            ("Edge-touching removed", edge_removed),
            ("Small (non-edge) removed", remove_n - edge_removed),
            ("Total to remove", remove_n),
            ("Resulting count", total_current - remove_n),
        ],
        subtitle="Press Enter/q to continue",
    )

    confirm_idx = single_select_menu(
        "Reduce Class Instances",
        ["Proceed (backs up label folders, then rewrites in place)", "Cancel"],
        subtitle=f"This will modify label files under '{dataset_root}'.",
    )
    if confirm_idx != 0:
        print("Cancelled. No files were changed.")
        return

    backups = {}
    for label_dir in label_dirs:
        backups[label_dir] = _backup_dir(label_dir)
        print(f"Backed up {label_dir} -> {backups[label_dir]}")

    removals_by_file = {}
    for c in to_remove:
        removals_by_file.setdefault(c["path"], []).append(c["raw"])

    files_touched = 0
    for label_dir, cache in file_caches.items():
        for path, boxes in cache.items():
            if path not in removals_by_file:
                continue
            to_drop = list(removals_by_file[path])
            kept_lines = []
            for cid, x, y, w, h, raw in boxes:
                if raw in to_drop:
                    to_drop.remove(raw)
                    continue
                kept_lines.append(raw)
            with open(path, "w") as f:
                f.write("\n".join(kept_lines) + ("\n" if kept_lines else ""))
            files_touched += 1

    print(f"\nSuccess! Updated {files_touched} label file(s).")
    print(f"'{target_name}' instance count: {total_current} -> {total_current - remove_n}")
    print("If anything looks wrong, restore from the backup folder(s) listed above.")


# ---------------------------------------------------------------------
# Scoring / file helpers
# ---------------------------------------------------------------------
def _parse_label_file(path):
    """Return list of (class_id:int, x:float, y:float, w:float, h:float, raw_line:str)."""
    boxes = []
    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(float(parts[0]))
                x, y, w, h = (float(v) for v in parts[1:5])
            except ValueError:
                continue
            boxes.append((cid, x, y, w, h, stripped))
    return boxes


def _touches_edge(x, y, w, h):
    xmin, xmax = x - w / 2, x + w / 2
    ymin, ymax = y - h / 2, y + h / 2
    return xmin <= EDGE_MARGIN or ymin <= EDGE_MARGIN or xmax >= 1 - EDGE_MARGIN or ymax >= 1 - EDGE_MARGIN


def _collect_candidates(label_dir, target_id):
    """
    Walk every label file, return:
      candidates: list of dicts describing each target-class box
      file_cache: {path: [full list of parsed boxes]} for later rewriting
    """
    candidates = []
    file_cache = {}

    for fname in os.listdir(label_dir):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(label_dir, fname)
        boxes = _parse_label_file(path)
        file_cache[path] = boxes

        target_boxes = [b for b in boxes if b[0] == target_id]
        density = len(target_boxes)  # how many target-class instances in this image

        for cid, x, y, w, h, raw in target_boxes:
            candidates.append({
                "path": path,
                "raw": raw,
                "area": w * h,
                "edge": _touches_edge(x, y, w, h),
                "density": density,
            })

    return candidates, file_cache


def _rank_for_removal(candidates):
    # Remove edge-touching first, then smallest area, then densest image first.
    return sorted(
        candidates,
        key=lambda c: (not c["edge"], c["area"], -c["density"]),
    )


def _backup_dir(label_dir):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = label_dir.rstrip("/\\") + f"_backup_{stamp}"
    shutil.copytree(label_dir, backup_path)
    return backup_path