"""
Feature: Merge / Clean Classes

Collapses messy/duplicate class names (e.g. "Chair", "chasir", "chair-")
down to a single clean class, remaps every label file's class IDs to match,
and updates the dataset's class definitions (data.yaml or classes.txt).

Unlike a hardcoded script, this feature:
- Pulls `old_names` automatically from whichever dataset folder you select
  (from its data.yaml / classes.txt) - you never type them in by hand.
- Loads the merge map from an external `merge_map.json` file that lives in
  the dataset folder, so you can edit it in any text editor without
  touching this code. If it doesn't exist yet, a starter template is
  generated for you (identity mapping - every class maps to itself) so you
  only need to edit the classes you actually want merged.
"""
import os
import json
import glob
import yaml

from dataset_utils import (
    list_candidate_dataset_dirs,
    get_dataset_splits,
    find_yaml_file,
    load_classes,
)
from cli_ui import single_select_menu

MERGE_MAP_FILENAME = "/../features/merge_map.json"


def run():
    dataset_options = list_candidate_dataset_dirs(".")
    if not dataset_options:
        print("No folders found in the current directory to use as a dataset.")
        return

    dataset_idx = single_select_menu(
        "Merge / Clean Classes", dataset_options, subtitle="Select your dataset folder:"
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

    # old_names = whatever classes this dataset already has (yaml, falling back to classes.txt)
    class_names = load_classes(dataset_root, splits[0])
    if not class_names:
        print("Could not find class names (looked for a data.yaml and a classes.txt).")
        return
    old_ids = sorted(class_names.keys())
    old_names = [class_names[i] for i in old_ids]

    merge_map_path = os.path.join(dataset_root, MERGE_MAP_FILENAME)
    if not os.path.exists(merge_map_path):
        _write_template_merge_map(merge_map_path, old_names)
        print(f"\nNo '{MERGE_MAP_FILENAME}' found for this dataset, so I created a starter one at:")
        print(f"  {merge_map_path}")
        print("\nEvery class currently maps to itself. Open the file and edit the *values* "
              "for any classes you want merged together, e.g.:")
        print('  "chasir": "chair",')
        print('  "sofaChair": "chair",')
        print("\nThen run this feature again to apply the merge.")
        return

    try:
        with open(merge_map_path, "r") as f:
            merge_map = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Could not parse '{merge_map_path}': {e}")
        return

    new_names = []
    for name in old_names:
        clean_name = merge_map.get(name, name)
        if clean_name not in new_names:
            new_names.append(clean_name)

    id_translator = {}
    for old_id, name in zip(old_ids, old_names):
        clean_name = merge_map.get(name, name)
        id_translator[old_id] = new_names.index(clean_name)

    print(f"\nThis will reduce classes from {len(old_names)} down to {len(new_names)}:")
    for old_id, name in zip(old_ids, old_names):
        new_id = id_translator[old_id]
        marker = "  ->" if new_names[new_id] != name else "  =="
        print(f"  [{old_id:>2}] {name:<20s} {marker} [{new_id:>2}] {new_names[new_id]}")

    confirm_idx = single_select_menu(
        "Merge / Clean Classes",
        ["Proceed (rewrites label files in place)", "Cancel"],
        subtitle=f"This will modify label files under '{dataset_root}'.",
    )
    if confirm_idx != 0:
        print("Cancelled. No files were changed.")
        return

    files_modified = _rewrite_label_files(dataset_root, splits, id_translator)
    _update_class_definitions(dataset_root, new_names)

    print(f"\nSuccess! Updated {files_modified} label file(s) and saved the new class list.")


def _write_template_merge_map(path, old_names):
    template = {name: name for name in old_names}
    with open(path, "w") as f:
        json.dump(template, f, indent=2)


def _rewrite_label_files(dataset_root, splits, id_translator):
    files_modified = 0
    for split in splits:
        lbl_dir = os.path.join(dataset_root, split, "labels")
        if not os.path.isdir(lbl_dir):
            continue

        for lbl_file in glob.glob(os.path.join(lbl_dir, "*.txt")):
            with open(lbl_file, "r") as f:
                lines = f.readlines()

            new_lines = []
            file_changed = False
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue

                old_id = int(parts[0])
                if old_id in id_translator:
                    new_id = id_translator[old_id]
                    if new_id != old_id:
                        parts[0] = str(new_id)
                        file_changed = True

                new_lines.append(" ".join(parts) + "\n")

            if file_changed:
                with open(lbl_file, "w") as f:
                    f.writelines(new_lines)
                files_modified += 1

    return files_modified


def _update_class_definitions(dataset_root, new_names):
    yaml_path = find_yaml_file(dataset_root)
    if yaml_path:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}

        data["nc"] = len(new_names)
        data["names"] = new_names

        with open(yaml_path, "w") as f:
            if "train" in data:
                f.write(f"train: {data['train']}\n")
            if "val" in data:
                f.write(f"val: {data['val']}\n\n")
            f.write(f"nc: {data['nc']}\n")
            f.write(f"names: {data['names']}\n")
        return

    # No data.yaml - fall back to updating any classes.txt found in the dataset's splits
    for split_dir in glob.glob(os.path.join(dataset_root, "*")):
        txt_path = os.path.join(split_dir, "classes.txt")
        if os.path.exists(txt_path):
            with open(txt_path, "w") as f:
                f.write("\n".join(new_names) + "\n")