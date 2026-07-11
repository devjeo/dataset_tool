"""
Feature: Remove a Class

Permanently removes one class from a dataset:
- Every label line for that class is dropped from every split.
- Any image whose labels ONLY referenced that class is deleted entirely
  (both the image and its now-empty label file), since it has nothing
  left to train on.
- Any remaining class with a higher ID shifts down by 1 to close the gap,
  so the class list stays dense (0..N-1) - matching the compacted list
  written back to data.yaml/classes.txt at the end.

Differences from a hardcoded script:
- The dataset folder is found by browsing (not a fixed path).
- The class to remove is picked from a live list read from the dataset's
  own data.yaml/classes.txt, not typed in by hand.
- Splits are auto-detected (whatever folders have an 'images' subfolder),
  instead of a fixed ['train', 'valid', 'val'] list.
- There's a confirmation step before anything is deleted, since this is
  destructive and can't be undone.
"""
import os
import glob
import yaml

from dataset_utils import get_dataset_splits, load_classes, find_yaml_file
from cli_ui import single_select_menu, browse_directory_menu


def run():
    dataset_dir = browse_directory_menu(
        "Remove a Class",
        start_dir=".",
        accept_predicate=lambda p: bool(get_dataset_splits(p)),
        accept_label="Use this folder as dataset",
    )
    if dataset_dir is None:
        print("Cancelled.")
        return

    splits = get_dataset_splits(dataset_dir)
    class_names = load_classes(dataset_dir, splits[0])
    if not class_names:
        print("Could not find class names (looked for a data.yaml and a classes.txt).")
        return

    ids_sorted = sorted(class_names.keys())
    class_list = [class_names[i] for i in ids_sorted]

    class_idx = single_select_menu(
        "Remove a Class", class_list, subtitle="Select the class to remove:"
    )
    if class_idx is None:
        print("Cancelled.")
        return

    target_name = class_list[class_idx]
    target_id = ids_sorted[class_idx]

    print(f"\nThis will permanently remove class '{target_name}' (id {target_id}) from '{dataset_dir}':")
    print("  - Every label line for this class is dropped, in every split.")
    print("  - Any image whose boxes ONLY belonged to this class is deleted "
          "(image + label file).")
    print("  - Remaining class IDs above it shift down by 1 to close the gap.")

    confirm_idx = single_select_menu(
        "Remove a Class",
        ["Proceed (this deletes files - cannot be undone)", "Cancel"],
        subtitle=f"Remove '{target_name}' from '{dataset_dir}'?",
    )
    if confirm_idx != 0:
        print("Cancelled. No files were changed.")
        return

    removed_images, modified_labels = _remove_class_files(dataset_dir, splits, target_id)

    new_names = [name for i, name in enumerate(class_list) if i != class_idx]
    _update_class_definitions(dataset_dir, new_names)

    print(f"\nDone. Removed {removed_images} image(s) (and their labels) that ONLY contained '{target_name}'.")
    print(f"Modified {modified_labels} label file(s) to remove the class and shift remaining IDs.")


def _remove_class_files(dataset_dir, splits, target_id):
    removed_images = 0
    modified_labels = 0

    for split in splits:
        lbl_dir = os.path.join(dataset_dir, split, "labels")
        img_dir = os.path.join(dataset_dir, split, "images")
        if not os.path.isdir(lbl_dir):
            continue

        for lbl_file in glob.glob(os.path.join(lbl_dir, "*.txt")):
            with open(lbl_file, "r") as f:
                lines = f.readlines()

            new_lines = []
            changed = False
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(parts[0])

                if class_id == target_id:
                    changed = True
                    continue  # drop this box - it belongs to the removed class
                elif class_id > target_id:
                    parts[0] = str(class_id - 1)  # shift down to close the gap
                    new_lines.append(" ".join(parts) + "\n")
                    changed = True
                else:
                    new_lines.append(line if line.endswith("\n") else line + "\n")

            if not changed:
                continue  # this file never referenced the removed class - leave it alone

            if not new_lines:
                # every box in this file belonged to the removed class - nothing left to keep
                os.remove(lbl_file)
                base_name = os.path.splitext(os.path.basename(lbl_file))[0]
                for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
                    img_path = os.path.join(img_dir, base_name + ext)
                    if os.path.exists(img_path):
                        os.remove(img_path)
                        removed_images += 1
                        break
            else:
                with open(lbl_file, "w") as f:
                    f.writelines(new_lines)
                modified_labels += 1

    return removed_images, modified_labels


def _update_class_definitions(dataset_dir, new_names):
    yaml_path = find_yaml_file(dataset_dir)
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
    for split_dir in glob.glob(os.path.join(dataset_dir, "*")):
        txt_path = os.path.join(split_dir, "classes.txt")
        if os.path.exists(txt_path):
            with open(txt_path, "w") as f:
                f.write("\n".join(new_names) + "\n")