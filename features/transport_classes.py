"""
Feature: Transport / Import Classes

Copies images + labels for a chosen set of classes from a SOURCE dataset
into a DESTINATION dataset, remapping class IDs to match the destination's
class list (creating it if needed) and renaming files sequentially so
nothing collides with what's already in the destination.

Class selection reuses the same green/red include/exclude screen as
Visualize Annotations - whatever you leave green gets transported.

Only folders that already have BOTH a train split and a valid (or val)
split can be picked as the source - anything missing one is filtered out
before it ever shows up as a choice, rather than erroring after selection.

Important: only *selected* classes are ever written to the new label
files. If a label line belongs to a class you didn't select (e.g. a
'backpack' box sitting in a file alongside boxes you DID want), that line
is simply dropped from the transported label - it's not an error, and it
never gets a wrong/mismatched ID in the destination.
"""
import os
import shutil
import glob
import yaml

from dataset_utils import get_dataset_splits, load_classes
from cli_ui import single_select_menu, class_toggle_menu, browse_directory_menu

RENAME_PREFIX = "data_"

# Normalize common alternate split names so a source using 'val' still
# lands in a 'valid' folder in the destination, matching this toolkit's convention.
_SPLIT_NAME_MAP = {"val": "valid"}


def _dest_split_name(split):
    return _SPLIT_NAME_MAP.get(split, split)


def _has_train_and_valid(path):
    """Only accept a folder as a source if it actually has both a train
    split and a valid split (accepting 'val' as an alias for 'valid')."""
    normalized = {_dest_split_name(s) for s in get_dataset_splits(path)}
    return "train" in normalized and "valid" in normalized


def run():
    # 1. Browse to the SOURCE dataset - only folders with BOTH a train and
    #    a valid/val split can be accepted, so incomplete datasets never
    #    even show up as a choice.
    source_dir = browse_directory_menu(
        "Transport / Import Classes",
        start_dir=".",
        accept_predicate=_has_train_and_valid,
        accept_label="Use this folder as SOURCE dataset",
    )
    if source_dir is None:
        print("Cancelled.")
        return

    source_splits = get_dataset_splits(source_dir)

    class_names = load_classes(source_dir, source_splits[0])
    if not class_names:
        print("Could not find class names for the source dataset.")
        return
    source_ids = sorted(class_names.keys())
    source_class_list = [class_names[i] for i in source_ids]

    # 2. Choose which classes to transport (green = included, red = excluded).
    included = class_toggle_menu(source_class_list)
    if included is None:
        print("Cancelled.")
        return
    classes_to_transport = [name for name in source_class_list if included.get(name, True)]
    if not classes_to_transport:
        print("No classes were selected. Nothing to transport.")
        return

    # 3. Browse to the DESTINATION dataset (any folder is fine here, since it
    #    may not exist as a proper dataset yet - it can also be created fresh).
    dest_dir = browse_directory_menu(
        "Transport / Import Classes",
        start_dir=".",
        accept_predicate=None,
        accept_label="Use this folder as DESTINATION",
        allow_create=True,
    )
    if dest_dir is None:
        print("Cancelled.")
        return

    if os.path.abspath(dest_dir) == os.path.abspath(source_dir):
        print("Destination can't be the same folder as the source dataset. Cancelled.")
        return

    print(f"\nSource:      {source_dir}")
    print(f"Destination: {dest_dir}")
    print(f"Classes to transport ({len(classes_to_transport)}): {', '.join(classes_to_transport)}")

    confirm_idx = single_select_menu(
        "Transport / Import Classes",
        ["Proceed (copies files into destination)", "Cancel"],
        subtitle=f"Copy selected classes from '{source_dir}' into '{dest_dir}'?",
    )
    if confirm_idx != 0:
        print("Cancelled. No files were changed.")
        return

    transported, dropped_lines = _transport(
        source_dir, dest_dir, source_splits, class_names, classes_to_transport
    )

    print(f"\nSuccessfully transported {transported} image/label pair(s) into '{dest_dir}'.")
    if dropped_lines:
        print(f"(Dropped {dropped_lines} bounding box line(s) belonging to classes you didn't select - "
              f"this is expected, not an error.)")


def _transport(source_dir, dest_dir, source_splits, source_class_names, classes_to_transport):
    dest_data = _load_or_init_dest_yaml(dest_dir)

    source_name_to_id = {name: cid for cid, name in source_class_names.items()}

    # Build source_id -> dest_id ONLY for the classes we're transporting.
    # Any class not in this dict is, by construction, a class we didn't
    # select - label lines referencing it get dropped later, not errored on.
    id_mapping = {}
    for name in classes_to_transport:
        if name not in source_name_to_id:
            continue
        source_id = source_name_to_id[name]

        if name not in dest_data["names"]:
            dest_data["names"].append(name)
            dest_data["nc"] = len(dest_data["names"])
        dest_id = dest_data["names"].index(name)

        id_mapping[source_id] = dest_id

    if not id_mapping:
        print("None of the selected classes could be matched in the source dataset.")
        return 0, 0

    # Continue numbering from whatever's already in the destination so nothing collides.
    existing_count = 0
    for split in source_splits:
        img_dir = os.path.join(dest_dir, _dest_split_name(split), "images")
        if os.path.isdir(img_dir):
            existing_count += len(glob.glob(os.path.join(img_dir, "*.*")))
    counter = existing_count + 1

    transported = 0
    dropped_lines = 0

    for split in source_splits:
        dsplit = _dest_split_name(split)
        src_img_dir = os.path.join(source_dir, split, "images")
        src_lbl_dir = os.path.join(source_dir, split, "labels")
        dst_img_dir = os.path.join(dest_dir, dsplit, "images")
        dst_lbl_dir = os.path.join(dest_dir, dsplit, "labels")

        if not os.path.isdir(src_lbl_dir):
            continue
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)

        for lbl_file in glob.glob(os.path.join(src_lbl_dir, "*.txt")):
            with open(lbl_file, "r") as f:
                lines = f.readlines()

            valid_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(parts[0])

                if class_id in id_mapping:
                    new_id = id_mapping[class_id]
                    valid_lines.append(f"{new_id} " + " ".join(parts[1:]) + "\n")
                else:
                    dropped_lines += 1

            if not valid_lines:
                continue

            base_name = os.path.splitext(os.path.basename(lbl_file))[0]
            img_path, img_ext = None, None
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
                candidate = os.path.join(src_img_dir, base_name + ext)
                if os.path.exists(candidate):
                    img_path, img_ext = candidate, ext
                    break

            if not img_path:
                print(f"Warning: label found but no matching image for '{base_name}'. Skipped.")
                continue

            new_base = f"{RENAME_PREFIX}{counter:06d}"
            shutil.copy2(img_path, os.path.join(dst_img_dir, new_base + img_ext))
            with open(os.path.join(dst_lbl_dir, new_base + ".txt"), "w") as f:
                f.writelines(valid_lines)

            counter += 1
            transported += 1

    _save_dest_yaml(dest_dir, dest_data)

    return transported, dropped_lines


def _load_or_init_dest_yaml(dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    yaml_path = os.path.join(dest_dir, "data.yaml")

    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}
        names = data.get("names", [])
        if isinstance(names, dict):
            names = [names[i] for i in sorted(names.keys())]
        data["names"] = names
        data.setdefault("nc", len(names))
        data.setdefault("train", "../train/images")
        data.setdefault("val", "../valid/images")
        return data

    return {"train": "../train/images", "val": "../valid/images", "nc": 0, "names": []}


def _save_dest_yaml(dest_dir, data):
    yaml_path = os.path.join(dest_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"train: {data['train']}\n")
        f.write(f"val: {data['val']}\n\n")
        f.write(f"nc: {data['nc']}\n")
        f.write(f"names: {data['names']}\n")