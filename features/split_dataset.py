"""
Feature: Split into Train/Valid

Takes a "master" dataset that just has flat images/ and labels/ folders
(no train/valid split yet) and randomly divides it into a destination
dataset's train/ and valid/ folders at a chosen ratio.

Differences from a hardcoded script:
- Master and destination folders are found by browsing (descend into
  subfolders, or ".." to go back) instead of being fixed paths - useful
  since a real dataset is rarely sitting right at the top level.
- The split ratio is chosen from presets or entered as a custom percentage.
- The real file extension is preserved when copying images (the original
  approach always renamed to .jpg regardless of source format, which would
  silently mislabel a .png as .jpg).
- Label matching tries "<stem>.txt" first, then "<stem>_mask.txt", so it
  works whether your labels use a "_mask" suffix or not.
- If the destination already has files, new ones continue numbering
  after them instead of overwriting.
- If the master dataset has a data.yaml and the destination doesn't yet
  have one, it's copied over automatically.
"""
import os
import random
import shutil

from dataset_utils import find_yaml_file
from cli_ui import single_select_menu, browse_directory_menu

RATIO_PRESETS = [
    ("80% train / 20% valid", 0.8),
    ("70% train / 30% valid", 0.7),
    ("90% train / 10% valid", 0.9),
    ("Custom...", None),
]


def _is_flat_dataset(path):
    return os.path.isdir(os.path.join(path, "images")) and os.path.isdir(os.path.join(path, "labels"))


def run():
    master_dir = browse_directory_menu(
        "Split into Train/Valid",
        start_dir=".",
        accept_predicate=_is_flat_dataset,
        accept_label="Use this folder as MASTER dataset",
    )
    if master_dir is None:
        print("Cancelled.")
        return

    ratio_labels = [label for label, _ in RATIO_PRESETS]
    ratio_idx = single_select_menu(
        "Split into Train/Valid", ratio_labels, subtitle="Choose the train/valid split ratio:"
    )
    if ratio_idx is None:
        print("Cancelled.")
        return

    train_ratio = RATIO_PRESETS[ratio_idx][1]
    if train_ratio is None:
        train_ratio = _prompt_custom_ratio()
        if train_ratio is None:
            print("Cancelled.")
            return

    dest_dir = browse_directory_menu(
        "Split into Train/Valid",
        start_dir=".",
        accept_predicate=None,  # any folder can be a destination
        accept_label="Use this folder as DESTINATION",
        allow_create=True,
    )
    if dest_dir is None:
        print("Cancelled.")
        return

    if os.path.abspath(dest_dir) == os.path.abspath(master_dir):
        print("Destination can't be the same folder as the master dataset. Cancelled.")
        return

    print(f"\nMaster:      {master_dir}")
    print(f"Destination: {dest_dir}")
    print(f"Split ratio: {round(train_ratio * 100)}% train / {round((1 - train_ratio) * 100)}% valid")

    confirm_idx = single_select_menu(
        "Split into Train/Valid",
        ["Proceed (copies files into destination)", "Cancel"],
        subtitle=f"Split '{master_dir}' into 'train' and 'valid' inside '{dest_dir}'?",
    )
    if confirm_idx != 0:
        print("Cancelled. No files were changed.")
        return

    train_count, valid_count, skipped = _split_dataset(master_dir, dest_dir, train_ratio)

    print(f"\nDone! Train: {train_count} image(s), Valid: {valid_count} image(s).")
    if skipped:
        print(f"Skipped {skipped} image(s) with no matching label file.")


def _prompt_custom_ratio():
    raw = input("Enter train percentage (e.g. 85 for 85% train / 15% valid): ").strip()
    try:
        pct = float(raw)
    except ValueError:
        print("Not a number.")
        return None
    if not (0 < pct < 100):
        print("Percentage must be between 0 and 100.")
        return None
    return pct / 100.0


def _find_label_for_image(img_path, labels_dir):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    for candidate_name in (f"{stem}.txt", f"{stem}_mask.txt"):
        candidate = os.path.join(labels_dir, candidate_name)
        if os.path.exists(candidate):
            return candidate
    return None


def _next_start_index(dest_images_dir):
    """Look at existing numeric filenames so new copies don't overwrite anything already there."""
    if not os.path.isdir(dest_images_dir):
        return 0
    max_idx = -1
    for f in os.listdir(dest_images_dir):
        stem = os.path.splitext(f)[0]
        if stem.isdigit():
            max_idx = max(max_idx, int(stem))
    return max_idx + 1


def _copy_split(image_list, master_labels_dir, dest_images_dir, dest_labels_dir, start_index):
    os.makedirs(dest_images_dir, exist_ok=True)
    os.makedirs(dest_labels_dir, exist_ok=True)

    count = start_index
    skipped = 0
    for img_path in image_list:
        label_path = _find_label_for_image(img_path, master_labels_dir)
        if label_path is None:
            print(f"  [Warning] Skipping {os.path.basename(img_path)} - no matching label found.")
            skipped += 1
            continue

        ext = os.path.splitext(img_path)[1]  # preserve the real extension
        shutil.copy2(img_path, os.path.join(dest_images_dir, f"{count}{ext}"))
        shutil.copy2(label_path, os.path.join(dest_labels_dir, f"{count}.txt"))
        count += 1

    return count - start_index, skipped


def _split_dataset(master_dir, dest_dir, train_ratio):
    master_images_dir = os.path.join(master_dir, "images")
    master_labels_dir = os.path.join(master_dir, "labels")

    all_images = [
        os.path.join(master_images_dir, f)
        for f in os.listdir(master_images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    random.shuffle(all_images)

    split_index = int(len(all_images) * train_ratio)
    train_images = all_images[:split_index]
    valid_images = all_images[split_index:]

    train_images_dir = os.path.join(dest_dir, "train", "images")
    train_labels_dir = os.path.join(dest_dir, "train", "labels")
    valid_images_dir = os.path.join(dest_dir, "valid", "images")
    valid_labels_dir = os.path.join(dest_dir, "valid", "labels")

    train_start = _next_start_index(train_images_dir)
    valid_start = _next_start_index(valid_images_dir)

    print(f"Copying {len(train_images)} file(s) to train...")
    train_count, train_skipped = _copy_split(
        train_images, master_labels_dir, train_images_dir, train_labels_dir, train_start
    )

    print(f"Copying {len(valid_images)} file(s) to valid...")
    valid_count, valid_skipped = _copy_split(
        valid_images, master_labels_dir, valid_images_dir, valid_labels_dir, valid_start
    )

    _copy_yaml_if_missing(master_dir, dest_dir)

    return train_count, valid_count, train_skipped + valid_skipped


def _copy_yaml_if_missing(master_dir, dest_dir):
    src_yaml = find_yaml_file(master_dir)
    if not src_yaml:
        return
    if find_yaml_file(dest_dir):
        return  # don't clobber an existing destination data.yaml
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy2(src_yaml, os.path.join(dest_dir, os.path.basename(src_yaml)))