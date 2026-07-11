"""
Feature: Visualize Annotations

Draws YOLO-format bounding boxes onto images so you can visually check
your labels. Lets you pick the split (train/valid/...) and which classes
to include, both via the interactive CLI menus in cli_ui.py.
"""
import os
import cv2
import shutil

from dataset_utils import list_candidate_dataset_dirs, get_dataset_splits, load_classes
from cli_ui import single_select_menu, class_toggle_menu

OUTPUT_ROOT = "visualized_images"

BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 0, 0)


def run():
    # 1. Pick the dataset folder itself from whatever folders exist here.
    dataset_options = list_candidate_dataset_dirs(".")
    if not dataset_options:
        print("No folders found in the current directory to use as a dataset.")
        return

    dataset_idx = single_select_menu(
        "Visualize Annotations", dataset_options, subtitle="Select your dataset folder:"
    )
    if dataset_idx is None:
        print("Cancelled.")
        return
    dataset_root = dataset_options[dataset_idx]

    # 2. Pick the split (train/valid/...) inside that dataset folder.
    splits = get_dataset_splits(dataset_root)
    if not splits:
        print(f"No splits found under '{dataset_root}' "
              f"(expected e.g. train/ or valid/ containing an 'images' folder).")
        return

    split_idx = single_select_menu(
        "Visualize Annotations", splits, subtitle=f"Select a split inside '{dataset_root}':"
    )
    if split_idx is None:
        print("Cancelled.")
        return
    split = splits[split_idx]

    images_dir = os.path.join(dataset_root, split, "images")
    labels_dir = os.path.join(dataset_root, split, "labels")

    class_names = load_classes(dataset_root, split)
    if not class_names:
        print("Could not find class names (looked for a data.yaml in "
              f"'{dataset_root}' and a classes.txt in '{split}/').")
        return

    class_list = [class_names[i] for i in sorted(class_names.keys())]
    included = class_toggle_menu(class_list)
    if included is None:
        print("Cancelled.")
        return

    excluded_ids = {cid for cid, name in class_names.items() if not included.get(name, True)}

    output_dir = os.path.join(OUTPUT_ROOT, split)
    count = _draw_yolo_labels(images_dir, labels_dir, output_dir, class_names, excluded_ids)

    print(f"\nDone! Drew boxes on {count} image(s).")
    print(f"Check the '{output_dir}' folder to review them.")


def _draw_yolo_labels(images_dir, labels_dir, output_dir, class_names, excluded_ids):
    # Clear the specific output directory for this split if it already exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    os.makedirs(output_dir, exist_ok=True)
    processed_count = 0

    if not os.path.isdir(images_dir):
        print(f"Images folder not found: {images_dir}")
        return 0

    # Pre-filter valid images to know the exact total for the progress bar
    all_files = os.listdir(images_dir)
    valid_images = [f for f in all_files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    total_images = len(valid_images)

    if total_images == 0:
        print("No images found to process.")
        return 0

    print() # Add a newline before the progress bar starts

    for i, filename in enumerate(valid_images):
        # --- Native Progress Bar Logic ---
        percent = (i + 1) / total_images
        bar_length = 40
        filled_length = int(bar_length * percent)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(f'\rProcessing Images: |{bar}| {percent:.1%} ({i+1}/{total_images})', end='', flush=True)
        # ---------------------------------

        img_path = os.path.join(images_dir, filename)
        label_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + ".txt")
        
        if not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        img_height, img_width = img.shape[:2]

        with open(label_path, "r") as f:
            lines = f.readlines()

        drawn_any = False
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            if class_id in excluded_ids or class_id not in class_names:
                continue

            drawn_any = True

            x_center_norm, y_center_norm, width_norm, height_norm = map(float, parts[1:5])
            w_pixels = int(width_norm * img_width)
            h_pixels = int(height_norm * img_height)
            x_center_pixels = int(x_center_norm * img_width)
            y_center_pixels = int(y_center_norm * img_height)

            x_min = int(x_center_pixels - w_pixels / 2)
            y_min = int(y_center_pixels - h_pixels / 2)
            x_max = int(x_center_pixels + w_pixels / 2)
            y_max = int(y_center_pixels + h_pixels / 2)

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), BOX_COLOR, 2)

            label_text = class_names[class_id]
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(
                img,
                (x_min, y_min - text_size[1] - 5),
                (x_min + text_size[0], y_min),
                BOX_COLOR,
                -1,
            )
            cv2.putText(
                img, label_text, (x_min, y_min - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 2,
            )

        if drawn_any:
            cv2.imwrite(os.path.join(output_dir, filename), img)
            processed_count += 1

    print() # Add a final newline so subsequent prints don't overwrite the full bar
    return processed_count