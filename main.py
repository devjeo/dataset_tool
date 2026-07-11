"""
YOLO Dataset Toolkit - interactive CLI

Run:
    python main.py

A simple text menu (arrow keys + Enter) lets you pick a feature. To add a
new feature:
    1. Create a module in features/ with a `run()` function.
    2. Import it below and add ("Display Name", module.run) to FEATURES.
"""
from cli_ui import single_select_menu
from features import visualize_annotations
from features import merge_classes
from features import transport_classes
from features import split_dataset
from features import annotate_gui
from features import class_counter
from features import remove_class
from features import reduce_class


def _not_implemented():
    print("This feature isn't implemented yet. Add it in features/ and register it in main.py!")


FEATURES = [
    ("Visualize Annotations", visualize_annotations.run),
    ("Merge / Clean Classes", merge_classes.run),
    ("Transport / Import Classes", transport_classes.run),
    ("Split into Train/Valid", split_dataset.run),
    ("Annotate / Edit Boxes (GUI)", annotate_gui.run),
    ("Class Counter (train/valid instance counts)", class_counter.run),
    ("Remove a Class", remove_class.run),
    ("Reduce Classes (merge into fewer classes)", reduce_class.run),
    ("Dataset Statistics (coming soon)", _not_implemented),
    ("Exit", None),
]


def main():
    while True:
        labels = [name for name, _ in FEATURES]
        idx = single_select_menu("YOLO Dataset Toolkit", labels, subtitle="Select a feature:")
        if idx is None:
            break

        name, func = FEATURES[idx]
        if name == "Exit":
            break

        print(f"\n=== {name} ===\n")
        func()
        input("\nPress Enter to return to the menu...")

    print("Goodbye!")


if __name__ == "__main__":
    main()