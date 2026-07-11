# YOLO Dataset Toolkit

An interactive, terminal-based CLI for managing, cleaning, and editing YOLO format object detection datasets. 

## Overview
This toolkit provides a centralized menu to run various dataset operations without needing to remember complex command-line arguments or run isolated scripts. It features an interactive text menu (navigated via arrow keys) and is built with a modular architecture so you can easily plug in new tools as your machine learning projects grow.

## Current Features
*   **Visualize Annotations:** Preview bounding boxes directly on your dataset images.
*   **Merge / Clean Classes:** Fix class typos or combine identical categories.
*   **Transport / Import Classes:** Move or import specific classes between different datasets.
*   **Split Dataset:** Automatically divide your images and labels into `train` and `valid` sets.
*   **Annotate / Edit (GUI):** Launch a graphical interface to manually adjust or draw new bounding boxes.
*   **Class Counter:** Generate instance counts across your training and validation splits to check for data imbalances.
*   **Remove / Reduce Classes:** Strip unwanted objects from your labels or consolidate your class list.

## Usage
Ensure you have the required dependencies installed, then run the main script from your terminal:

```bash
python main.py