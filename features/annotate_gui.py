"""
Feature: Annotate / Edit Boxes (GUI)

Opens a small Tkinter window for a chosen split (train/valid) where you can:
- Navigate every image with the Left/Right arrow keys
- See existing bounding boxes drawn over the image
- Click a class button at the bottom to make it "active" (highlighted), then
  click-drag on an empty area of the canvas to draw a new box - it is
  assigned to the active class automatically as soon as you release the
  mouse. (You can also draw first and click a class afterwards if you'd
  rather do it that way - both work.)
- Click a box to select it (highlighted yellow). While selected you can:
    - drag inside it to move it
    - drag one of the small square handles on its edges/corners to resize it
    - press Delete/Backspace or click "Delete selected box" to remove it
- Every change (add/move/resize/delete) is written to the label .txt file
  immediately - there's no separate Save step, so nothing is lost when you
  navigate to another image or close the window.

This is a different UI toolkit than the rest of the CLI (Tkinter instead of
curses), since drawing/clicking on an actual image needs real graphics and
mouse input that a terminal can't provide. The CLI's curses menus are used
to pick the dataset/split first, then this window takes over; closing the
window returns you to the CLI menu.
"""
import os
import yaml

from dataset_utils import get_dataset_splits, load_classes, find_yaml_file
from cli_ui import single_select_menu, browse_directory_menu

CANVAS_WIDTH = 900
CANVAS_HEIGHT = 650
BOX_COLOR = "#00e676"        # green - existing box
SELECTED_COLOR = "#ffeb3b"   # yellow - selected box
PENDING_COLOR = "#00b0ff"    # blue (dashed) - box being drawn, awaiting a class


def run():
    dataset_dir = browse_directory_menu(
        "Annotate / Edit Boxes",
        start_dir=".",
        accept_predicate=lambda p: bool(get_dataset_splits(p)),
        accept_label="Use this folder as dataset",
    )
    if dataset_dir is None:
        print("Cancelled.")
        return

    splits = get_dataset_splits(dataset_dir)
    split_idx = single_select_menu(
        "Annotate / Edit Boxes", splits, subtitle="Select a split to annotate:"
    )
    if split_idx is None:
        print("Cancelled.")
        return
    split = splits[split_idx]

    images_dir = os.path.join(dataset_dir, split, "images")
    labels_dir = os.path.join(dataset_dir, split, "labels")

    if not os.path.isdir(images_dir):
        print(f"Images folder not found: {images_dir}")
        return

    image_files = sorted(
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not image_files:
        print("No images found in this split.")
        return

    os.makedirs(labels_dir, exist_ok=True)

    class_names = load_classes(dataset_dir, split) or {}
    class_list = [class_names[i] for i in sorted(class_names.keys())]

    print(f"\nOpening annotator for {len(image_files)} image(s) in '{split}'...")
    print("(Close the window to return to the menu.)")

    try:
        import tkinter  # noqa: F401
    except ImportError:
        print("Tkinter isn't available in this Python installation. "
              "On Linux, install it with: sudo apt install python3-tk")
        return

    app = AnnotatorApp(dataset_dir, split, images_dir, labels_dir, image_files, class_list)
    app.run()


class AnnotatorApp:
    def __init__(self, dataset_dir, split, images_dir, labels_dir, image_files, class_list):
        import tkinter as tk

        self.dataset_dir = dataset_dir
        self.split = split
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_files = image_files
        self.class_list = class_list  # list of class names; index = class id

        self.index = 0
        self.boxes = []           # list of {class_id, cx, cy, w, h} (normalized 0-1)
        self.selected_box = None  # index into self.boxes
        self.pending_box = None   # (x1, y1, x2, y2) in canvas coords, awaiting a class
        self.drag_start = None

        self.active_class_id = None   # class chosen via the buttons; new boxes use this
        self.class_buttons = {}       # class_id -> Button widget
        self.edit_mode = None         # None | 'draw' | 'move' | 'resize'
        self.edit_handle = None       # which handle is being dragged, when resizing
        self.edit_orig_box = None     # canvas coords (x1,y1,x2,y2) at drag start
        self.edit_start_pt = None     # (x, y) canvas point where the drag started

        self.tk_image = None
        self.img_w = 0
        self.img_h = 0
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self.root = tk.Tk()
        self.root.title("Annotate / Edit Boxes")
        self._build_ui()
        self._load_current_image()

    def run(self):
        self.root.mainloop()

    # ------------------------------------------------------------------
    def _build_ui(self):
        import tkinter as tk

        self.title_label = tk.Label(self.root, text="", font=("Segoe UI", 11, "bold"))
        self.title_label.pack(pady=(8, 2))

        self.canvas = tk.Canvas(self.root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="#222222")
        self.canvas.pack(padx=8, pady=4)

        self.hint_label = tk.Label(
            self.root,
            text="Click a class below to make it active, then drag on empty area: draw new box    "
                 "Click a box: select    Drag inside selected: move    Drag a handle: resize    "
                 "Delete/Backspace: remove selected    Left/Right: navigate    Esc: cancel draw",
            font=("Segoe UI", 9), fg="#555555", wraplength=CANVAS_WIDTH,
        )
        self.hint_label.pack(pady=(0, 4))

        self.class_frame = tk.Frame(self.root)
        self.class_frame.pack(pady=(0, 4), fill="x")
        self._rebuild_class_buttons()

        controls_frame = tk.Frame(self.root)
        controls_frame.pack(pady=(0, 8))
        self.new_class_entry = tk.Entry(controls_frame, width=20)
        self.new_class_entry.pack(side="left", padx=4)
        tk.Button(controls_frame, text="Add new class", command=self._on_add_new_class).pack(side="left")
        tk.Button(controls_frame, text="Delete selected box", command=self._delete_selected).pack(
            side="left", padx=(12, 0)
        )

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Motion>", self._on_motion)

        self.root.bind("<Left>", lambda e: self._navigate(-1))
        self.root.bind("<Right>", lambda e: self._navigate(1))
        self.root.bind("<Delete>", lambda e: self._delete_selected())
        self.root.bind("<BackSpace>", lambda e: self._delete_selected())
        self.root.bind("<Escape>", lambda e: self._cancel_pending())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _rebuild_class_buttons(self):
        import tkinter as tk

        for widget in self.class_frame.winfo_children():
            widget.destroy()
        self.class_buttons = {}
        if not self.class_list:
            tk.Label(self.class_frame, text="(no classes yet - add one below)", fg="#888888").pack(side="left", padx=4)
            return
        for i, name in enumerate(self.class_list):
            btn = tk.Button(
                self.class_frame, text=name, relief="raised",
                command=lambda cid=i: self._on_class_clicked(cid),
            )
            btn.pack(side="left", padx=3, pady=2)
            self.class_buttons[i] = btn
        self._update_class_button_styles()

    def _update_class_button_styles(self):
        for cid, btn in self.class_buttons.items():
            if cid == self.active_class_id:
                btn.config(relief="sunken", bg=PENDING_COLOR, fg="#ffffff")
            else:
                btn.config(relief="raised", bg="SystemButtonFace", fg="#000000")

    # ------------------------------------------------------------------
    def _label_path_for(self, filename):
        return os.path.join(self.labels_dir, os.path.splitext(filename)[0] + ".txt")

    def _load_current_image(self):
        from PIL import Image, ImageTk, ImageOps

        filename = self.image_files[self.index]

        img_path = os.path.join(self.images_dir, filename)

        pil_img = Image.open(img_path)
        # Many phone/camera photos carry an EXIF orientation tag instead of
        # actually being stored rotated. PIL ignores that tag by default, so
        # without this the canvas can show the image "sideways" relative to
        # the pixel grid the label coordinates were written against - boxes
        # then look like they're in the wrong place. exif_transpose bakes
        # the rotation/flip into the pixels so what you see is what the
        # normalized coordinates actually refer to.
        pil_img = ImageOps.exif_transpose(pil_img)
        pil_img = pil_img.convert("RGB")
        self.img_w, self.img_h = pil_img.size

        self.scale = min(CANVAS_WIDTH / self.img_w, CANVAS_HEIGHT / self.img_h)
        disp_w, disp_h = max(1, int(self.img_w * self.scale)), max(1, int(self.img_h * self.scale))
        self.offset_x = (CANVAS_WIDTH - disp_w) // 2
        self.offset_y = (CANVAS_HEIGHT - disp_h) // 2

        resized = pil_img.resize((disp_w, disp_h), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)

        self.boxes = self._read_labels(filename)
        self.selected_box = None
        self.pending_box = None
        self.edit_mode = None
        self.edit_handle = None

        self.title_label.config(text=f"[{self.split}] {filename}  ({self.index + 1}/{len(self.image_files)})")
        self._redraw()

    def _read_labels(self, filename):
        label_path = self._label_path_for(filename)
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    boxes.append({
                        "class_id": int(parts[0]),
                        "cx": float(parts[1]), "cy": float(parts[2]),
                        "w": float(parts[3]), "h": float(parts[4]),
                    })
        return boxes

    def _save_labels(self):
        filename = self.image_files[self.index]
        label_path = self._label_path_for(filename)
        with open(label_path, "w") as f:
            for b in self.boxes:
                f.write(f"{b['class_id']} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}\n")

    # ------------------------------------------------------------------
    # Coordinate helpers (normalized YOLO <-> canvas pixel space)
    def _norm_to_canvas(self, box):
        cx, cy, w, h = box["cx"], box["cy"], box["w"], box["h"]
        x1 = (cx - w / 2) * self.img_w
        y1 = (cy - h / 2) * self.img_h
        x2 = (cx + w / 2) * self.img_w
        y2 = (cy + h / 2) * self.img_h
        return (
            self.offset_x + x1 * self.scale, self.offset_y + y1 * self.scale,
            self.offset_x + x2 * self.scale, self.offset_y + y2 * self.scale,
        )

    def _canvas_to_norm(self, x1, y1, x2, y2):
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        ix1 = max(0.0, min(self.img_w, (x1 - self.offset_x) / self.scale))
        iy1 = max(0.0, min(self.img_h, (y1 - self.offset_y) / self.scale))
        ix2 = max(0.0, min(self.img_w, (x2 - self.offset_x) / self.scale))
        iy2 = max(0.0, min(self.img_h, (y2 - self.offset_y) / self.scale))
        cx = (ix1 + ix2) / 2 / self.img_w
        cy = (iy1 + iy2) / 2 / self.img_h
        w = (ix2 - ix1) / self.img_w
        h = (iy2 - iy1) / self.img_h
        return cx, cy, w, h

    def _hit_test(self, x, y):
        """Return the index of the topmost (smallest-area) box containing
        canvas point (x, y), or None if no box contains it."""
        best_idx, best_area = None, None
        for i, b in enumerate(self.boxes):
            x1, y1, x2, y2 = self._norm_to_canvas(b)
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = (x2 - x1) * (y2 - y1)
                if best_area is None or area < best_area:
                    best_area, best_idx = area, i
        return best_idx

    HANDLE_RADIUS = 7

    def _handle_positions(self, box_canvas_coords):
        x1, y1, x2, y2 = box_canvas_coords
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        return {
            "nw": (x1, y1), "n": (mx, y1), "ne": (x2, y1),
            "e": (x2, my), "se": (x2, y2), "s": (mx, y2),
            "sw": (x1, y2), "w": (x1, my),
        }

    def _handle_at(self, x, y):
        """Return the handle name under canvas point (x, y) for the
        currently selected box, or None."""
        if self.selected_box is None:
            return None
        box_coords = self._norm_to_canvas(self.boxes[self.selected_box])
        for name, (hx, hy) in self._handle_positions(box_coords).items():
            if abs(x - hx) <= self.HANDLE_RADIUS and abs(y - hy) <= self.HANDLE_RADIUS:
                return name
        return None

    CURSOR_FOR_HANDLE = {
        "nw": "top_left_corner", "n": "top_side", "ne": "top_right_corner",
        "e": "right_side", "se": "bottom_right_corner", "s": "bottom_side",
        "sw": "bottom_left_corner", "w": "left_side",
    }

    # ------------------------------------------------------------------
    def _redraw(self):
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.tk_image)

        for i, b in enumerate(self.boxes):
            x1, y1, x2, y2 = self._norm_to_canvas(b)
            color = SELECTED_COLOR if i == self.selected_box else BOX_COLOR
            width = 3 if i == self.selected_box else 2
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=width)
            label = (
                self.class_list[b["class_id"]]
                if 0 <= b["class_id"] < len(self.class_list)
                else f"id {b['class_id']}"
            )
            self.canvas.create_text(x1 + 3, y1 - 8, text=label, anchor="w", fill=color, font=("Segoe UI", 9, "bold"))

        if self.selected_box is not None and 0 <= self.selected_box < len(self.boxes):
            box_coords = self._norm_to_canvas(self.boxes[self.selected_box])
            for hx, hy in self._handle_positions(box_coords).values():
                r = self.HANDLE_RADIUS - 2
                self.canvas.create_rectangle(
                    hx - r, hy - r, hx + r, hy + r,
                    fill=SELECTED_COLOR, outline="#000000",
                )

        if self.pending_box:
            x1, y1, x2, y2 = self.pending_box
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=PENDING_COLOR, width=2, dash=(4, 2))

    # ------------------------------------------------------------------
    # Mouse handlers
    def _on_press(self, event):
        # If a box is already selected, check for a resize handle or a
        # click inside it (to move) before falling back to select/draw.
        if self.selected_box is not None:
            handle = self._handle_at(event.x, event.y)
            if handle is not None:
                self.edit_mode = "resize"
                self.edit_handle = handle
                self.edit_orig_box = self._norm_to_canvas(self.boxes[self.selected_box])
                self.edit_start_pt = (event.x, event.y)
                return
            x1, y1, x2, y2 = self._norm_to_canvas(self.boxes[self.selected_box])
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                self.edit_mode = "move"
                self.edit_orig_box = (x1, y1, x2, y2)
                self.edit_start_pt = (event.x, event.y)
                return

        hit = self._hit_test(event.x, event.y)
        if hit is not None:
            self.selected_box = hit
            self.pending_box = None
            self.drag_start = None
            self.edit_mode = None
            self._redraw()
        else:
            self.selected_box = None
            self.edit_mode = "draw"
            self.drag_start = (event.x, event.y)
            self.pending_box = (event.x, event.y, event.x, event.y)
            self._redraw()

    def _on_drag(self, event):
        if self.edit_mode == "resize":
            self._apply_resize(event.x, event.y)
        elif self.edit_mode == "move":
            self._apply_move(event.x, event.y)
        elif self.edit_mode == "draw":
            if self.drag_start is None:
                return
            x0, y0 = self.drag_start
            self.pending_box = (x0, y0, event.x, event.y)
            self._redraw()

    def _on_release(self, event):
        if self.edit_mode in ("resize", "move"):
            self.edit_mode = None
            self.edit_handle = None
            self.edit_orig_box = None
            self.edit_start_pt = None
            self._save_labels()
            self._redraw()
            return

        if self.edit_mode == "draw":
            self.edit_mode = None
            x0, y0 = self.drag_start
            self.drag_start = None
            if abs(event.x - x0) < 5 or abs(event.y - y0) < 5:
                # too small to count as an intentional drag - treat as an empty click
                self.pending_box = None
                self._redraw()
                return
            self.pending_box = (x0, y0, event.x, event.y)
            if self.active_class_id is not None:
                # A class is already active - assign it immediately, no
                # need to click a class button again.
                self._finish_pending_box(self.active_class_id)
            else:
                self._redraw()

    def _on_motion(self, event):
        """Update the cursor to hint at what a click/drag will do."""
        if self.edit_mode is not None:
            return
        if self.selected_box is not None:
            handle = self._handle_at(event.x, event.y)
            if handle is not None:
                self.canvas.config(cursor=self.CURSOR_FOR_HANDLE[handle])
                return
            x1, y1, x2, y2 = self._norm_to_canvas(self.boxes[self.selected_box])
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                self.canvas.config(cursor="fleur")
                return
        self.canvas.config(cursor="crosshair")

    def _apply_move(self, x, y):
        x1, y1, x2, y2 = self.edit_orig_box
        sx, sy = self.edit_start_pt
        dx, dy = x - sx, y - sy
        cx, cy, w, h = self._canvas_to_norm(x1 + dx, y1 + dy, x2 + dx, y2 + dy)
        b = self.boxes[self.selected_box]
        b["cx"], b["cy"], b["w"], b["h"] = cx, cy, w, h
        self._redraw()

    def _apply_resize(self, x, y):
        x1, y1, x2, y2 = self.edit_orig_box
        handle = self.edit_handle
        if "n" in handle:
            y1 = y
        if "s" in handle:
            y2 = y
        if "w" in handle:
            x1 = x
        if "e" in handle:
            x2 = x
        cx, cy, w, h = self._canvas_to_norm(x1, y1, x2, y2)
        b = self.boxes[self.selected_box]
        b["cx"], b["cy"], b["w"], b["h"] = cx, cy, w, h
        self._redraw()

    # ------------------------------------------------------------------
    def _on_class_clicked(self, class_id):
        # Clicking a class button makes it "active": the next box you draw
        # is assigned to it automatically. It also immediately assigns any
        # box that's already been drawn and is waiting for a class.
        self.active_class_id = class_id
        self._update_class_button_styles()
        if self.pending_box is not None:
            self._finish_pending_box(class_id)

    def _finish_pending_box(self, class_id):
        if self.pending_box is None:
            return
        x1, y1, x2, y2 = self.pending_box
        cx, cy, w, h = self._canvas_to_norm(x1, y1, x2, y2)
        if w <= 0 or h <= 0:
            self.pending_box = None
            self._redraw()
            return
        self.boxes.append({"class_id": class_id, "cx": cx, "cy": cy, "w": w, "h": h})
        self.pending_box = None
        self.selected_box = len(self.boxes) - 1
        self._save_labels()
        self._redraw()

    def _on_add_new_class(self):
        from tkinter import messagebox

        name = self.new_class_entry.get().strip()
        if not name:
            return
        if name in self.class_list:
            messagebox.showinfo("Already exists", f"'{name}' is already a class.")
            return
        self.class_list.append(name)
        self._rebuild_class_buttons()
        self._save_classes()
        self.new_class_entry.delete(0, "end")
        # Make the freshly added class active (and assign it to a box that
        # was already drawn and waiting for a class, if any).
        self._on_class_clicked(len(self.class_list) - 1)

    def _save_classes(self):
        yaml_path = find_yaml_file(self.dataset_dir)
        if yaml_path:
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f) or {}
            data["names"] = self.class_list
            data["nc"] = len(self.class_list)
            with open(yaml_path, "w") as f:
                if "train" in data:
                    f.write(f"train: {data['train']}\n")
                if "val" in data:
                    f.write(f"val: {data['val']}\n\n")
                f.write(f"nc: {data['nc']}\n")
                f.write(f"names: {data['names']}\n")
        else:
            txt_path = os.path.join(self.dataset_dir, self.split, "classes.txt")
            with open(txt_path, "w") as f:
                f.write("\n".join(self.class_list) + "\n")

    def _delete_selected(self):
        if self.selected_box is None:
            return
        del self.boxes[self.selected_box]
        self.selected_box = None
        self._save_labels()
        self._redraw()

    def _cancel_pending(self):
        self.pending_box = None
        self.drag_start = None
        self._redraw()

    # ------------------------------------------------------------------
    def _navigate(self, direction):
        new_index = self.index + direction
        if 0 <= new_index < len(self.image_files):
            self._save_labels()  # belt-and-suspenders: flush current edits first
            self.index = new_index
            self._load_current_image()

    def _on_close(self):
        self._save_labels()
        self.root.destroy()