"""
Reusable curses-based CLI UI components.

- single_select_menu: pick one option from a list using Up/Down + Enter
- class_toggle_menu:  toggle each item between INCLUDED (green) and
                       EXCLUDED (red) using Left/Right, Up/Down to move

Both are blocking calls (they own the terminal for their duration via
curses.wrapper) and return plain Python values, so callers never touch
curses directly.
"""
import curses
import os


def _init_colors():
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)  # included
    curses.init_pair(2, curses.COLOR_RED, -1)    # excluded
    curses.init_pair(3, curses.COLOR_CYAN, -1)   # titles / info


# ---------------------------------------------------------------------
# Single select menu
# ---------------------------------------------------------------------
def _single_select(stdscr, title, options, subtitle):
    curses.curs_set(0)
    _init_colors()
    idx = 0

    while True:
        stdscr.clear()
        h, _ = stdscr.getmaxyx()

        stdscr.addstr(0, 2, title, curses.A_BOLD | curses.color_pair(3))
        row = 2
        if subtitle:
            stdscr.addstr(row, 2, subtitle, curses.color_pair(3))
            row += 2

        for i, opt in enumerate(options):
            prefix = "-> " if i == idx else "   "
            attr = curses.A_REVERSE if i == idx else curses.A_NORMAL
            if row + i < h:
                stdscr.addstr(row + i, 2, f"{prefix}{opt}", attr)

        footer_row = row + len(options) + 2
        if footer_row < h:
            stdscr.addstr(footer_row, 2, "Up/Down: move   Enter: select   q: cancel")

        key = stdscr.getch()
        if key in (curses.KEY_UP, ord('k')):
            idx = (idx - 1) % len(options)
        elif key in (curses.KEY_DOWN, ord('j')):
            idx = (idx + 1) % len(options)
        elif key in (curses.KEY_ENTER, 10, 13):
            return idx
        elif key in (ord('q'), 27):
            return None


def single_select_menu(title, options, subtitle=None):
    """Show a single-select arrow-key menu. Returns the chosen index, or
    None if the user cancelled (q / Esc)."""
    if not options:
        return None
    return curses.wrapper(_single_select, title, options, subtitle)


# ---------------------------------------------------------------------
# Class include/exclude toggle menu
# ---------------------------------------------------------------------
def _class_toggle(stdscr, classes):
    curses.curs_set(0)
    _init_colors()
    included = {c: True for c in classes}
    idx = 0

    while True:
        stdscr.clear()
        h, _ = stdscr.getmaxyx()

        stdscr.addstr(0, 2, "Select classes to include", curses.A_BOLD | curses.color_pair(3))
        stdscr.addstr(1, 2, "Left: include (green)   Right: exclude (red)   Up/Down: move")
        stdscr.addstr(2, 2, "a: include all   d: exclude all   Enter: confirm   q: cancel")

        row = 4
        for i, c in enumerate(classes):
            is_included = included[c]
            label = "INCLUDED" if is_included else "EXCLUDED"
            color = curses.color_pair(1) if is_included else curses.color_pair(2)
            prefix = "-> " if i == idx else "   "
            attr = color | (curses.A_REVERSE if i == idx else curses.A_BOLD)
            if row + i < h:
                stdscr.addstr(row + i, 2, f"{prefix}{c:<25s} [{label}]", attr)

        key = stdscr.getch()
        if key in (curses.KEY_UP, ord('k')):
            idx = (idx - 1) % len(classes)
        elif key in (curses.KEY_DOWN, ord('j')):
            idx = (idx + 1) % len(classes)
        elif key == curses.KEY_LEFT:
            included[classes[idx]] = True
        elif key == curses.KEY_RIGHT:
            included[classes[idx]] = False
        elif key == ord('a'):
            included = {c: True for c in classes}
        elif key == ord('d'):
            included = {c: False for c in classes}
        elif key in (curses.KEY_ENTER, 10, 13):
            return included
        elif key in (ord('q'), 27):
            return None


def class_toggle_menu(classes):
    """Show the green/red class include/exclude screen.
    Returns {class_name: True/False}, or None if cancelled."""
    if not classes:
        return None
    return curses.wrapper(_class_toggle, classes)


# ---------------------------------------------------------------------
# Read-only scrollable table (single column, e.g. class counts)
# ---------------------------------------------------------------------
def _format_row(row, col_widths):
    cells = []
    for i, val in enumerate(row):
        text = str(val)
        cells.append(text.ljust(col_widths[i]) if i == 0 else text.rjust(col_widths[i]))
    return "  ".join(cells)


def _table_screen(stdscr, title, headers, rows, subtitle):
    curses.curs_set(0)
    _init_colors()

    col_widths = []
    for i in range(len(headers)):
        width = len(str(headers[i]))
        for r in rows:
            width = max(width, len(str(r[i])))
        col_widths.append(width)

    header_line = _format_row(headers, col_widths)

    scroll = 0
    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()

        stdscr.addstr(0, 2, title, curses.A_BOLD | curses.color_pair(3))
        top = 2
        if subtitle:
            stdscr.addstr(top, 2, str(subtitle)[: max(0, w - 4)], curses.color_pair(3))
            top += 2

        body_top = top + 1  # +1 for the header row
        available_h = max(1, h - body_top - 2)  # leave room for footer

        max_scroll = max(0, len(rows) - available_h)
        scroll = min(scroll, max_scroll)

        stdscr.addstr(top, 2, header_line[: max(0, w - 4)], curses.A_BOLD | curses.color_pair(3))

        for line_i in range(available_h):
            row_i = scroll + line_i
            if row_i >= len(rows):
                break
            y = body_top + line_i
            if y >= h - 1:
                continue
            stdscr.addstr(y, 2, _format_row(rows[row_i], col_widths)[: max(0, w - 4)])

        footer = "Up/Down: scroll   q / Enter: close"
        if h - 1 >= 0:
            stdscr.addstr(h - 1, 2, footer[: max(0, w - 4)])

        key = stdscr.getch()
        if key in (curses.KEY_UP, ord('k')):
            scroll = max(0, scroll - 1)
        elif key in (curses.KEY_DOWN, ord('j')):
            scroll = min(max_scroll, scroll + 1)
        elif key in (ord('q'), 27, curses.KEY_ENTER, 10, 13):
            return


def table_menu(title, headers, rows, subtitle=None):
    """Show a read-only, scrollable table - a single column of rows,
    scrolled with Up/Down (e.g. class counts: Class / Train / Valid / Total).

    headers: list of column header strings.
    rows:    list of tuples/lists, one per row, values aligned to headers.
    """
    if not headers:
        return
    if not rows:
        rows = [("(no data)",) + ("",) * (len(headers) - 1)]
    curses.wrapper(_table_screen, title, headers, rows, subtitle)


# ---------------------------------------------------------------------
# Folder browser
# ---------------------------------------------------------------------
def browse_directory_menu(title, start_dir=".", accept_predicate=None,
                           accept_label="Use this folder", allow_create=False):
    """Interactive folder browser: pick a subfolder to descend into, ".."
    to go back up, "Use this folder" if the current folder satisfies
    accept_predicate (or always, if accept_predicate is None), and
    optionally "Create a new folder here".

    Returns the chosen absolute directory path, or None if cancelled.
    """
    current = os.path.abspath(start_dir)

    while True:
        try:
            entries = sorted(
                e for e in os.listdir(current)
                if os.path.isdir(os.path.join(current, e)) and not e.startswith(".")
            )
        except OSError as e:
            print(f"Could not list '{current}': {e}")
            entries = []

        can_accept = accept_predicate is None or accept_predicate(current)
        parent = os.path.dirname(current)
        has_parent = bool(parent) and parent != current

        # (kind, display_label) pairs
        options = []
        if can_accept:
            options.append(("USE", f"[{accept_label}]"))
        if allow_create:
            options.append(("CREATE", "[Create a new folder here]"))
        if has_parent:
            options.append(("UP", ".. (go back)"))
        options.extend(("DIR", e) for e in entries)

        if not options:
            print(f"'{current}' has no subfolders and can't be used as-is.")
            return None

        display = [label for _, label in options]
        idx = single_select_menu(title, display, subtitle=f"Current folder: {current}")
        if idx is None:
            return None

        kind, label = options[idx]
        if kind == "USE":
            return current
        elif kind == "CREATE":
            name = input("New folder name: ").strip()
            if not name:
                print("No name given.")
                continue
            new_path = os.path.join(current, name)
            os.makedirs(new_path, exist_ok=True)
            return new_path
        elif kind == "UP":
            current = parent
        elif kind == "DIR":
            current = os.path.join(current, label)