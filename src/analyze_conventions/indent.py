"""Indentation and tab detection."""

from collections import Counter


def detect_indent_width(files: list[str]) -> int:
    """Detect the most common indentation unit by analyzing leading whitespace."""
    tab_count = 0
    space_indents: Counter[int] = Counter()
    for fpath in files:
        try:
            with open(fpath, errors="replace") as f:
                for line in f:
                    stripped = line.lstrip(" \t")
                    if not stripped:  # pragma: no cover
                        continue
                    leading = line[: len(line) - len(stripped)]
                    if "\t" in leading:
                        tab_count += 1
                    elif leading:
                        space_indents[len(leading)] += 1
        except OSError:  # pragma: no cover
            continue

    # If tabs dominate, suggest a reasonable tab width (4 is standard)
    if tab_count > space_indents.total():
        return 4  # Tab-based indentation; 4 is the most common tab width

    if not space_indents:
        return 4  # fallback

    # Find the indent width (2-8) that divides the most common indent levels.
    # Use >= so that when scores are tied, the LARGEST width wins (e.g. width 4
    # beats width 2 when both divide all indent levels evenly).
    best_width = 4
    best_score = 0
    for width in range(2, 9):
        score = 0
        for indent, count in space_indents.items():
            if indent % width == 0:
                score += count
        if score >= best_score:
            best_score = score
            best_width = width

    return best_width


def detect_use_tab(files: list[str]) -> str:
    """Detect whether the codebase uses tabs or spaces for indentation.

    Returns 'Always' if tabs dominate, 'Never' if spaces dominate, 'Leave' if mixed.
    """
    tab_lines = 0
    space_lines = 0

    for fpath in files:
        try:
            with open(fpath, errors="replace") as f:
                for line in f:
                    stripped = line.lstrip(" \t")
                    leading = line[: len(line) - len(stripped)]
                    if not leading:
                        continue
                    if "\t" in leading:
                        tab_lines += 1
                    else:
                        space_lines += 1
        except OSError:  # pragma: no cover
            continue

    total = tab_lines + space_lines
    if total == 0:
        return "Leave"
    if tab_lines / total >= 0.6:
        return "Always"
    if space_lines / total >= 0.6:
        return "Never"
    return "Leave"
