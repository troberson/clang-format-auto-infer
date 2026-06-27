"""Brace style detection for control statements and functions."""

import re


def detect_brace_style_control(files: list[str]) -> str:
    """Detect brace style for control statements (if, for, while, switch, etc.).

    Returns 'Attach' (K&R, same line), 'Allman' (new line), or 'Leave' if mixed.
    """
    attach_pattern = re.compile(
        r"\b(if|for|while|switch|else|do|class|struct|enum|namespace)\b.*\{"
    )
    allman_pattern = re.compile(r"^\s*\{")

    attach_count = 0
    allman_count = 0

    for fpath in files:
        try:
            with open(fpath, errors="replace") as f:
                lines = f.readlines()
        except OSError:  # pragma: no cover
            continue

        i = 0
        while i < len(lines):
            line = lines[i].rstrip("\n\r")
            stripped = line.strip()

            # Skip comments
            if (
                stripped.startswith("//")
                or stripped.startswith("/*")
                or stripped.startswith("*")
            ):
                i += 1
                continue

            # Check for attach style on current line
            if attach_pattern.search(line):
                attach_count += 1
                i += 1
                continue

            # Check for allman style: brace on its own line preceded by a
            # control statement on the previous line
            if allman_pattern.match(line) and i > 0:
                prev = lines[i - 1].strip()
                if re.search(
                    r"\b(if|for|while|switch|else|do)\b\s*\(", prev
                ) and prev.endswith(")"):
                    allman_count += 1

            i += 1

    total = attach_count + allman_count
    if total == 0:
        return "Leave"
    if attach_count / total >= 0.6:
        return "Attach"
    if allman_count / total >= 0.6:
        return "Allman"
    return "Leave"


def detect_brace_style_function(files: list[str]) -> str:
    """Detect brace style for function definitions.

    Returns 'Attach' (K&R, same line), 'Allman' (new line), or 'Leave' if mixed.
    """
    # Match function definitions: identifier followed by ( ... ) {
    # Exclude control flow keywords.
    func_attach_pattern = re.compile(
        r"\b(?!if\b|for\b|while\b|switch\b|if\b|else\b|do\b|class\b|struct\b|enum\b|namespace\b)\w+\s*\([^)]*\)\s*(const|override|noexcept|final)?\s*\{"
    )
    allman_pattern = re.compile(r"^\s*\{")

    attach_count = 0
    allman_count = 0

    for fpath in files:
        try:
            with open(fpath, errors="replace") as f:
                lines = f.readlines()
        except OSError:  # pragma: no cover
            continue

        i = 0
        while i < len(lines):
            line = lines[i].rstrip("\n\r")
            stripped = line.strip()

            # Skip comments
            if (
                stripped.startswith("//")
                or stripped.startswith("/*")
                or stripped.startswith("*")
            ):
                i += 1
                continue

            # Check for attach style on current line
            if func_attach_pattern.search(line):
                attach_count += 1
                i += 1
                continue

            # Check for allman style: brace on its own line preceded by a
            # function signature on the previous line
            if allman_pattern.match(line) and i > 0:
                prev = lines[i - 1].strip()
                if (
                    prev.endswith(")")
                    or prev.endswith(") const")
                    or prev.endswith(") override")
                ):
                    # Only count if the previous line looks like a function signature
                    # (contains a parenthesis that's not a control statement)
                    if not re.search(r"\b(if|for|while|switch)\b\s*\)$", prev):
                        allman_count += 1

            i += 1

    total = attach_count + allman_count
    if total == 0:
        return "Leave"
    if attach_count / total >= 0.6:
        return "Attach"
    if allman_count / total >= 0.6:
        return "Allman"
    return "Leave"


def detect_brace_style(files: list[str]) -> str:
    """Detect brace style: Allman (brace on new line) vs Attach (K&R, same line).

    Looks at control statements (if, for, while, switch) and function definitions.
    Returns 'Allman', 'Attach', or 'Leave' if mixed/unclear.

    Note: This is a combined detector. For finer control, use
    detect_brace_style_control() and detect_brace_style_function() separately.
    """
    # Match opening brace on the same line as a control statement or function
    attach_pattern = re.compile(
        r"\b(if|for|while|switch|else|do|class|struct|enum|namespace)\b.*\{"
    )
    func_attach_pattern = re.compile(
        r"\w+\s*\([^)]*\)\s*(const|override|noexcept|final)?\s*\{"
    )
    allman_pattern = re.compile(r"^\s*\{")

    attach_count = 0
    allman_count = 0

    for fpath in files:
        try:
            with open(fpath, errors="replace") as f:
                lines = f.readlines()
        except OSError:  # pragma: no cover
            continue

        i = 0
        while i < len(lines):
            line = lines[i].rstrip("\n\r")
            stripped = line.strip()

            # Skip comments
            if (
                stripped.startswith("//")
                or stripped.startswith("/*")
                or stripped.startswith("*")
            ):
                i += 1
                continue

            # Check for attach style on current line
            if attach_pattern.search(line) or func_attach_pattern.search(line):
                attach_count += 1
                i += 1
                continue

            # Check for allman style: brace on its own line preceded by a
            # control statement or function signature on the previous line
            if allman_pattern.match(line) and i > 0:
                prev = lines[i - 1].strip()
                if (
                    prev.endswith(")")
                    or prev.endswith(") const")
                    or prev.endswith(") override")
                ):
                    allman_count += 1

            i += 1

    total = attach_count + allman_count
    if total == 0:
        return "Leave"
    if attach_count / total >= 0.6:
        return "Attach"
    if allman_count / total >= 0.6:
        return "Allman"
    return "Leave"
