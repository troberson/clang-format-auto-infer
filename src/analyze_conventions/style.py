"""Style detectors for BSD-style churn options.

Detects options that cause massive reformatting on BSD-style codebases
when left at their default values.
"""

import re


def detect_allow_short_functions(files: list[str]) -> str:
    """Detect AllowShortFunctionsOnASingleLine convention.

    Scans for short functions (body <= 1 line) that are written on a single
    line vs across multiple lines.

    Returns 'All' if short functions are on one line, 'None' if they span
    multiple lines, or 'Leave' if mixed/unclear.
    """
    single_line_count = 0
    multiline_count = 0

    # Match function signature with opening brace on same line
    func_attach_pattern = re.compile(
        r"\w+\s*\([^)]*\)\s*(const|override|noexcept|final)?\s*\{"
    )
    # Match function signature without brace (Allman style)
    func_allman_pattern = re.compile(
        r"^\s*\w[\w*&<>:]*\s+\w+\s*\([^)]*\)\s*(const|override|noexcept|final)?\s*$"
    )
    # Match a complete single-line function body: { ... }
    single_line_body = re.compile(r"\{[^{}]+\}")
    # Control statements to skip
    control_pattern = re.compile(r"\b(if|for|while|switch|else|do)\b\s*\(")

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
            if stripped.startswith("//") or stripped.startswith("/*"):
                i += 1
                continue

            # Skip control statements
            if control_pattern.search(line):
                i += 1
                continue

            # Check for attach-style function (brace on same line)
            if func_attach_pattern.search(line):
                brace_match = single_line_body.search(line)
                if brace_match:
                    body = brace_match.group(0)
                    if body.count("{") == 1 and body.count("}") == 1:
                        single_line_count += 1
                else:
                    # Has opening brace but no closing on same line
                    multiline_count += 1
                i += 1
                continue

            # Check for Allman-style function (signature on its own line)
            if func_allman_pattern.match(line):
                # Check if next non-empty line is an opening brace
                j = i + 1
                while j < len(lines):
                    next_stripped = lines[j].strip()
                    if next_stripped:
                        if next_stripped == "{":
                            multiline_count += 1
                        break
                    j += 1

            i += 1

    total = single_line_count + multiline_count
    if total == 0:
        return "Leave"
    if single_line_count / total >= 0.6:
        return "All"
    if multiline_count / total >= 0.6:
        return "None"
    return "Leave"


def detect_allow_short_blocks(files: list[str]) -> str:
    """Detect AllowShortBlocksOnASingleLine convention.

    Scans for short blocks ({ ... }) that are written on a single line
    vs across multiple lines.

    Returns 'All' if short blocks are on one line, 'Never' if they span
    multiple lines, or 'Leave' if mixed/unclear.
    """
    single_line_count = 0
    multiline_count = 0

    # Control statements that introduce blocks
    block_pattern = re.compile(
        r"\b(if|for|while|switch|else|do|class|struct|enum|namespace)\b"
    )

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
            if stripped.startswith("//") or stripped.startswith("/*"):
                i += 1
                continue

            # Look for a block-opening construct
            if "{" in stripped:
                brace_pos = stripped.find("{")
                rest = stripped[brace_pos:]
                close_pos = rest.find("}")

                if close_pos != -1 and close_pos > brace_pos:
                    # Potential single-line block
                    inner = rest[1:close_pos]
                    # Only count if no nested braces
                    if "{" not in inner and "}" not in inner:
                        single_line_count += 1
                        i += 1
                        continue

                # Opening brace without closing on same line -> multiline
                # Only count if this looks like a block (not just any brace)
                if block_pattern.search(line) or stripped.endswith("{"):
                    multiline_count += 1

            i += 1

    total = single_line_count + multiline_count
    if total == 0:
        return "Leave"
    if single_line_count / total >= 0.6:
        return "All"
    if multiline_count / total >= 0.6:
        return "Never"
    return "Leave"


def detect_align_consecutive_declarations(files: list[str]) -> bool | None:
    """Detect if consecutive variable declarations are aligned.

    Scans for consecutive declarations where the variable names are
    aligned using extra whitespace (tabs or spaces).

    Returns True if alignment is detected (majority of consecutive
    declaration groups are aligned), False if not, or None if unclear.
    """
    aligned_groups = 0
    unaligned_groups = 0

    # Match a declaration line: optional indent, type words, variable name, semicolon/equals
    decl_pattern = re.compile(r"^(\s*)(\w[\w*&<>:]*\s+)+(\w+)\s*[;=]")

    for fpath in files:
        try:
            with open(fpath, errors="replace") as f:
                lines = f.readlines()
        except OSError:  # pragma: no cover
            continue

        i = 0
        while i < len(lines):
            m = decl_pattern.match(lines[i])
            if m:
                # Found a declaration, check if next lines are also declarations
                group = [m]
                j = i + 1
                while j < len(lines):
                    next_m = decl_pattern.match(lines[j])
                    if next_m:
                        group.append(next_m)
                        j += 1
                    else:
                        break

                if len(group) >= 2:
                    # Check if variable names are at the same column
                    var_positions = [m.start(3) for m in group]
                    if len(set(var_positions)) == 1:
                        aligned_groups += 1
                    else:
                        unaligned_groups += 1

                i = j
            else:
                i += 1

    total = aligned_groups + unaligned_groups
    if total == 0:
        return None
    if aligned_groups / total >= 0.6:
        return True
    if unaligned_groups / total >= 0.6:
        return False
    return None


def detect_indent_goto_labels(files: list[str]) -> bool | None:
    """Detect if goto labels are indented or at column 0.

    Scans for goto labels (identifier followed by colon at start of line).

    Returns True if labels are indented, False if at column 0, or None
    if mixed/unclear.
    """
    indented_count = 0
    column_zero_count = 0

    label_pattern = re.compile(r"^(\s*)(\w+):")

    for fpath in files:
        try:
            with open(fpath, errors="replace") as f:
                lines = f.readlines()
        except OSError:  # pragma: no cover
            continue

        for line in lines:
            m = label_pattern.match(line)
            if m:
                indent = m.group(1)
                # Skip case labels - they're handled separately
                # Check if previous non-empty line has 'case' or 'switch'
                label_name = m.group(2)
                if label_name in ("case", "default"):
                    continue

                if indent:
                    indented_count += 1
                else:
                    column_zero_count += 1

    total = indented_count + column_zero_count
    if total == 0:
        return None
    if indented_count / total >= 0.6:
        return True
    if column_zero_count / total >= 0.6:
        return False
    return None


def detect_align_case_labels(files: list[str]) -> bool | None:
    """Detect if case labels with short statements are aligned.

    Scans for case labels and checks if the statements after the colon
    are aligned across cases.

    Returns True if alignment is detected, False if not, or None if unclear.
    """
    aligned_groups = 0
    unaligned_groups = 0

    case_pattern = re.compile(r"^\s+case\s+.+:\s*(.*)")

    for fpath in files:
        try:
            with open(fpath, errors="replace") as f:
                lines = f.readlines()
        except OSError:  # pragma: no cover
            continue

        i = 0
        while i < len(lines):
            if case_pattern.match(lines[i]):
                # Found a case label, collect consecutive case labels
                stmt_positions = []
                j = i
                while j < len(lines) and case_pattern.match(lines[j]):
                    m = case_pattern.match(lines[j])
                    if m:
                        # Position of the statement text after the colon
                        stmt_start = m.start(1)
                        stmt_positions.append(stmt_start)
                    j += 1

                if len(stmt_positions) >= 2:
                    if len(set(stmt_positions)) == 1:
                        aligned_groups += 1
                    else:
                        unaligned_groups += 1

                i = j
            else:
                i += 1

    total = aligned_groups + unaligned_groups
    if total == 0:
        return None
    if aligned_groups / total >= 0.6:
        return True
    if unaligned_groups / total >= 0.6:
        return False
    return None


def detect_trailing_comment_style(files: list[str]) -> str:
    """Detect trailing comment alignment style.

    Scans for trailing comments (comments at end of code lines) and
    determines if they are aligned to a column.

    Returns 'Always' if comments are aligned, 'Never' if not, or 'Leave'
    if mixed/unclear.
    """
    aligned_count = 0
    unaligned_count = 0

    trailing_comment = re.compile(r"^(.+?\S)\s+(/\*|//)")

    for fpath in files:
        try:
            with open(fpath, errors="replace") as f:
                lines = f.readlines()
        except OSError:  # pragma: no cover
            continue

        i = 0
        while i < len(lines):
            line = lines[i].rstrip("\n\r")
            m = trailing_comment.match(line)
            if m:
                comment_pos = m.start(2)
                # Check next line with a trailing comment
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].rstrip("\n\r")
                    next_m = trailing_comment.match(next_line)
                    if next_m:
                        next_pos = next_m.start(2)
                        if comment_pos == next_pos:
                            aligned_count += 1
                        else:
                            unaligned_count += 1
                        break
                    elif next_line.strip() and not next_line.strip().startswith("//"):
                        # Non-comment, non-empty line - stop searching
                        break
                    j += 1

            i += 1

    total = aligned_count + unaligned_count
    if total == 0:
        return "Leave"
    if aligned_count / total >= 0.6:
        return "Always"
    if unaligned_count / total >= 0.6:
        return "Never"
    return "Leave"
