#!/usr/bin/env python3
"""Analyze a codebase to derive sensible clang-format settings.

Detects:
- IndentWidth: most common indentation unit
- ColumnLimit: natural line-length cutoff (excluding outliers like copyright headers)
- AccessModifierOffset: alignment offset for public:/private:/protected:
- MaxEmptyLinesToKeep: maximum consecutive empty lines found in the codebase
- NumericLiteralCase.HexDigit: uppercase vs lowercase hex digits
- NumericLiteralCase.Prefix: 0x vs 0X prefix case
"""

import argparse
import os
import re
import sys
from collections import Counter


def _source_files(root_dir: str) -> list[str]:
    """Find C/C++ source and header files recursively."""
    extensions = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".m", ".mm"}
    files: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in extensions:
                files.append(os.path.join(dirpath, fn))
    return files


def detect_indent_width(files: list[str]) -> int:
    """Detect the most common indentation unit by analyzing leading whitespace."""
    tab_count = 0
    space_indents: Counter[int] = Counter()
    for fpath in files:
        try:
            with open(fpath, errors="replace") as f:
                for line in f:
                    stripped = line.lstrip(" \t")
                    if not stripped:
                        continue
                    leading = line[: len(line) - len(stripped)]
                    if "\t" in leading:
                        tab_count += 1
                    elif leading:
                        space_indents[len(leading)] += 1
        except OSError:
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


def detect_column_limit(
    files: list[str], percentile: float = 0.90, tab_width: int = 4
) -> int | None:
    """Detect natural column limit by analyzing line length distribution.

    Filters out lines that are likely copyright headers or other noise:
    - Lines longer than 200 chars are excluded (likely copyright/legal text)
    - Lines that are pure comments starting with /* or // and very long

    Expands tabs to their visual width before measuring.
    """
    STANDARD_LIMITS = [40, 72, 79, 80, 100, 120]
    SNAP_THRESHOLD = 5  # snap if within this many characters

    lengths: list[int] = []
    for fpath in files:
        try:
            with open(fpath, errors="replace") as f:
                for line in f:
                    # Expand tabs to visual width, matching clang-format's printable character count
                    visual = line.rstrip("\n\r").expandtabs(tab_width)
                    length = len(visual)
                    if length > 200:
                        continue  # skip obvious outliers
                    if length > 0:
                        lengths.append(length)
        except OSError:
            continue

    if not lengths:
        return None

    lengths.sort()
    idx = int(len(lengths) * percentile)
    detected = lengths[idx]

    # Snap to the CLOSEST standard value if within threshold.
    best_standard = None
    best_distance = SNAP_THRESHOLD + 1
    for standard in STANDARD_LIMITS:
        distance = abs(detected - standard)
        if distance <= SNAP_THRESHOLD and distance < best_distance:
            best_distance = distance
            best_standard = standard
    if best_standard is not None:
        return best_standard

    return detected


def detect_language(files: list[str]) -> str | None:
    """Detect the primary language from file extensions."""
    ext_counts: Counter[str] = Counter()
    lang_map = {
        ".c": "C",
        ".m": "ObjC",
        ".mm": "ObjCpp",
        ".cc": "Cpp",
        ".cpp": "Cpp",
        ".cxx": "Cpp",
        ".h": "Cpp",  # headers are typically C++ in mixed projects
        ".hh": "Cpp",
        ".hpp": "Cpp",
        ".hxx": "Cpp",
    }
    for fpath in files:
        ext = os.path.splitext(fpath)[1].lower()
        if ext in lang_map:
            ext_counts[lang_map[ext]] += 1

    if not ext_counts:
        return None
    return ext_counts.most_common(1)[0][0]


def detect_qualifier_alignment(files: list[str]) -> str:
    """Detect qualifier alignment style (const/volatile positioning).

    - Left:  'const int', 'volatile char'
    - Right: 'int const', 'char volatile'
    - Leave:      mixed or unclear
    """
    type_keywords = {
        "int",
        "char",
        "float",
        "double",
        "void",
        "long",
        "short",
        "unsigned",
        "signed",
        "bool",
        "auto",
        "size_t",
        "ssize_t",
        "uint8_t",
        "uint16_t",
        "uint32_t",
        "uint64_t",
        "int8_t",
        "int16_t",
        "int32_t",
        "int64_t",
    }
    qualifiers = {"const", "volatile"}

    align_left = 0  # qualifier before type: 'const int'
    align_right = 0  # type before qualifier: 'int const'

    for fpath in files:
        try:
            with open(fpath, errors="replace") as f:
                for line in f:
                    # Skip comments
                    stripped = line.strip()
                    if (
                        stripped.startswith("//")
                        or stripped.startswith("/*")
                        or stripped.startswith("*")
                    ):
                        continue
                    words = stripped.split()
                    for i, word in enumerate(words):
                        clean = word.strip("*,&;()")
                        if clean in qualifiers:
                            # Check if next word is a type keyword -> AlignLeft
                            if i + 1 < len(words):
                                next_word = words[i + 1].strip("*,&;()")
                                if next_word in type_keywords:
                                    align_left += 1
                            # Check if previous word is a type keyword -> AlignRight
                            # (independent check — a qualifier can have both prev and next words)
                            if i > 0:
                                prev_word = words[i - 1].strip("*,&;()")
                                if prev_word in type_keywords:
                                    align_right += 1
        except OSError:
            continue

    total = align_left + align_right
    if total == 0:
        return "Leave"

    # Require at least 60% dominance to suggest a specific alignment
    if align_left / total >= 0.6:
        return "Left"
    if align_right / total >= 0.6:
        return "Right"
    return "Leave"


def detect_access_modifier_offset(files: list[str]) -> int | None:
    """Detect the offset used for access modifiers (public:, private:, etc.)."""
    offsets: Counter[int] = Counter()
    pattern = re.compile(r"^( *)public:\s*$|^( *)private:\s*$|^( *)protected:\s*$")
    for fpath in files:
        try:
            with open(fpath, errors="replace") as f:
                for line in f:
                    m = pattern.match(line)
                    if m:
                        indent = len(m.group(1) or m.group(2) or m.group(3))
                        offsets[indent] += 1
        except OSError:
            continue

    if not offsets:
        return None
    return offsets.most_common(1)[0][0]


def detect_max_empty_lines(files: list[str]) -> int:
    """Find the maximum number of consecutive empty lines in the codebase."""
    max_consecutive = 0
    for fpath in files:
        try:
            with open(fpath, errors="replace") as f:
                consecutive = 0
                for line in f:
                    if line.strip() == "":
                        consecutive += 1
                        max_consecutive = max(max_consecutive, consecutive)
                    else:
                        consecutive = 0
        except OSError:
            continue

    return max_consecutive


def detect_numeric_literal_case(files: list[str]) -> tuple[str, str, str, str]:
    """Detect numeric literal conventions (hex, exponent, prefix, suffix).

    Returns:
        tuple: (hex_digit_case, prefix_case, exponent_case, suffix_case)
               where each is 'Upper', 'Lower', or 'Leave'
    """
    # Match hex literals: 0x or 0X followed by hex digits
    hex_pattern = re.compile(r"\b(0[xX])([0-9a-fA-F]+)\b")
    # Match float literals with exponent: e.g. 1.5e10, 3.14E-2, 1e5
    exponent_pattern = re.compile(r"[0-9]([eE])([+-]?[0-9]+)")
    # Match integer suffixes: e.g. 10u, 10l, 10ull, 10UL, 10ULL
    suffix_pattern = re.compile(
        r"\b[0-9][0-9a-fA-FxX]*(u|l|ul|lu|ull|llu|ll)(\b)", re.IGNORECASE
    )

    prefix_lower = 0  # 0x
    prefix_upper = 0  # 0X
    digit_lower = 0  # a-f
    digit_upper = 0  # A-F
    exponent_lower = 0  # e
    exponent_upper = 0  # E
    suffix_lower = 0  # u, l, ull
    suffix_upper = 0  # U, L, ULL
    suffix_mixed = 0  # uL, Ul, etc.

    for fpath in files:
        try:
            with open(fpath, errors="replace") as f:
                for line in f:
                    # Skip comments
                    stripped = line.strip()
                    if (
                        stripped.startswith("//")
                        or stripped.startswith("/*")
                        or stripped.startswith("*")
                    ):
                        continue
                    # Remove string literals to avoid false matches
                    line_clean = re.sub(r'"(?:[^"\\]|\\.)*"', '""', line)
                    line_clean = re.sub(r"'(?:[^'\\]|\\.)*'", "''", line_clean)

                    for m in hex_pattern.finditer(line_clean):
                        prefix = m.group(1)
                        digits = m.group(2)

                        if prefix == "0x":
                            prefix_lower += 1
                        else:
                            prefix_upper += 1

                        for ch in digits:
                            if ch in "abcdef":
                                digit_lower += 1
                            elif ch in "ABCDEF":
                                digit_upper += 1

                    for m in exponent_pattern.finditer(line_clean):
                        exp_char = m.group(1)
                        if exp_char == "e":
                            exponent_lower += 1
                        else:
                            exponent_upper += 1

                    for m in suffix_pattern.finditer(line_clean):
                        suf = m.group(1)
                        if suf == suf.lower():
                            suffix_lower += 1
                        elif suf == suf.upper():
                            suffix_upper += 1
                        else:
                            suffix_mixed += 1
        except OSError:
            continue

    # Determine hex digit case
    total_digits = digit_lower + digit_upper
    if total_digits == 0:
        hex_digit_case = "Leave"
    elif digit_lower / total_digits >= 0.6:
        hex_digit_case = "Lower"
    elif digit_upper / total_digits >= 0.6:
        hex_digit_case = "Upper"
    else:
        hex_digit_case = "Leave"

    # Determine prefix case
    total_prefix = prefix_lower + prefix_upper
    if total_prefix == 0:
        prefix_case = "Leave"
    elif prefix_lower / total_prefix >= 0.6:
        prefix_case = "Lower"
    elif prefix_upper / total_prefix >= 0.6:
        prefix_case = "Upper"
    else:
        prefix_case = "Leave"

    # Determine exponent letter case
    total_exponent = exponent_lower + exponent_upper
    if total_exponent == 0:
        exponent_case = "Leave"
    elif exponent_lower / total_exponent >= 0.6:
        exponent_case = "Lower"
    elif exponent_upper / total_exponent >= 0.6:
        exponent_case = "Upper"
    else:
        exponent_case = "Leave"

    # Determine suffix case (mixed counts toward neither)
    total_suffix = suffix_lower + suffix_upper
    if total_suffix == 0:
        suffix_case = "Leave"
    elif suffix_lower / total_suffix >= 0.6:
        suffix_case = "Lower"
    elif suffix_upper / total_suffix >= 0.6:
        suffix_case = "Upper"
    else:
        suffix_case = "Leave"

    return hex_digit_case, prefix_case, exponent_case, suffix_case


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a codebase to derive sensible clang-format settings."
    )
    _ = parser.add_argument("path", help="Path to the source directory to analyze.")
    _ = parser.add_argument(
        "--percentile",
        type=float,
        default=0.95,
        help="Percentile for column limit detection (0.0-1.0). Default: 0.95",
    )
    args = parser.parse_args()
    path: str = args.path
    percentile: float = args.percentile

    if not os.path.isdir(path):
        print(f"Error: '{path}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    files = _source_files(path)
    if not files:
        print(f"No C/C++ source files found in '{path}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing {len(files)} files in '{path}'...", file=sys.stderr)

    indent_width = detect_indent_width(files)
    column_limit = detect_column_limit(files, percentile, indent_width or 4)
    language = detect_language(files)
    qualifier_align = detect_qualifier_alignment(files)
    access_offset = detect_access_modifier_offset(files)
    max_empty = detect_max_empty_lines(files)
    hex_digit_case, prefix_case, exponent_case, suffix_case = (
        detect_numeric_literal_case(files)
    )

    print(f"# Detected conventions from {len(files)} files", file=sys.stderr)

    # Output valid YAML
    print("DisableFormat: false")
    print(f"Language: {language}")
    print(f"QualifierAlignment: {qualifier_align}")
    print(f"IndentWidth: {indent_width}")
    print(f"ColumnLimit: {column_limit}")
    if access_offset is not None:
        print(f"AccessModifierOffset: {access_offset}")
    print(f"MaxEmptyLinesToKeep: {max_empty}")
    print("DerivePointerAlignment: true")
    print("NumericLiteralCase:")
    print(f"  HexDigit: {hex_digit_case}")
    print(f"  Prefix: {prefix_case}")
    print(f"  ExponentLetter: {exponent_case}")
    print(f"  Suffix: {suffix_case}")


if __name__ == "__main__":
    main()
