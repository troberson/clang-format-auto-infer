"""Numeric literal case detection."""

import re


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
                    if (  # pragma: no cover
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
        except OSError:  # pragma: no cover
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
