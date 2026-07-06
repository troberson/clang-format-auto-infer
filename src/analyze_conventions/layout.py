"""Layout detectors: column limit, empty lines, access modifier offset."""

import re
from collections import Counter


def detect_column_limit(files: list[str], percentile: float = 0.90) -> int | None:
    """Detect natural column limit by analyzing line length distribution.

    Filters out lines that are likely copyright headers or other noise:
    - Lines longer than 200 chars are excluded (likely copyright/legal text)
    - Lines that are pure comments starting with /* or // and very long

    Counts printable characters, matching clang-format's ColumnLimit behavior.
    Tabs count as 1 character, not visual width.
    """
    STANDARD_LIMITS = [79, 80, 100, 120, 128]
    SNAP_THRESHOLD = 5  # snap if within this many characters

    lengths: list[int] = []
    for fpath in files:
        try:
            with open(fpath, errors="replace") as f:
                for line in f:
                    # Count printable characters, matching clang-format's ColumnLimit.
                    # clang-format counts tabs as 1 character, not visual width.
                    length = len(line.rstrip("\n\r"))
                    if length > 200:
                        continue  # skip obvious outliers
                    if length > 0:
                        lengths.append(length)
        except OSError:  # pragma: no cover
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
                        indent = len(
                            m.group(1)
                            if m.group(1) is not None
                            else m.group(2)
                            if m.group(2) is not None
                            else m.group(3)
                        )
                        offsets[indent] += 1
        except OSError:  # pragma: no cover
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
        except OSError:  # pragma: no cover
            continue

    return max_consecutive
