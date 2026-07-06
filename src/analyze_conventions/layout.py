"""Layout detectors: column limit, empty lines, access modifier offset."""

import re
import sys
from collections import Counter


def detect_column_limit(
    files: list[str],
    percentile: float = 0.95,
    tab_width: int | None = None,
    debug: bool = False,
) -> int | None:
    """Detect natural column limit by analyzing line length distribution.

    Filters out lines that are likely copyright headers or other noise:
    - Lines longer than 200 chars are excluded (likely copyright/legal text)
    - Lines that are pure comments starting with /* or // and very long

    When tab_width is provided, expands tabs to visual width for accurate
    column limit detection. This is important for tab-indented codebases
    where raw character count underestimates visual width.
    """
    STANDARD_LIMITS = [79, 80, 100, 120, 128]
    SNAP_THRESHOLD = 10  # snap if within this many characters

    lengths: list[int] = []
    for fpath in files:
        try:
            with open(fpath, errors="replace") as f:
                for line in f:
                    raw = line.rstrip("\n\r")
                    if tab_width and tab_width > 0 and "\t" in raw:
                        # Expand tabs to visual width for accurate detection
                        length = len(raw.expandtabs(tab_width))
                    else:
                        length = len(raw)
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

    if debug:
        n = len(lengths)
        debug_pcts = [0.50, 0.75, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00]
        pctl_line = "  Column limit percentiles:"
        for p in debug_pcts:
            val = lengths[min(int(n * p), n - 1)]
            pctl_line += f" {p:.0%}={val}"
        print(pctl_line, file=sys.stderr)
        print(
            f"  Detected: {detected} (at {percentile:.0%}), tab_width={tab_width}, files={len(files)}",
            file=sys.stderr,
        )

    # Snap to the CLOSEST standard value if within threshold.
    best_standard = None
    best_distance = SNAP_THRESHOLD + 1
    for standard in STANDARD_LIMITS:
        distance = abs(detected - standard)
        if distance <= SNAP_THRESHOLD and distance < best_distance:
            best_distance = distance
            best_standard = standard
    if best_standard is not None:
        if debug:
            print(
                f"  Snapped {detected} -> {best_standard} (distance {best_distance})",
                file=sys.stderr,
            )
        return best_standard

    if debug:
        print(
            f"  No snap (all standards >{SNAP_THRESHOLD} away), returning {detected}",
            file=sys.stderr,
        )
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
