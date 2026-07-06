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

    # Collect raw line strings in a single pass.
    raw_lines: list[str] = []
    has_tabs = False
    for fpath in files:
        try:
            with open(fpath, errors="replace") as f:
                for line in f:
                    raw = line.rstrip("\n\r")
                    if "\t" in raw:
                        has_tabs = True
                    if len(raw) > 0 and len(raw) <= 200:
                        raw_lines.append(raw)
        except OSError:  # pragma: no cover
            continue

    if not raw_lines:
        return None

    # Determine which tab widths to try.
    tab_widths_to_try: list[int | None]
    if has_tabs:
        tab_widths_to_try = [4, 8]
    else:
        tab_widths_to_try = [tab_width] if tab_width else [None]

    def _compute_lengths(tw):
        """Compute expanded lengths for a given tab width."""
        result = []
        for raw in raw_lines:
            if tw and tw > 0 and "\t" in raw:
                result.append(len(raw.expandtabs(tw)))
            else:
                result.append(len(raw))
        result.sort()
        return result

    def _try_snap(lengths, tw, debug):
        """Try to snap the percentile value to a standard. Returns (value, distance) or None."""
        n = len(lengths)
        idx = int(n * percentile)
        detected = lengths[idx]

        snap_distance = SNAP_THRESHOLD + 1
        snap_value = None
        for standard in STANDARD_LIMITS:
            distance = abs(detected - standard)
            if distance <= SNAP_THRESHOLD and distance < snap_distance:
                snap_distance = distance
                snap_value = standard

        if snap_value is None:
            return None

        if debug:
            debug_pcts = [0.50, 0.75, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00]
            pctl_line = "  Column limit percentiles (tab_width=%s):" % (
                tw if tw else "none"
            )
            for p in debug_pcts:
                val = lengths[min(int(n * p), n - 1)]
                pctl_line += " %.0f%%=%d" % (p * 100, val)
            print(pctl_line, file=sys.stderr)
            print(
                "  Detected: %d (at %.0f%%), tab_width=%s, files=%d"
                % (
                    detected,
                    percentile * 100,
                    tw if tw else "none",
                    len(files),
                ),
                file=sys.stderr,
            )
            print(
                "  Snapped %d -> %d (distance %d)"
                % (
                    detected,
                    snap_value,
                    snap_distance,
                ),
                file=sys.stderr,
            )
        return (snap_value, snap_distance)

    # Try each tab width, pick the one that snaps closest.
    best_result = None
    best_snap_distance = SNAP_THRESHOLD + 1
    for tw in tab_widths_to_try:
        lengths = _compute_lengths(tw)
        result = _try_snap(lengths, tw, debug)
        if result is not None and result[1] < best_snap_distance:
            best_snap_distance = result[1]
            best_result = result[0]

    if best_result is not None:
        return best_result

    # No tab width produced a snap. Fall back to the primary tab_width.
    fallback_tw = tab_width if tab_width else None
    lengths = _compute_lengths(fallback_tw)
    n = len(lengths)
    idx = int(n * percentile)
    detected = lengths[idx]

    if debug:
        debug_pcts = [0.50, 0.75, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00]
        pctl_line = "  Column limit percentiles (tab_width=%s):" % (
            fallback_tw if fallback_tw else "none"
        )
        for p in debug_pcts:
            val = lengths[min(int(n * p), n - 1)]
            pctl_line += " %.0f%%=%d" % (p * 100, val)
        print(pctl_line, file=sys.stderr)
        print(
            "  Detected: %d (at %.0f%%), tab_width=%s, files=%d"
            % (
                detected,
                percentile * 100,
                fallback_tw if fallback_tw else "none",
                len(files),
            ),
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
                "  Snapped %d -> %d (distance %d)"
                % (
                    detected,
                    best_standard,
                    best_distance,
                ),
                file=sys.stderr,
            )
        return best_standard

    if debug:
        print(
            "  No snap (all standards >%d away), returning %d"
            % (
                SNAP_THRESHOLD,
                detected,
            ),
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
