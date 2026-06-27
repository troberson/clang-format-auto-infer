"""Comment reflow detection."""

import re


def detect_reflow_comments(files: list[str], column_limit: int | None = None) -> str:
    """Detect whether comments should be reflowed.

    If the majority of comment lines are already within the column limit,
    the codebase likely has hand-formatted comments that should be preserved.
    Returns 'Never' if comments are already well-formatted, 'Always' if many
    exceed the limit, or 'Leave' if unclear.

    Args:
        files: List of source files to analyze.
        column_limit: The detected column limit. If None, defaults to 80.

    Returns:
        'Never', 'Always', or 'Leave'.
    """
    if column_limit is None:
        column_limit = 80

    comment_pattern = re.compile(r"^\s*(//|/\*|\*)")

    within_limit = 0
    exceeds_limit = 0

    for fpath in files:
        try:
            with open(fpath, errors="replace") as f:
                for line in f:
                    if not comment_pattern.match(line):
                        continue
                    visual = line.rstrip("\n\r").expandtabs(4)
                    if len(visual) <= column_limit:
                        within_limit += 1
                    else:
                        exceeds_limit += 1
        except OSError:  # pragma: no cover
            continue

    total = within_limit + exceeds_limit
    if total == 0:
        return "Leave"

    # If 70%+ of comment lines are already within the limit, preserve them.
    if within_limit / total >= 0.7:
        return "Never"
    if exceeds_limit / total >= 0.6:
        return "Always"
    return "Leave"
