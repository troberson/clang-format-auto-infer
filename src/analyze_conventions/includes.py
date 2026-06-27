"""Include sorting detection."""

import re


def detect_sort_includes(files: list[str]) -> bool:
    """Detect whether include blocks appear sorted.

    Returns True if includes appear to be sorted, False if not.
    """
    include_pattern = re.compile(r'#\s*include\s*[<"]([^>"]+)[>"]')

    sorted_count = 0
    unsorted_count = 0

    for fpath in files:
        try:
            with open(fpath, errors="replace") as f:
                includes = []
                for line in f:
                    m = include_pattern.match(line.strip())
                    if m:
                        includes.append(m.group(1))

                # Check if includes in this file are sorted
                if len(includes) >= 2:
                    if includes == sorted(includes):
                        sorted_count += 1
                    else:
                        unsorted_count += 1
        except OSError:  # pragma: no cover
            continue

    total = sorted_count + unsorted_count
    if total == 0:
        return False
    if sorted_count / total >= 0.6:
        return True
    return False
