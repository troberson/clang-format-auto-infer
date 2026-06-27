"""Language detection from file extensions."""

import os
from collections import Counter


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
