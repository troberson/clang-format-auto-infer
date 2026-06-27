"""Common utilities shared across detectors."""

import os


def _source_files(root_dir: str) -> list[str]:  # pyright: ignore[reportUnusedFunction]
    """Find C/C++ source and header files recursively."""
    extensions = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".m", ".mm"}
    files: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in extensions:
                files.append(os.path.join(dirpath, fn))
    return files
