"""Tests for detect_sort_includes."""

from pathlib import Path

from src.analyze_conventions.includes import detect_sort_includes


class TestDetectSortIncludes:
    def test_sorted_includes(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text(
            '#include "bar.h"\n#include "foo.h"\n#include <stdio.h>\nint main() {}\n'
        )
        assert detect_sort_includes([str(p)]) is True

    def test_unsorted_includes(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text('#include "b.h"\n#include "a.h"\n')
        assert detect_sort_includes([str(p)]) is False

    def test_unsorted_block(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text(
            '#include "b.h"\n#include "a.h"\nint x;\n#include <c.h>\n#include <d.h>\n'
        )
        assert detect_sort_includes([str(p)]) is False

    def test_single_include_returns_false(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("#include <stdio.h>\nint main() {}\n")
        assert detect_sort_includes([str(p)]) is False

    def test_no_includes_returns_false(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("int x = 1;\n")
        assert detect_sort_includes([str(p)]) is False

    def test_multiple_blocks_majority_sorted(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text(
            '#include "a.h"\n#include "b.h"\nint x;\n#include <c.h>\n#include <d.h>\n'
        )
        assert detect_sort_includes([str(p)]) is True
