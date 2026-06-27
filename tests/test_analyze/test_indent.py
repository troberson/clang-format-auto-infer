"""Tests for detect_indent_width and detect_use_tab."""

from pathlib import Path

from src.analyze_conventions.indent import detect_indent_width, detect_use_tab


class TestDetectIndentWidth:
    def test_single_level_4_spaces(self, tmp_path: Path):
        """Single indent level of 4 spaces should detect width 4."""
        p = tmp_path / "a.c"
        _ = p.write_text("int main() {\n    return 0;\n}\n")
        assert detect_indent_width([str(p)]) == 4

    def test_single_level_2_spaces(self, tmp_path: Path):
        p = tmp_path / "a.c"
        _ = p.write_text("int main() {\n  return 0;\n}\n")
        assert detect_indent_width([str(p)]) == 2

    def test_multiple_levels_4_spaces(self, tmp_path: Path):
        """Multiple indent levels (4, 8) should detect width 4."""
        p = tmp_path / "a.c"
        _ = p.write_text(
            "int main() {\n    if (x) {\n        return 1;\n    }\n    return 0;\n}\n"
        )
        assert detect_indent_width([str(p)]) == 4

    def test_tabs(self, tmp_path: Path):
        p = tmp_path / "a.c"
        _ = p.write_text("int main() {\n\treturn 0;\n}\n")
        assert detect_indent_width([str(p)]) == 4

    def test_empty_files_fallback(self, tmp_path: Path):
        p = tmp_path / "a.c"
        _ = p.write_text("")
        assert detect_indent_width([str(p)]) == 4

    def test_mixed_tabs_and_spaces_tabs_dominant(self, tmp_path: Path):
        """When tabs outnumber space indents, prefer tab width."""
        p = tmp_path / "a.c"
        _ = p.write_text("\tline1;\n\tline2;\n  line3;\n")
        assert detect_indent_width([str(p)]) == 4

    def test_regression_single_indent_prefers_largest_divisor(self, tmp_path: Path):
        """Regression: single 4-space indent should NOT resolve to width 2.

        The width-scoring algorithm checks widths 2-8. Both 2 and 4 divide
        an indent of 4 evenly. The largest divisor should win.
        """
        p = tmp_path / "a.c"
        _ = p.write_text("int main() {\n    return 0;\n}\n")
        result = detect_indent_width([str(p)])
        assert result == 4, (
            f"Expected indent width 4 but got {result}. "
            "The algorithm should prefer the largest width that divides "
            "the most indent levels, not the smallest."
        )


class TestDetectUseTab:
    def test_tabs_dominant(self, tmp_path: Path):
        p = tmp_path / "a.c"
        _ = p.write_text("\tint x;\n\t\tint y;\n  int z;\n")
        assert detect_use_tab([str(p)]) == "Always"

    def test_spaces_dominant(self, tmp_path: Path):
        p = tmp_path / "a.c"
        _ = p.write_text("    int x;\n        int y;\n\tint z;\n")
        assert detect_use_tab([str(p)]) == "Never"

    def test_mixed_returns_leave(self, tmp_path: Path):
        p = tmp_path / "a.c"
        _ = p.write_text("\tint x;\n    int y;\n")
        assert detect_use_tab([str(p)]) == "Leave"

    def test_empty_file(self, tmp_path: Path):
        p = tmp_path / "a.c"
        _ = p.write_text("")
        assert detect_use_tab([str(p)]) == "Leave"

    def test_skips_empty_lines(self, tmp_path: Path):
        p = tmp_path / "a.c"
        _ = p.write_text("\n\n\tint x;\n\n\tint y;\n")
        assert detect_use_tab([str(p)]) == "Always"
