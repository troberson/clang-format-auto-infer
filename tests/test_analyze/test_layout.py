"""Tests for detect_column_limit, detect_max_empty_lines, detect_access_modifier_offset."""

from pathlib import Path

from src.analyze_conventions.layout import (
    detect_access_modifier_offset,
    detect_column_limit,
    detect_max_empty_lines,
)


class TestDetectColumnLimit:
    def test_snap_to_standard_80(self, tmp_path: Path):
        """Lines near 80 should snap to the standard limit 80."""
        p = tmp_path / "a.c"
        _ = p.write_text("x" * 78 + "\n" + "x" * 82 + "\n" + "x" * 79 + "\n")
        result = detect_column_limit([str(p)])
        assert result == 80, f"Expected snap to 80, got {result}"

    def test_no_files(self):
        assert detect_column_limit([]) is None

    def test_skips_very_long_lines(self, tmp_path: Path):
        p = tmp_path / "a.c"
        _ = p.write_text("x" * 250 + "\n" + "x" * 100 + "\n")
        result = detect_column_limit([str(p)])
        assert result == 100

    def test_snap_to_120(self, tmp_path: Path):
        p = tmp_path / "a.c"
        _ = p.write_text("\n".join(["x" * 118, "x" * 122, "x" * 119]) + "\n")
        result = detect_column_limit([str(p)])
        assert result == 120

    def test_no_snap_when_far(self, tmp_path: Path):
        """When detected value is far from any standard, return raw value."""
        p = tmp_path / "a.c"
        _ = p.write_text("\n".join(["x" * 93 for _ in range(20)]) + "\n")
        result = detect_column_limit([str(p)])
        # 93 is >5 away from both 80 and 100
        assert result == 93

    def test_custom_percentile(self, tmp_path: Path):
        """Lower percentile picks shorter lines."""
        p = tmp_path / "a.c"
        lines = ["x" * 50] * 10 + ["x" * 200]  # 200 is filtered out
        _ = p.write_text("\n".join(lines) + "\n")
        result = detect_column_limit([str(p)], percentile=0.50)
        assert result == 50


class TestDetectAccessModifierOffset:
    def test_detects_offset(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("class Foo {\n  public:\n    void bar();\n};\n")
        assert detect_access_modifier_offset([str(p)]) == 2

    def test_no_access_modifiers(self, tmp_path: Path):
        p = tmp_path / "a.c"
        _ = p.write_text("int x = 0;\n")
        assert detect_access_modifier_offset([str(p)]) is None

    def test_multiple_modifiers_picks_most_common(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("class Foo {\n  public:\n  private:\n    protected:\n};\n")
        # indent 2 appears twice, indent 4 once
        assert detect_access_modifier_offset([str(p)]) == 2


class TestDetectMaxEmptyLines:
    def test_consecutive_empty_lines(self, tmp_path: Path):
        p = tmp_path / "a.c"
        _ = p.write_text("int x;\n\n\n\nint y;\n")
        assert detect_max_empty_lines([str(p)]) == 3

    def test_no_empty_lines(self, tmp_path: Path):
        p = tmp_path / "a.c"
        _ = p.write_text("int x;\nint y;\n")
        assert detect_max_empty_lines([str(p)]) == 0

    def test_across_multiple_files(self, tmp_path: Path):
        p = tmp_path / "a.c"
        _ = p.write_text("int x;\n\nint y;\n")
        p = tmp_path / "b.c"
        _ = p.write_text("int a;\n\n\n\nint b;\n")
        files = [str(tmp_path / "a.c"), str(tmp_path / "b.c")]
        assert detect_max_empty_lines(files) == 3
