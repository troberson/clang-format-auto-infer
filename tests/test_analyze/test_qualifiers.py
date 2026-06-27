"""Tests for detect_qualifier_alignment."""

from pathlib import Path

from src.analyze_conventions.qualifiers import detect_qualifier_alignment


class TestDetectQualifierAlignment:
    def test_left_alignment(self, tmp_path: Path):
        p = tmp_path / "a.c"
        _ = p.write_text("const int x = 0;\nconst char c = 'a';\n")
        assert detect_qualifier_alignment([str(p)]) == "Left"

    def test_right_alignment(self, tmp_path: Path):
        p = tmp_path / "a.c"
        _ = p.write_text("int const x = 0;\nchar const c = 'a';\n")
        assert detect_qualifier_alignment([str(p)]) == "Right"

    def test_mixed_returns_leave(self, tmp_path: Path):
        """50/50 split should return Leave, not Left or Right."""
        p = tmp_path / "a.c"
        _ = p.write_text("const int x = 0;\nint const y = 1;\n")
        result = detect_qualifier_alignment([str(p)])
        assert result == "Leave", (
            f"Expected 'Leave' for mixed qualifiers but got '{result}'. "
            "The 60% dominance threshold should not be met with a 50/50 split."
        )

    def test_no_qualifiers(self, tmp_path: Path):
        p = tmp_path / "a.c"
        _ = p.write_text("int x = 0;\nchar c = 'a';\n")
        assert detect_qualifier_alignment([str(p)]) == "Leave"

    def test_skips_comments(self, tmp_path: Path):
        p = tmp_path / "a.c"
        _ = p.write_text("// const int x = 0;\nint y = 1;\n")
        assert detect_qualifier_alignment([str(p)]) == "Leave"

    def test_regression_mixed_qualifiers_both_counted(self, tmp_path: Path):
        """Regression: 'const int' and 'int const' in the same file must both
        be counted. A bug with 'elif' instead of 'if' caused only one branch
        to fire per qualifier occurrence."""
        p = tmp_path / "a.c"
        _ = p.write_text("const int a = 0;\nconst int b = 1;\nint const c = 2;\n")
        # 2 left, 1 right -> 2/3 = 66.7% >= 60% -> Left
        result = detect_qualifier_alignment([str(p)])
        assert result == "Left", (
            f"Expected 'Left' (2/3 dominance) but got '{result}'. "
            "Qualifier detection must count both left and right patterns."
        )
