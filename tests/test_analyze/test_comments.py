"""Tests for detect_reflow_comments."""

from pathlib import Path

from src.analyze_conventions.comments import detect_reflow_comments


class TestDetectReflowComments:
    def test_comments_within_limit_returns_never(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("// short comment\n/* also short */\nint x;\n")
        assert detect_reflow_comments([str(p)], 80) == "Never"

    def test_comments_exceed_limit_returns_always(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text(("// " + "x" * 120 + "\n") * 3)
        assert detect_reflow_comments([str(p)], 80) == "Always"

    def test_mixed_returns_leave(self, tmp_path: Path):
        # 3 short, 2 long -> 60% within (below 70% threshold), 40% exceeds (below 60% threshold)
        p = tmp_path / "a.cpp"
        _ = p.write_text("// short\n" * 3 + ("// " + "x" * 120 + "\n") * 2)
        assert detect_reflow_comments([str(p)], 80) == "Leave"

    def test_no_comments_returns_leave(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("int x = 5;\n")
        assert detect_reflow_comments([str(p)], 80) == "Leave"

    def test_uses_custom_column_limit(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("// this is a comment that is 50 chars wide\n")
        assert detect_reflow_comments([str(p)], 60) == "Never"
        assert detect_reflow_comments([str(p)], 40) == "Always"

    def test_default_column_limit_80(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("// short\n")
        assert detect_reflow_comments([str(p)], None) == "Never"
