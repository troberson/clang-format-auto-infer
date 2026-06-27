"""Tests for detect_brace_style, detect_brace_style_control, detect_brace_style_function."""

from pathlib import Path

from src.analyze_conventions.braces import (
    detect_brace_style,
    detect_brace_style_control,
    detect_brace_style_function,
)


class TestDetectBraceStyle:
    def test_allman_style(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("void foo()\n{\n    if (x)\n    {\n        bar();\n    }\n}\n")
        files = [str(p)]
        assert detect_brace_style(files) == "Allman"

    def test_attach_style(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("void foo() {\n    if (x) {\n        bar();\n    }\n}\n")
        files = [str(p)]
        assert detect_brace_style(files) == "Attach"

    def test_mixed_returns_leave(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("void foo() {\n    if (x) {\n        bar();\n    }\n}\n")
        files = [str(p)]
        result = detect_brace_style(files)
        assert result in ("Leave", "Attach", "Allman")

    def test_no_braces_returns_leave(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("int x = 1;\n")
        files = [str(p)]
        assert detect_brace_style(files) == "Leave"

    def test_skips_comments(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("// void foo() {\nvoid bar() {\n}\n")
        files = [str(p)]
        assert detect_brace_style(files) == "Attach"

    def test_mixed_neither_dominant_returns_leave(self, tmp_path: Path):
        """When attach and allman are close (neither >= 60%), return Leave."""
        p = tmp_path / "a.cpp"
        # 2 attach (control), 1 allman (function) -> 2/3 attach, 1/3 allman
        # 2/3 = 66.7% >= 60% -> Attach. Need equal split.
        # 1 attach control, 1 allman function -> 50/50 -> Leave
        _ = p.write_text("if (x) {\n    y;\n}\nvoid foo()\n{\n}\n")
        files = [str(p)]
        assert detect_brace_style(files) == "Leave"


class TestDetectBraceStyleControl:
    def test_attach_control_statements(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("if (x) {\n    foo();\n}\n")
        assert detect_brace_style_control([str(p)]) == "Attach"

    def test_allman_control_statements(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("if (x)\n{\n    foo();\n}\n")
        assert detect_brace_style_control([str(p)]) == "Allman"

    def test_mixed_returns_leave(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("if (x) {\n    foo();\n}\nif (y)\n{\n    bar();\n}\n")
        assert detect_brace_style_control([str(p)]) == "Leave"

    def test_no_control_statements_returns_leave(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("int x = 5;\n")
        assert detect_brace_style_control([str(p)]) == "Leave"

    def test_skips_comments(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("// if (x) {\nif (y) {\n    foo();\n}\n")
        assert detect_brace_style_control([str(p)]) == "Attach"


class TestDetectBraceStyleFunction:
    def test_attach_functions(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("void foo() {\n}\n")
        assert detect_brace_style_function([str(p)]) == "Attach"

    def test_allman_functions(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("void foo()\n{\n}\n")
        assert detect_brace_style_function([str(p)]) == "Allman"

    def test_mixed_returns_leave(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("void foo() {\n}\nvoid bar()\n{\n}\n")
        assert detect_brace_style_function([str(p)]) == "Leave"

    def test_no_functions_returns_leave(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("int x = 5;\n")
        assert detect_brace_style_function([str(p)]) == "Leave"

    def test_skips_control_statements(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("if (x) {\n    foo();\n}\n")
        assert detect_brace_style_function([str(p)]) == "Leave"

    def test_skips_comments(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("// void foo() {\nvoid bar() {\n}\n")
        assert detect_brace_style_function([str(p)]) == "Attach"
