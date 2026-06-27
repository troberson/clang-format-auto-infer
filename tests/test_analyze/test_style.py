"""Tests for BSD-style churn detectors in style.py."""

from pathlib import Path

from src.analyze_conventions.style import (
    detect_allow_short_blocks,
    detect_allow_short_functions,
    detect_align_case_labels,
    detect_align_consecutive_declarations,
    detect_indent_goto_labels,
    detect_trailing_comment_style,
)


class TestDetectAllowShortFunctions:
    def test_single_line_functions_returns_all(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text(
            "int foo() { return 0; }\nint bar() { return 1; }\nint baz() { return 2; }\n"
        )
        assert detect_allow_short_functions([str(p)]) == "All"

    def test_multiline_functions_returns_none(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text(
            "int foo()\n{\n    return 0;\n}\nint bar()\n{\n    return 1;\n}\n"
        )
        assert detect_allow_short_functions([str(p)]) == "None"

    def test_mixed_returns_leave(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("int foo() { return 0; }\nint bar()\n{\n    return 1;\n}\n")
        assert detect_allow_short_functions([str(p)]) == "Leave"

    def test_no_functions_returns_leave(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("int x = 5;\n")
        assert detect_allow_short_functions([str(p)]) == "Leave"

    def test_skips_control_statements(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("if (x) { return; }\nwhile (y) { break; }\n")
        assert detect_allow_short_functions([str(p)]) == "Leave"

    def test_skips_comments(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("// int foo() { return 0; }\nint bar() { return 1; }\n")
        assert detect_allow_short_functions([str(p)]) == "All"

    def test_allman_with_blank_lines(self, tmp_path: Path):
        """Allman function with blank lines between signature and brace."""
        p = tmp_path / "a.cpp"
        _ = p.write_text("int foo()\n\n{\n    return 0;\n}\n")
        assert detect_allow_short_functions([str(p)]) == "None"


class TestDetectAllowShortBlocks:
    def test_single_line_blocks_returns_all(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("if (x) { foo(); }\nif (y) { bar(); }\nif (z) { baz(); }\n")
        assert detect_allow_short_blocks([str(p)]) == "All"

    def test_multiline_blocks_returns_never(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("if (x)\n{\n    foo();\n}\nif (y)\n{\n    bar();\n}\n")
        assert detect_allow_short_blocks([str(p)]) == "Never"

    def test_mixed_returns_leave(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("if (x) { foo(); }\nif (y)\n{\n    bar();\n}\n")
        assert detect_allow_short_blocks([str(p)]) == "Leave"

    def test_no_blocks_returns_leave(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("int x = 5;\n")
        assert detect_allow_short_blocks([str(p)]) == "Leave"

    def test_skips_nested_braces(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("void foo() { if (x) { bar(); } }\n")
        # Nested braces should not count as single-line block
        # The outer function brace has no closing on same line -> multiline
        # The inner if brace is single-line but nested, skipped by inner check
        assert detect_allow_short_blocks([str(p)]) == "Never"

    def test_skips_comments(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("// if (x) { foo(); }\nif (y) { bar(); }\n")
        assert detect_allow_short_blocks([str(p)]) == "All"


class TestDetectAlignConsecutiveDeclarations:
    def test_aligned_declarations_returns_true(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("int     x;\nint     y;\nchar    c;\n")
        assert detect_align_consecutive_declarations([str(p)]) is True

    def test_unaligned_declarations_returns_false(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("int x;\nint y;\nchar c;\n")
        assert detect_align_consecutive_declarations([str(p)]) is False

    def test_mixed_groups_returns_none(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("int     x;\nint     y;\nvoid foo();\nchar c;\nchar    d;\n")
        # First group aligned (pos 8,8), second group unaligned (pos 5,8)
        assert detect_align_consecutive_declarations([str(p)]) is None

    def test_no_consecutive_declarations(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("int x = 5;\n")
        assert detect_align_consecutive_declarations([str(p)]) is None

    def test_single_declaration_ignored(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("int x;\nvoid foo() {}\nint y;\n")
        assert detect_align_consecutive_declarations([str(p)]) is None


class TestDetectIndentGotoLabels:
    def test_labels_at_column_zero_returns_false(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("start:\n    goto end;\nend:\n    return 0;\n")
        assert detect_indent_goto_labels([str(p)]) is False

    def test_indented_labels_returns_true(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("    start:\n        goto end;\n    end:\n        return 0;\n")
        assert detect_indent_goto_labels([str(p)]) is True

    def test_mixed_returns_none(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("start:\n    goto end;\n    end:\n")
        assert detect_indent_goto_labels([str(p)]) is None

    def test_no_labels_returns_none(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("int x = 5;\n")
        assert detect_indent_goto_labels([str(p)]) is None

    def test_skips_case_labels(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("    case 1:\n        foo();\n    default:\n        bar();\n")
        assert detect_indent_goto_labels([str(p)]) is None


class TestDetectAlignCaseLabels:
    def test_aligned_case_statements_returns_true(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("    case 1: foo();\n    case 2: bar();\n    case 3: baz();\n")
        assert detect_align_case_labels([str(p)]) is True

    def test_unaligned_case_statements_returns_false(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text(
            "    case 1:    foo();\n    case 2: bar();\n    case 3:    baz();\n"
        )
        assert detect_align_case_labels([str(p)]) is False

    def test_no_case_labels_returns_none(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("int x = 5;\n")
        assert detect_align_case_labels([str(p)]) is None

    def test_single_case_ignored(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("    case 1: foo();\n")
        assert detect_align_case_labels([str(p)]) is None

    def test_mixed_case_alignment_returns_none(self, tmp_path: Path):
        """Mixed aligned and unaligned case groups return None."""
        p = tmp_path / "a.cpp"
        # 1 aligned group, 1 unaligned group -> 50/50 -> None
        _ = p.write_text(
            "    case 1: foo();\n    case 2: bar();\n    void other();\n    case 3:    baz();\n    case 4: qux();\n"
        )
        assert detect_align_case_labels([str(p)]) is None


class TestDetectTrailingCommentStyle:
    def test_aligned_trailing_comments_returns_always(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text(
            "int x = 5;       // initialization\nint y = 10;      // another value\nchar c = 'a';    // character\n"
        )
        assert detect_trailing_comment_style([str(p)]) == "Always"

    def test_unaligned_trailing_comments_returns_never(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text(
            "int x = 5; // init\nint y = 10;      // value\nchar c = 'a';   // char\n"
        )
        assert detect_trailing_comment_style([str(p)]) == "Never"

    def test_mixed_returns_leave(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text(
            "int x = 5;        // aligned\nint y = 10;       // also aligned\nchar c = 'a'; // not aligned\n"
        )
        # Positions: 18, 18 (aligned), then 14 (unaligned vs prev)
        # 1 aligned pair, 1 unaligned pair -> Leave
        assert detect_trailing_comment_style([str(p)]) == "Leave"

    def test_no_trailing_comments_returns_leave(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("int x = 5;\n")
        assert detect_trailing_comment_style([str(p)]) == "Leave"

    def test_skips_standalone_comments(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text(
            "// this is a standalone comment\nint x = 5;        // trailing\nint y = 10;       // trailing\n"
        )
        assert detect_trailing_comment_style([str(p)]) == "Always"

    def test_stops_at_non_comment_line(self, tmp_path: Path):
        """Trailing comment detection stops searching at non-comment code lines."""
        p = tmp_path / "a.cpp"
        _ = p.write_text(
            "int x = 5;        // comment\nint y = 10;       // comment\nint z = 15;\nint w = 20; // different pos\n"
        )
        # First pair is aligned (pos 18,18), z has no comment, w is standalone
        assert detect_trailing_comment_style([str(p)]) == "Always"

    def test_skips_empty_lines_between_comments(self, tmp_path: Path):
        """Trailing comment detection skips empty lines when searching for next comment."""
        p = tmp_path / "a.cpp"
        _ = p.write_text(
            "int x = 5;        // comment\n\nint y = 10;       // comment\n"
        )
        # Empty line between should be skipped, comments at pos 18,18 -> aligned
        assert detect_trailing_comment_style([str(p)]) == "Always"
