"""Integration tests for analyze() and package exports."""

from pathlib import Path

from src.analyze_conventions import analyze


def _write_file(tmp_path: Path, name: str, content: str) -> str:
    p = tmp_path / name
    _ = p.write_text(content)
    return str(p)


class TestAnalyze:
    def test_returns_empty_for_no_files(self, tmp_path: Path):
        result = analyze(str(tmp_path))
        assert result == {}

    def test_returns_detected_conventions(self, tmp_path: Path):
        _ = _write_file(
            tmp_path,
            "a.cpp",
            "int x = 0xABCD;\nfloat y = 1e2;\nint z = 1u;\n",
        )
        result = analyze(str(tmp_path))
        assert "Language" in result
        assert "IndentWidth" in result
        assert "ColumnLimit" in result
        assert "QualifierAlignment" in result
        assert "MaxEmptyLinesToKeep" in result
        assert result["NumericLiteralCase.HexDigit"] == "Upper"
        assert result["NumericLiteralCase.ExponentLetter"] == "Lower"
        assert result["NumericLiteralCase.Suffix"] == "Lower"

    def test_excludes_leave_values(self, tmp_path: Path):
        _ = _write_file(
            tmp_path,
            "a.cpp",
            "int x = 0xABCD;\nint y = 0xabcd;\n",
        )
        result = analyze(str(tmp_path))
        # Mixed hex digits -> Leave, so key should not be present
        assert "NumericLiteralCase.HexDigit" not in result

    def test_excludes_none_access_offset(self, tmp_path: Path):
        _ = _write_file(tmp_path, "a.cpp", "int x = 1;\n")
        result = analyze(str(tmp_path))
        # No access modifiers -> None, so key should not be present
        assert "AccessModifierOffset" not in result

    def test_includes_access_offset_when_present(self, tmp_path: Path):
        _ = _write_file(
            tmp_path,
            "a.cpp",
            "class Foo {\npublic:\n    int x;\n};\n",
        )
        result = analyze(str(tmp_path))
        assert "AccessModifierOffset" in result

    def test_returns_all_numeric_literal_fields(self, tmp_path: Path):
        _ = _write_file(
            tmp_path,
            "a.cpp",
            "int x = 0xABCD;\nint y = 0X1;\nint z = 0X2;\nfloat w = 1e2;\nint v = 1U;\n",
        )
        result = analyze(str(tmp_path))
        assert "NumericLiteralCase.HexDigit" in result
        assert "NumericLiteralCase.Prefix" in result
        assert "NumericLiteralCase.ExponentLetter" in result
        assert "NumericLiteralCase.Suffix" in result

    def test_split_brace_style_emits_custom(self, tmp_path: Path):
        # K&R for control, Allman for functions -> Custom + sub-options
        _ = _write_file(
            tmp_path,
            "a.cpp",
            "void foo()\n{\n    if (x) {\n        y;\n    }\n}\n",
        )
        result = analyze(str(tmp_path))
        assert result["BreakBeforeBraces"] == "Custom"
        assert result["BraceWrapping.AfterControlStatement"] == "Never"
        assert result["BraceWrapping.AfterFunction"] is True

    def test_same_brace_style_emits_sub_options(self, tmp_path: Path):
        # All K&R -> Custom + both sub-options set to Never
        _ = _write_file(
            tmp_path,
            "a.cpp",
            "void foo() {\n    if (x) {\n        y;\n    }\n}\n",
        )
        result = analyze(str(tmp_path))
        assert result["BreakBeforeBraces"] == "Custom"
        assert result["BraceWrapping.AfterControlStatement"] == "Never"
        assert result["BraceWrapping.AfterFunction"] is False

    def test_analyze_includes_reflow_comments(self, tmp_path: Path):
        # Comments within column limit -> ReflowComments: Never
        _ = _write_file(
            tmp_path,
            "a.cpp",
            "// short comment\nint x = 1;\n",
        )
        result = analyze(str(tmp_path))
        assert result["ReflowComments"] == "Never"

    def test_tab_width_matches_indent_when_use_tab(self, tmp_path: Path):
        # Tabs with 4-space indent -> TabWidth: 4
        _ = _write_file(
            tmp_path,
            "a.cpp",
            "\tint x;\n\t\tint y;\n",
        )
        result = analyze(str(tmp_path))
        assert result["UseTab"] == "Always"
        assert result["TabWidth"] == result["IndentWidth"]

    def test_emits_align_case_labels(self, tmp_path: Path):
        _ = _write_file(tmp_path, "a.cpp", "    case 1: foo();\n    case 2: bar();\n")
        result = analyze(str(tmp_path))
        assert "AlignConsecutiveShortCaseStatements.Enabled" in result

    def test_emits_align_trailing_comments(self, tmp_path: Path):
        _ = _write_file(
            tmp_path,
            "a.cpp",
            "int x = 5;        // comment\nint y = 10;       // comment\n",
        )
        result = analyze(str(tmp_path))
        assert "AlignTrailingComments.Kind" in result


class TestPackageExports:
    """Verify the package re-exports all symbols."""

    def test_exports_analyze(self):
        import src.analyze_conventions as pkg

        assert hasattr(pkg, "analyze")
        assert callable(pkg.analyze)

    def test_exports_detectors(self):
        import src.analyze_conventions as pkg

        for name in [
            "_source_files",
            "detect_indent_width",
            "detect_column_limit",
            "detect_language",
            "detect_qualifier_alignment",
            "detect_access_modifier_offset",
            "detect_max_empty_lines",
            "detect_numeric_literal_case",
            "detect_brace_style",
            "detect_brace_style_control",
            "detect_brace_style_function",
            "detect_use_tab",
            "detect_reflow_comments",
            "detect_sort_includes",
            "detect_allow_short_functions",
            "detect_allow_short_blocks",
            "detect_align_consecutive_declarations",
            "detect_indent_goto_labels",
            "detect_align_case_labels",
            "detect_trailing_comment_style",
        ]:
            assert hasattr(pkg, name), f"package missing {name}"
            assert callable(getattr(pkg, name)), f"{name} not callable"

    def test_analyze_works(self, tmp_path: Path):
        import src.analyze_conventions as pkg

        p = tmp_path / "a.cpp"
        _ = p.write_text("int x = 1;\n")
        result = pkg.analyze(str(tmp_path))
        assert "Language" in result
