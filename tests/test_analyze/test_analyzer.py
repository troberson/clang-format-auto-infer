"""Integration tests for analyze() and package exports."""

from pathlib import Path

from src.analyze_conventions import DetectedOption, analyze, analyze_with_metadata


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


class TestDetectedOption:
    """Unit tests for the DetectedOption dataclass."""

    def test_equality_with_raw_value(self):
        opt = DetectedOption(4, "detected")
        assert opt == 4
        assert opt != 8

    def test_equality_with_another_detected_option(self):
        a = DetectedOption(True, "detected", "structure")
        b = DetectedOption(True, "forced", "polish")
        assert a == b  # compares only .value

    def test_hashable(self):
        opt = DetectedOption("Custom", "forced")
        assert hash(opt) == hash("Custom")
        # Must be usable as dict key / in set
        s = {opt}
        assert DetectedOption("Custom", "detected") in s

    def test_default_tier_is_polish(self):
        opt = DetectedOption(42, "detected")
        assert opt.tier == "polish"

    def test_explicit_tier(self):
        opt = DetectedOption(42, "detected", "structure")
        assert opt.tier == "structure"


class TestAnalyzeWithMetadata:
    """Tests for analyze_with_metadata() returning DetectedOption objects."""

    def test_returns_detectedoption_objects(self, tmp_path: Path):
        _ = _write_file(tmp_path, "a.cpp", "int x = 1;\n")
        result = analyze_with_metadata(str(tmp_path))
        assert "Language" in result
        assert isinstance(result["Language"], DetectedOption)

    def test_empty_directory(self, tmp_path: Path):
        result = analyze_with_metadata(str(tmp_path))
        assert result == {}

    def test_confidence_guessed_for_indent_width_pure_tab(self, tmp_path: Path):
        _ = _write_file(tmp_path, "a.cpp", "\tint x;\n\t\tint y;\n")
        result = analyze_with_metadata(str(tmp_path))
        assert result["IndentWidth"].confidence == "guessed"

    def test_confidence_guessed_for_tab_width_pure_tab(self, tmp_path: Path):
        _ = _write_file(tmp_path, "a.cpp", "\tint x;\n\t\tint y;\n")
        result = analyze_with_metadata(str(tmp_path))
        assert "TabWidth" in result
        assert result["TabWidth"].confidence == "guessed"

    def test_confidence_detected_for_indent_width_spaces(self, tmp_path: Path):
        _ = _write_file(tmp_path, "a.cpp", "    int x;\n        int y;\n")
        result = analyze_with_metadata(str(tmp_path))
        assert result["IndentWidth"].confidence == "detected"

    def test_forced_confidence_for_custom_options(self, tmp_path: Path):
        _ = _write_file(tmp_path, "a.cpp", "int x = 1;\n")
        result = analyze_with_metadata(str(tmp_path))
        assert result["BreakBeforeBraces"].confidence == "forced"
        assert result["SpaceBeforeParens"].confidence == "forced"
        assert result["SpacesInParens"].confidence == "forced"

    def test_structure_tier_for_brace_wrapping(self, tmp_path: Path):
        _ = _write_file(
            tmp_path,
            "a.cpp",
            "void foo()\n{\n    if (x) {\n        y;\n    }\n}\n",
        )
        result = analyze_with_metadata(str(tmp_path))
        assert result["BraceWrapping.AfterControlStatement"].tier == "structure"
        assert result["BraceWrapping.AfterFunction"].tier == "structure"

    def test_resolve_tier_for_guessed_indent(self, tmp_path: Path):
        _ = _write_file(tmp_path, "a.cpp", "\tint x;\n\t\tint y;\n")
        result = analyze_with_metadata(str(tmp_path))
        assert result["IndentWidth"].tier == "resolve"
        assert result["TabWidth"].tier == "resolve"

    def test_structure_tier_for_space_indent(self, tmp_path: Path):
        _ = _write_file(tmp_path, "a.cpp", "    int x;\n        int y;\n")
        result = analyze_with_metadata(str(tmp_path))
        assert result["IndentWidth"].tier == "structure"

    def test_values_match_analyze(self, tmp_path: Path):
        """analyze_with_metadata values must match analyze() output."""
        _ = _write_file(
            tmp_path,
            "a.cpp",
            "int x = 0xABCD;\nfloat y = 1e2;\nint z = 1u;\n"
            + "void foo()\n{\n    if (x) {\n        y;\n    }\n}\n",
        )
        plain = analyze(str(tmp_path))
        meta = analyze_with_metadata(str(tmp_path))
        for key, value in plain.items():
            assert key in meta, f"key {key} missing from metadata result"
            assert meta[key].value == value, f"value mismatch for {key}"

    def test_structure_tier_for_column_limit(self, tmp_path: Path):
        _ = _write_file(tmp_path, "a.cpp", "int x = 1;\n")
        result = analyze_with_metadata(str(tmp_path))
        assert result["ColumnLimit"].tier == "structure"

    def test_structure_tier_for_use_tab(self, tmp_path: Path):
        _ = _write_file(tmp_path, "a.cpp", "\tint x;\n\t\tint y;\n")
        result = analyze_with_metadata(str(tmp_path))
        assert result["UseTab"].tier == "structure"

    def test_structure_tier_for_allow_short_functions(self, tmp_path: Path):
        _ = _write_file(
            tmp_path,
            "a.cpp",
            "void f() { return; }\nvoid g() { return; }\n",
        )
        result = analyze_with_metadata(str(tmp_path))
        if "AllowShortFunctionsOnASingleLine" in result:
            assert result["AllowShortFunctionsOnASingleLine"].tier == "structure"

    def test_structure_tier_for_align_trailing_comments(self, tmp_path: Path):
        _ = _write_file(
            tmp_path,
            "a.cpp",
            "int x = 5;        // comment\nint y = 10;       // comment\n",
        )
        result = analyze_with_metadata(str(tmp_path))
        if "AlignTrailingComments.Kind" in result:
            assert result["AlignTrailingComments.Kind"].tier == "structure"

    def test_metadata_includes_access_offset(self, tmp_path: Path):
        _ = _write_file(
            tmp_path,
            "a.cpp",
            "class Foo {\npublic:\n    int x;\n};\n",
        )
        result = analyze_with_metadata(str(tmp_path))
        assert "AccessModifierOffset" in result
        assert isinstance(result["AccessModifierOffset"], DetectedOption)

    def test_metadata_includes_reflow_comments(self, tmp_path: Path):
        _ = _write_file(
            tmp_path,
            "a.cpp",
            "// short comment\nint x = 1;\n",
        )
        result = analyze_with_metadata(str(tmp_path))
        assert "ReflowComments" in result
        assert isinstance(result["ReflowComments"], DetectedOption)

    def test_metadata_includes_indent_goto_labels(self, tmp_path: Path):
        _ = _write_file(
            tmp_path,
            "a.c",
            "void f() {\n    goto end;\nend:\n    return;\n}\n",
        )
        result = analyze_with_metadata(str(tmp_path))
        assert "IndentGotoLabels" in result
        assert isinstance(result["IndentGotoLabels"], DetectedOption)

    def test_metadata_includes_align_case_labels(self, tmp_path: Path):
        _ = _write_file(
            tmp_path,
            "a.c",
            "    case 1: foo();\n    case 2: bar();\n",
        )
        result = analyze_with_metadata(str(tmp_path))
        assert "AlignConsecutiveShortCaseStatements.Enabled" in result
        assert isinstance(
            result["AlignConsecutiveShortCaseStatements.Enabled"], DetectedOption
        )
