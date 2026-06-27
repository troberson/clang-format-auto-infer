"""Tests for analyze_conventions — detection heuristics with temp files."""

import os
from pathlib import Path


_project_root = Path(__file__).resolve().parent.parent

from src.analyze_conventions import (  # noqa: E402
    _source_files,  # pyright: ignore[reportPrivateUsage]
    detect_indent_width,
    detect_column_limit,
    detect_language,
    detect_qualifier_alignment,
    detect_access_modifier_offset,
    detect_max_empty_lines,
    detect_numeric_literal_case,
)


def _write_file(tmp_path: Path, name: str, content: str) -> str:
    p = tmp_path / name
    _ = p.write_text(content)
    return str(p)


# ---------------------------------------------------------------------------
# find_source_files
# ---------------------------------------------------------------------------


class TestFindSourceFiles:
    def test_finds_c_and_cpp(self, tmp_path: Path):
        _ = _write_file(tmp_path, "a.c", "int x;")
        _ = _write_file(tmp_path, "b.cpp", "int y;")
        _ = _write_file(tmp_path, "c.txt", "not code")
        files = _source_files(str(tmp_path))
        names = {os.path.basename(f) for f in files}
        assert "a.c" in names
        assert "b.cpp" in names
        assert "c.txt" not in names

    def test_recursive(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        _ = _write_file(sub, "deep.h", "// header")
        files = _source_files(str(tmp_path))
        assert any("deep.h" in f for f in files)

    def test_empty_directory(self, tmp_path: Path):
        assert _source_files(str(tmp_path)) == []

    def test_objective_c_extensions(self, tmp_path: Path):
        _ = _write_file(tmp_path, "a.m", "@interface Foo")
        _ = _write_file(tmp_path, "b.mm", "#import <Foundation/Foundation.h>")
        files = _source_files(str(tmp_path))
        names = {os.path.basename(f) for f in files}
        assert "a.m" in names
        assert "b.mm" in names


# ---------------------------------------------------------------------------
# detect_indent_width
# ---------------------------------------------------------------------------


class TestDetectIndentWidth:
    def test_single_level_4_spaces(self, tmp_path: Path):
        """Single indent level of 4 spaces should detect width 4."""
        _ = _write_file(tmp_path, "a.c", "int main() {\n    return 0;\n}\n")
        assert detect_indent_width([str(tmp_path / "a.c")]) == 4

    def test_single_level_2_spaces(self, tmp_path: Path):
        _ = _write_file(tmp_path, "a.c", "int main() {\n  return 0;\n}\n")
        assert detect_indent_width([str(tmp_path / "a.c")]) == 2

    def test_multiple_levels_4_spaces(self, tmp_path: Path):
        """Multiple indent levels (4, 8) should detect width 4."""
        content = (
            "int main() {\n    if (x) {\n        return 1;\n    }\n    return 0;\n}\n"
        )
        _ = _write_file(tmp_path, "a.c", content)
        assert detect_indent_width([str(tmp_path / "a.c")]) == 4

    def test_tabs(self, tmp_path: Path):
        _ = _write_file(tmp_path, "a.c", "int main() {\n\treturn 0;\n}\n")
        assert detect_indent_width([str(tmp_path / "a.c")]) == 4

    def test_empty_files_fallback(self, tmp_path: Path):
        _ = _write_file(tmp_path, "a.c", "")
        assert detect_indent_width([str(tmp_path / "a.c")]) == 4

    def test_mixed_tabs_and_spaces_tabs_dominant(self, tmp_path: Path):
        """When tabs outnumber space indents, prefer tab width."""
        content = "\tline1;\n\tline2;\n  line3;\n"
        _ = _write_file(tmp_path, "a.c", content)
        assert detect_indent_width([str(tmp_path / "a.c")]) == 4

    def test_regression_single_indent_prefers_largest_divisor(self, tmp_path: Path):
        """Regression: single 4-space indent should NOT resolve to width 2.

        The width-scoring algorithm checks widths 2-8. Both 2 and 4 divide
        an indent of 4 evenly. The largest divisor should win.
        """
        _ = _write_file(tmp_path, "a.c", "int main() {\n    return 0;\n}\n")
        result = detect_indent_width([str(tmp_path / "a.c")])
        assert result == 4, (
            f"Expected indent width 4 but got {result}. "
            "The algorithm should prefer the largest width that divides "
            "the most indent levels, not the smallest."
        )


# ---------------------------------------------------------------------------
# detect_column_limit
# ---------------------------------------------------------------------------


class TestDetectColumnLimit:
    def test_snap_to_standard_80(self, tmp_path: Path):
        """Lines near 80 should snap to the standard limit 80."""
        content = "x" * 78 + "\n" + "x" * 82 + "\n" + "x" * 79 + "\n"
        _ = _write_file(tmp_path, "a.c", content)
        result = detect_column_limit([str(tmp_path / "a.c")])
        assert result == 80, f"Expected snap to 80, got {result}"

    def test_no_files(self):
        assert detect_column_limit([]) is None

    def test_skips_very_long_lines(self, tmp_path: Path):
        content = "x" * 250 + "\n" + "x" * 100 + "\n"
        _ = _write_file(tmp_path, "a.c", content)
        result = detect_column_limit([str(tmp_path / "a.c")])
        assert result == 100

    def test_snap_to_120(self, tmp_path: Path):
        content = "\n".join(["x" * 118, "x" * 122, "x" * 119]) + "\n"
        _ = _write_file(tmp_path, "a.c", content)
        result = detect_column_limit([str(tmp_path / "a.c")])
        assert result == 120

    def test_no_snap_when_far(self, tmp_path: Path):
        """When detected value is far from any standard, return raw value."""
        content = "\n".join(["x" * 93 for _ in range(20)]) + "\n"
        _ = _write_file(tmp_path, "a.c", content)
        result = detect_column_limit([str(tmp_path / "a.c")])
        # 93 is >5 away from both 80 and 100
        assert result == 93

    def test_custom_percentile(self, tmp_path: Path):
        """Lower percentile picks shorter lines."""
        lines = ["x" * 50] * 10 + ["x" * 200]  # 200 is filtered out
        content = "\n".join(lines) + "\n"
        _ = _write_file(tmp_path, "a.c", content)
        result = detect_column_limit([str(tmp_path / "a.c")], percentile=0.50)
        assert result == 50


# ---------------------------------------------------------------------------
# detect_language
# ---------------------------------------------------------------------------


class TestDetectLanguage:
    def test_cpp_dominant(self, tmp_path: Path):
        _ = _write_file(tmp_path, "a.cpp", "x")
        _ = _write_file(tmp_path, "b.h", "y")
        _ = _write_file(tmp_path, "c.c", "z")
        assert (
            detect_language(
                [
                    str(tmp_path / "a.cpp"),
                    str(tmp_path / "b.h"),
                    str(tmp_path / "c.c"),
                ]
            )
            == "Cpp"
        )

    def test_no_files(self):
        assert detect_language([]) is None

    def test_pure_c(self, tmp_path: Path):
        _ = _write_file(tmp_path, "a.c", "x")
        _ = _write_file(tmp_path, "b.c", "y")
        assert (
            detect_language(
                [
                    str(tmp_path / "a.c"),
                    str(tmp_path / "b.c"),
                ]
            )
            == "C"
        )


# ---------------------------------------------------------------------------
# detect_qualifier_alignment
# ---------------------------------------------------------------------------


class TestDetectQualifierAlignment:
    def test_left_alignment(self, tmp_path: Path):
        content = "const int x = 0;\nconst char c = 'a';\n"
        _ = _write_file(tmp_path, "a.c", content)
        assert detect_qualifier_alignment([str(tmp_path / "a.c")]) == "Left"

    def test_right_alignment(self, tmp_path: Path):
        content = "int const x = 0;\nchar const c = 'a';\n"
        _ = _write_file(tmp_path, "a.c", content)
        assert detect_qualifier_alignment([str(tmp_path / "a.c")]) == "Right"

    def test_mixed_returns_leave(self, tmp_path: Path):
        """50/50 split should return Leave, not Left or Right."""
        content = "const int x = 0;\nint const y = 1;\n"
        _ = _write_file(tmp_path, "a.c", content)
        result = detect_qualifier_alignment([str(tmp_path / "a.c")])
        assert result == "Leave", (
            f"Expected 'Leave' for mixed qualifiers but got '{result}'. "
            "The 60% dominance threshold should not be met with a 50/50 split."
        )

    def test_no_qualifiers(self, tmp_path: Path):
        content = "int x = 0;\nchar c = 'a';\n"
        _ = _write_file(tmp_path, "a.c", content)
        assert detect_qualifier_alignment([str(tmp_path / "a.c")]) == "Leave"

    def test_skips_comments(self, tmp_path: Path):
        content = "// const int x = 0;\nint y = 1;\n"
        _ = _write_file(tmp_path, "a.c", content)
        assert detect_qualifier_alignment([str(tmp_path / "a.c")]) == "Leave"

    def test_regression_mixed_qualifiers_both_counted(self, tmp_path: Path):
        """Regression: 'const int' and 'int const' in the same file must both
        be counted. A bug with 'elif' instead of 'if' caused only one branch
        to fire per qualifier occurrence."""
        content = (
            "const int a = 0;\n"  # left
            "const int b = 1;\n"  # left
            "int const c = 2;\n"  # right
        )
        _ = _write_file(tmp_path, "a.c", content)
        # 2 left, 1 right -> 2/3 = 66.7% >= 60% -> Left
        result = detect_qualifier_alignment([str(tmp_path / "a.c")])
        assert result == "Left", (
            f"Expected 'Left' (2/3 dominance) but got '{result}'. "
            "Qualifier detection must count both left and right patterns."
        )


# ---------------------------------------------------------------------------
# detect_access_modifier_offset
# ---------------------------------------------------------------------------


class TestDetectAccessModifierOffset:
    def test_detects_offset(self, tmp_path: Path):
        content = "class Foo {\n  public:\n    void bar();\n};\n"
        _ = _write_file(tmp_path, "a.cpp", content)
        assert detect_access_modifier_offset([str(tmp_path / "a.cpp")]) == 2

    def test_no_access_modifiers(self, tmp_path: Path):
        content = "int x = 0;\n"
        _ = _write_file(tmp_path, "a.c", content)
        assert detect_access_modifier_offset([str(tmp_path / "a.c")]) is None

    def test_multiple_modifiers_picks_most_common(self, tmp_path: Path):
        content = "class Foo {\n  public:\n  private:\n    protected:\n};\n"
        _ = _write_file(tmp_path, "a.cpp", content)
        # indent 2 appears twice, indent 4 once
        assert detect_access_modifier_offset([str(tmp_path / "a.cpp")]) == 2


# ---------------------------------------------------------------------------
# detect_max_empty_lines
# ---------------------------------------------------------------------------


class TestDetectMaxEmptyLines:
    def test_consecutive_empty_lines(self, tmp_path: Path):
        content = "int x;\n\n\n\nint y;\n"
        _ = _write_file(tmp_path, "a.c", content)
        assert detect_max_empty_lines([str(tmp_path / "a.c")]) == 3

    def test_no_empty_lines(self, tmp_path: Path):
        content = "int x;\nint y;\n"
        _ = _write_file(tmp_path, "a.c", content)
        assert detect_max_empty_lines([str(tmp_path / "a.c")]) == 0

    def test_across_multiple_files(self, tmp_path: Path):
        _ = _write_file(tmp_path, "a.c", "int x;\n\nint y;\n")
        _ = _write_file(tmp_path, "b.c", "int a;\n\n\n\nint b;\n")
        files = [str(tmp_path / "a.c"), str(tmp_path / "b.c")]
        assert detect_max_empty_lines(files) == 3


# ---------------------------------------------------------------------------
# detect_numeric_literal_case
# ---------------------------------------------------------------------------


class TestDetectNumericLiteralCase:
    def test_uppercase_hex(self, tmp_path: Path):
        content = "int x = 0xFF;\nint y = 0xABCD;\n"
        _ = _write_file(tmp_path, "a.c", content)
        hex_digit, prefix, _, _ = detect_numeric_literal_case([str(tmp_path / "a.c")])
        assert hex_digit == "Upper"
        assert prefix == "Lower"

    def test_lowercase_hex(self, tmp_path: Path):
        content = "int x = 0xff;\nint y = 0xabcd;\n"
        _ = _write_file(tmp_path, "a.c", content)
        hex_digit, prefix, _, _ = detect_numeric_literal_case([str(tmp_path / "a.c")])
        assert hex_digit == "Lower"
        assert prefix == "Lower"

    def test_uppercase_prefix(self, tmp_path: Path):
        content = "int x = 0XFF;\nint y = 0XAB;\n"
        _ = _write_file(tmp_path, "a.c", content)
        _, prefix, _, _ = detect_numeric_literal_case([str(tmp_path / "a.c")])
        assert prefix == "Upper"

    def test_exponent_lowercase(self, tmp_path: Path):
        content = "float a = 1.5e10;\nfloat b = 3.0e-2;\n"
        _ = _write_file(tmp_path, "a.c", content)
        _, _, exponent, _ = detect_numeric_literal_case([str(tmp_path / "a.c")])
        assert exponent == "Lower", (
            f"Expected exponent 'Lower' but got '{exponent}'. "
            "The exponent regex must match 'e' in scientific notation like 1.5e10."
        )

    def test_exponent_uppercase(self, tmp_path: Path):
        content = "float a = 1.5E10;\nfloat b = 3.0E-2;\n"
        _ = _write_file(tmp_path, "a.c", content)
        _, _, exponent, _ = detect_numeric_literal_case([str(tmp_path / "a.c")])
        assert exponent == "Upper"

    def test_suffix_uppercase(self, tmp_path: Path):
        content = "unsigned long x = 10ULL;\nunsigned y = 20U;\n"
        _ = _write_file(tmp_path, "a.c", content)
        _, _, _, suffix = detect_numeric_literal_case([str(tmp_path / "a.c")])
        assert suffix == "Upper", (
            f"Expected suffix 'Upper' but got '{suffix}'. "
            "The suffix regex must match uppercase suffixes like U, ULL."
        )

    def test_suffix_lowercase(self, tmp_path: Path):
        content = "unsigned long x = 10ull;\nunsigned y = 20u;\n"
        _ = _write_file(tmp_path, "a.c", content)
        _, _, _, suffix = detect_numeric_literal_case([str(tmp_path / "a.c")])
        assert suffix == "Lower"

    def test_no_literals_returns_leave(self, tmp_path: Path):
        content = "int x = 0;\n"
        _ = _write_file(tmp_path, "a.c", content)
        hex_digit, prefix, exponent, suffix = detect_numeric_literal_case(
            [str(tmp_path / "a.c")]
        )
        assert hex_digit == "Leave"
        assert prefix == "Leave"
        assert exponent == "Leave"
        assert suffix == "Leave"

    def test_skips_string_literals(self, tmp_path: Path):
        content = 'char *s = "0xFF";\nint x = 0x11;\n'
        _ = _write_file(tmp_path, "a.c", content)
        _, prefix, _, _ = detect_numeric_literal_case([str(tmp_path / "a.c")])
        assert prefix == "Lower"

    def test_regression_exponent_pattern_matches_scientific(self, tmp_path: Path):
        """Regression: the exponent regex must match patterns like 1.5e10,
        3.0E-2, 1e5, etc. A buggy pattern [0-9](?:[.eE])([eE]) would never
        match because it requires a literal dot-or-e followed by another e."""
        content = "double d = 1e5;\n"
        _ = _write_file(tmp_path, "a.c", content)
        _, _, exponent, _ = detect_numeric_literal_case([str(tmp_path / "a.c")])
        assert exponent == "Lower", (
            f"Expected 'Lower' but got '{exponent}'. "
            "Exponent pattern must match simple scientific notation like 1e5."
        )

    def test_regression_suffix_matches_uppercase(self, tmp_path: Path):
        """Regression: suffix pattern must match uppercase U, L, ULL etc.
        A lowercase-only pattern (u|l|ul|...) misses real-world code."""
        content = "uint64_t x = 0xFFFFFFFFFFFFFFFFULL;\n"
        _ = _write_file(tmp_path, "a.c", content)
        _, _, _, suffix = detect_numeric_literal_case([str(tmp_path / "a.c")])
        assert suffix == "Upper", (
            f"Expected 'Upper' but got '{suffix}'. "
            "Suffix pattern must be case-insensitive to match ULL, UL, etc."
        )


# ---------------------------------------------------------------------------
# CLI (main)
# ---------------------------------------------------------------------------


class TestDetectNumericLiteralCaseMixed:
    """Tests for mixed/unclear case branches that return 'Leave'."""

    def test_mixed_hex_digits_returns_leave(self, tmp_path):
        """Equal mix of upper and lower hex digits returns Leave."""
        _ = _write_file(tmp_path, "a.cpp", "int x = 0xABCD;\nint y = 0xabcd;\n")
        result = detect_numeric_literal_case([str(tmp_path / "a.cpp")])
        assert result[0] == "Leave"  # hex_digit_case

    def test_mixed_prefix_returns_leave(self, tmp_path):
        """Equal mix of 0x and 0X returns Leave."""
        _ = _write_file(tmp_path, "a.cpp", "int x = 0x1;\nint y = 0X2;\n")
        result = detect_numeric_literal_case([str(tmp_path / "a.cpp")])
        assert result[1] == "Leave"  # prefix_case

    def test_mixed_exponent_returns_leave(self, tmp_path):
        """Equal mix of e and E returns Leave."""
        _ = _write_file(tmp_path, "a.cpp", "float x = 1e2;\nfloat y = 3E4;\n")
        result = detect_numeric_literal_case([str(tmp_path / "a.cpp")])
        assert result[2] == "Leave"  # exponent_case

    def test_mixed_suffix_returns_leave(self, tmp_path):
        """Equal mix of upper and lower suffixes returns Leave."""
        _ = _write_file(tmp_path, "a.cpp", "int x = 1u;\nint y = 2U;\n")
        result = detect_numeric_literal_case([str(tmp_path / "a.cpp")])
        assert result[3] == "Leave"  # suffix_case

    def test_mixed_suffix_case_counts(self, tmp_path):
        """Mixed case suffix like 'uL' counts as mixed, not upper or lower."""
        _ = _write_file(tmp_path, "a.cpp", "int x = 1uL;\n")
        result = detect_numeric_literal_case([str(tmp_path / "a.cpp")])
        assert result[3] == "Leave"  # suffix_case - mixed counts toward neither


# ---------------------------------------------------------------------------
# analyze()
# ---------------------------------------------------------------------------


class TestAnalyze:
    def test_returns_empty_for_no_files(self, tmp_path):
        from src.analyze_conventions import analyze

        result = analyze(str(tmp_path))
        assert result == {}

    def test_returns_detected_conventions(self, tmp_path):
        from src.analyze_conventions import analyze

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

    def test_excludes_leave_values(self, tmp_path):
        from src.analyze_conventions import analyze

        _ = _write_file(
            tmp_path,
            "a.cpp",
            "int x = 0xABCD;\nint y = 0xabcd;\n",
        )
        result = analyze(str(tmp_path))
        # Mixed hex digits → Leave, so key should not be present
        assert "NumericLiteralCase.HexDigit" not in result

    def test_excludes_none_access_offset(self, tmp_path):
        from src.analyze_conventions import analyze

        _ = _write_file(tmp_path, "a.cpp", "int x = 1;\n")
        result = analyze(str(tmp_path))
        # No access modifiers → None, so key should not be present
        assert "AccessModifierOffset" not in result

    def test_includes_access_offset_when_present(self, tmp_path):
        from src.analyze_conventions import analyze

        _ = _write_file(
            tmp_path,
            "a.cpp",
            "class Foo {\npublic:\n    int x;\n};\n",
        )
        result = analyze(str(tmp_path))
        assert "AccessModifierOffset" in result

    def test_returns_all_numeric_literal_fields(self, tmp_path):
        from src.analyze_conventions import analyze

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
