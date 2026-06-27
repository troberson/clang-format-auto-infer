"""Tests for detect_numeric_literal_case."""

from pathlib import Path

from src.analyze_conventions.literals import detect_numeric_literal_case


class TestDetectNumericLiteralCase:
    def test_uppercase_hex(self, tmp_path: Path):
        p = tmp_path / "a.c"
        _ = p.write_text("int x = 0xFF;\nint y = 0xABCD;\n")
        hex_digit, prefix, _, _ = detect_numeric_literal_case([str(p)])
        assert hex_digit == "Upper"
        assert prefix == "Lower"

    def test_lowercase_hex(self, tmp_path: Path):
        p = tmp_path / "a.c"
        _ = p.write_text("int x = 0xff;\nint y = 0xabcd;\n")
        hex_digit, prefix, _, _ = detect_numeric_literal_case([str(p)])
        assert hex_digit == "Lower"
        assert prefix == "Lower"

    def test_uppercase_prefix(self, tmp_path: Path):
        p = tmp_path / "a.c"
        _ = p.write_text("int x = 0XFF;\nint y = 0XAB;\n")
        _, prefix, _, _ = detect_numeric_literal_case([str(p)])
        assert prefix == "Upper"

    def test_exponent_lowercase(self, tmp_path: Path):
        p = tmp_path / "a.c"
        _ = p.write_text("float a = 1.5e10;\nfloat b = 3.0e-2;\n")
        _, _, exponent, _ = detect_numeric_literal_case([str(p)])
        assert exponent == "Lower", (
            f"Expected exponent 'Lower' but got '{exponent}'. "
            "The exponent regex must match 'e' in scientific notation like 1.5e10."
        )

    def test_exponent_uppercase(self, tmp_path: Path):
        p = tmp_path / "a.c"
        _ = p.write_text("float a = 1.5E10;\nfloat b = 3.0E-2;\n")
        _, _, exponent, _ = detect_numeric_literal_case([str(p)])
        assert exponent == "Upper"

    def test_suffix_uppercase(self, tmp_path: Path):
        p = tmp_path / "a.c"
        _ = p.write_text("unsigned long x = 10ULL;\nunsigned y = 20U;\n")
        _, _, _, suffix = detect_numeric_literal_case([str(p)])
        assert suffix == "Upper", (
            f"Expected suffix 'Upper' but got '{suffix}'. "
            "The suffix regex must match uppercase suffixes like U, ULL."
        )

    def test_suffix_lowercase(self, tmp_path: Path):
        p = tmp_path / "a.c"
        _ = p.write_text("unsigned long x = 10ull;\nunsigned y = 20u;\n")
        _, _, _, suffix = detect_numeric_literal_case([str(p)])
        assert suffix == "Lower"

    def test_no_literals_returns_leave(self, tmp_path: Path):
        p = tmp_path / "a.c"
        _ = p.write_text("int x = 0;\n")
        hex_digit, prefix, exponent, suffix = detect_numeric_literal_case([str(p)])
        assert hex_digit == "Leave"
        assert prefix == "Leave"
        assert exponent == "Leave"
        assert suffix == "Leave"

    def test_skips_string_literals(self, tmp_path: Path):
        p = tmp_path / "a.c"
        _ = p.write_text('char *s = "0xFF";\nint x = 0x11;\n')
        _, prefix, _, _ = detect_numeric_literal_case([str(p)])
        assert prefix == "Lower"

    def test_regression_exponent_pattern_matches_scientific(self, tmp_path: Path):
        """Regression: the exponent regex must match patterns like 1.5e10,
        3.0E-2, 1e5, etc."""
        p = tmp_path / "a.c"
        _ = p.write_text("double d = 1e5;\n")
        _, _, exponent, _ = detect_numeric_literal_case([str(p)])
        assert exponent == "Lower", (
            f"Expected 'Lower' but got '{exponent}'. "
            "Exponent pattern must match simple scientific notation like 1e5."
        )

    def test_regression_suffix_matches_uppercase(self, tmp_path: Path):
        """Regression: suffix pattern must match uppercase U, L, ULL etc."""
        p = tmp_path / "a.c"
        _ = p.write_text("uint64_t x = 0xFFFFFFFFFFFFFFFFULL;\n")
        _, _, _, suffix = detect_numeric_literal_case([str(p)])
        assert suffix == "Upper", (
            f"Expected 'Upper' but got '{suffix}'. "
            "Suffix pattern must be case-insensitive to match ULL, UL, etc."
        )


class TestDetectNumericLiteralCaseMixed:
    """Tests for mixed/unclear case branches that return 'Leave'."""

    def test_mixed_hex_digits_returns_leave(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("int x = 0xABCD;\nint y = 0xabcd;\n")
        result = detect_numeric_literal_case([str(p)])
        assert result[0] == "Leave"

    def test_mixed_prefix_returns_leave(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("int x = 0x1;\nint y = 0X2;\n")
        result = detect_numeric_literal_case([str(p)])
        assert result[1] == "Leave"

    def test_mixed_exponent_returns_leave(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("float x = 1e2;\nfloat y = 3E4;\n")
        result = detect_numeric_literal_case([str(p)])
        assert result[2] == "Leave"

    def test_mixed_suffix_returns_leave(self, tmp_path: Path):
        p = tmp_path / "a.cpp"
        _ = p.write_text("int x = 1u;\nint y = 2U;\n")
        result = detect_numeric_literal_case([str(p)])
        assert result[3] == "Leave"

    def test_mixed_suffix_case_counts(self, tmp_path: Path):
        """Mixed case suffix like 'uL' counts as mixed, not upper or lower."""
        p = tmp_path / "a.cpp"
        _ = p.write_text("int x = 1uL;\n")
        result = detect_numeric_literal_case([str(p)])
        assert result[3] == "Leave"
