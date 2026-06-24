"""Tests for clang_format_parser — pure YAML roundtrip logic."""

import pytest
from src.clang_format_parser import parse_clang_format_options, generate_clang_format_config


class TestParseClangFormatOptions:
    def test_simple_scalar_options(self):
        yaml_str = "BasedOnStyle: LLVM\nColumnLimit: 100\nUseTab: Never\n"
        result = parse_clang_format_options(yaml_str)
        assert result is not None
        assert result["BasedOnStyle"] == {"type": "str", "value": "LLVM"}
        assert result["ColumnLimit"] == {"type": "int", "value": 100}
        assert result["UseTab"] == {"type": "str", "value": "Never"}

    def test_boolean_option(self):
        yaml_str = "BreakBeforeBraces: true\n"
        result = parse_clang_format_options(yaml_str)
        assert result["BreakBeforeBraces"]["type"] == "bool"
        assert result["BreakBeforeBraces"]["value"] is True

    def test_nested_dict_flattened(self):
        yaml_str = (
            "PointerAlignment: Left\n"
            "SortIncludes:\n"
            "  CaseSensitive: false\n"
            "  Priority: []\n"
        )
        result = parse_clang_format_options(yaml_str)
        assert result["PointerAlignment"]["value"] == "Left"
        assert result["SortIncludes.CaseSensitive"]["value"] is False
        assert result["SortIncludes.Priority"]["value"] == []

    def test_deeply_nested(self):
        yaml_str = (
            "NumericLiteralCase:\n"
            "  HexDigit: Upper\n"
            "  Prefix: Lower\n"
        )
        result = parse_clang_format_options(yaml_str)
        assert result["NumericLiteralCase.HexDigit"]["value"] == "Upper"
        assert result["NumericLiteralCase.Prefix"]["value"] == "Lower"

    def test_empty_yaml_returns_empty_dict(self):
        result = parse_clang_format_options("{}")
        assert result == {}

    def test_non_dict_yaml_returns_none(self):
        result = parse_clang_format_options("just a string")
        assert result is None

    def test_invalid_yaml_returns_none(self):
        result = parse_clang_format_options("::: invalid yaml :::")
        assert result is None


class TestGenerateClangFormatConfig:
    def test_simple_options(self):
        flat = {
            "BasedOnStyle": {"type": "str", "value": "LLVM"},
            "ColumnLimit": {"type": "int", "value": 100},
        }
        result = generate_clang_format_config(flat)
        assert "BasedOnStyle: LLVM" in result
        assert "ColumnLimit: 100" in result

    def test_nested_options_reconstructed(self):
        flat = {
            "SortIncludes.CaseSensitive": {"type": "bool", "value": False},
            "SortIncludes.Priority": {"type": "list", "value": []},
        }
        result = generate_clang_format_config(flat)
        assert "SortIncludes:" in result
        assert "CaseSensitive: false" in result

    def test_roundtrip(self):
        original_yaml = (
            "BasedOnStyle: LLVM\n"
            "ColumnLimit: 120\n"
            "UseTab: Never\n"
            "SortIncludes:\n"
            "  CaseSensitive: true\n"
            "  Priority:\n"
            "    - system\n"
            "    - literal\n"
        )
        flat = parse_clang_format_options(original_yaml)
        regenerated = generate_clang_format_config(flat)
        reparsed = parse_clang_format_options(regenerated)
        assert flat == reparsed

    def test_empty_input(self):
        result = generate_clang_format_config({})
        assert result.strip() == "{}"
