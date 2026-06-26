"""Tests for config_loader — file I/O with temp files."""

import json
import pytest
from src.config_loader import load_json_option_values, load_forced_options


class TestLoadJsonOptionValues:
    def test_valid_json(self, tmp_path):
        data = [
            {
                "name": "AlignConsecutiveAssignments",
                "type": "bool",
                "possible_values": ["true", "false"],
            },
            {
                "name": "ColumnLimit",
                "type": "int",
                "possible_values": ["80", "100", "120"],
            },
        ]
        f = tmp_path / "values.json"
        f.write_text(json.dumps(data))
        result = load_json_option_values(str(f))
        assert "AlignConsecutiveAssignments" in result
        assert result["ColumnLimit"]["possible_values"] == ["80", "100", "120"]

    def test_none_path_returns_empty(self):
        result = load_json_option_values(None)
        assert result == {}

    def test_missing_file_exits(self, tmp_path, capsys):  # pyright: ignore[reportUnusedParameter]
        with pytest.raises(SystemExit):
            _ = load_json_option_values(str(tmp_path / "nonexistent.json"))

    def test_invalid_json_exits(self, tmp_path, capsys):  # pyright: ignore[reportUnusedParameter]
        f = tmp_path / "bad.json"
        f.write_text("{not valid json")
        with pytest.raises(SystemExit):
            _ = load_json_option_values(str(f))

    def test_non_list_json_exits(self, tmp_path, capsys):  # pyright: ignore[reportUnusedParameter]
        f = tmp_path / "dict.json"
        f.write_text('{"key": "value"}')
        with pytest.raises(SystemExit):
            _ = load_json_option_values(str(f))

    def test_skips_items_without_name(self, tmp_path):
        data = [
            {"name": "ValidOption", "type": "bool", "possible_values": ["true"]},
            {"no_name": "should be skipped"},
        ]
        f = tmp_path / "values.json"
        f.write_text(json.dumps(data))
        result = load_json_option_values(str(f))
        assert "ValidOption" in result
        assert len(result) == 1


class TestLoadForcedOptions:
    def test_valid_yaml_flat(self, tmp_path):
        content = "BasedOnStyle: LLVM\nColumnLimit: 100\n"
        f = tmp_path / "forced.yml"
        f.write_text(content)
        result = load_forced_options(str(f))
        assert result["BasedOnStyle"] == "LLVM"
        assert result["ColumnLimit"] == 100

    def test_valid_yaml_nested(self, tmp_path):
        content = "NumericLiteralCase:\n  HexDigit: Upper\n  Prefix: Lower\n"
        f = tmp_path / "forced.yml"
        f.write_text(content)
        result = load_forced_options(str(f))
        assert result["NumericLiteralCase.HexDigit"] == "Upper"
        assert result["NumericLiteralCase.Prefix"] == "Lower"

    def test_none_path_returns_empty(self):
        result = load_forced_options(None)
        assert result == {}

    def test_missing_file_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            _ = load_forced_options(str(tmp_path / "nonexistent.yml"))

    def test_invalid_yaml_exits(self, tmp_path):
        f = tmp_path / "bad.yml"
        f.write_text("::: invalid yaml :::")
        with pytest.raises(SystemExit):
            _ = load_forced_options(str(f))

    def test_non_dict_yaml_exits(self, tmp_path):
        f = tmp_path / "list.yml"
        f.write_text("- item1\n- item2\n")
        with pytest.raises(SystemExit):
            _ = load_forced_options(str(f))
