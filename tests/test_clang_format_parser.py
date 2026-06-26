"""Tests for clang_format_parser — pure YAML roundtrip logic."""

from typing import Any
from unittest.mock import patch, MagicMock
import subprocess
from src.clang_format_parser import (
    IncrementalConfigBuilder,
    parse_clang_format_options,
    generate_clang_format_config,
    get_clang_format_options,
    save_checkpoint,
    load_checkpoint,
)


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
        assert result is not None
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
        assert result is not None
        assert result["PointerAlignment"]["value"] == "Left"
        assert result["SortIncludes.CaseSensitive"]["value"] is False
        assert result["SortIncludes.Priority"]["value"] == []

    def test_deeply_nested(self):
        yaml_str = "NumericLiteralCase:\n  HexDigit: Upper\n  Prefix: Lower\n"
        result = parse_clang_format_options(yaml_str)
        assert result is not None
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
        flat: dict[str, dict[str, str | Any]] = {
            "BasedOnStyle": {"type": "str", "value": "LLVM"},
            "ColumnLimit": {"type": "int", "value": 100},
        }
        result = generate_clang_format_config(flat)
        assert "BasedOnStyle: LLVM" in result
        assert "ColumnLimit: 100" in result

    def test_nested_options_reconstructed(self):
        flat: dict[str, dict[str, str | Any]] = {
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
        assert flat is not None
        regenerated = generate_clang_format_config(flat)
        reparsed = parse_clang_format_options(regenerated)
        assert flat == reparsed

    def test_empty_input(self):
        result = generate_clang_format_config({})
        assert result.strip() == "{}"


class TestGetClangFormatOptions:
    @patch("src.clang_format_parser.run_command")
    def test_success_returns_stdout(self, mock_run):
        mock_result = MagicMock()
        mock_result.stdout = "BasedOnStyle: LLVM\n"
        mock_run.return_value = mock_result
        result = get_clang_format_options()
        assert result == "BasedOnStyle: LLVM\n"
        mock_run.assert_called_once()

    @patch("src.clang_format_parser.run_command")
    def test_file_not_found_returns_none(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        result = get_clang_format_options()
        assert result is None

    @patch("src.clang_format_parser.run_command")
    def test_called_process_error_returns_none(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(
            1, ["clang-format"], stderr="error"
        )
        result = get_clang_format_options()
        assert result is None


class TestIncrementalConfigBuilder:
    def _make_flat(self) -> dict[str, dict[str, str | Any]]:
        return {
            "BasedOnStyle": {"type": "str", "value": "LLVM"},
            "ColumnLimit": {"type": "int", "value": 80},
            "UseTab": {"type": "str", "value": "Never"},
            "SortIncludes.CaseSensitive": {"type": "bool", "value": False},
            "SortIncludes.Priority": {"type": "list", "value": []},
        }

    def test_build_matches_generate_clang_format_config(self):
        flat = self._make_flat()
        builder = IncrementalConfigBuilder(flat)
        assert builder.build() == generate_clang_format_config(flat)

    def test_set_value_changes_yaml(self):
        flat = self._make_flat()
        builder = IncrementalConfigBuilder(flat)
        yaml_str = builder.set_value("ColumnLimit", 120)
        assert "ColumnLimit: 120" in yaml_str
        assert "ColumnLimit: 80" not in yaml_str

    def test_set_value_persists_across_calls(self):
        flat = self._make_flat()
        builder = IncrementalConfigBuilder(flat)
        _ = builder.set_value("ColumnLimit", 120)
        yaml_str = builder.build()
        assert "ColumnLimit: 120" in yaml_str

    def test_set_nested_value(self):
        flat = self._make_flat()
        builder = IncrementalConfigBuilder(flat)
        yaml_str = builder.set_value("SortIncludes.CaseSensitive", True)
        assert "CaseSensitive: true" in yaml_str

    def test_get_value_returns_current(self):
        flat = self._make_flat()
        builder = IncrementalConfigBuilder(flat)
        assert builder.get_value("ColumnLimit") == 80
        _ = builder.set_value("ColumnLimit", 120)
        assert builder.get_value("ColumnLimit") == 120

    def test_get_nested_value(self):
        flat = self._make_flat()
        builder = IncrementalConfigBuilder(flat)
        assert builder.get_value("SortIncludes.CaseSensitive") is False

    def test_roundtrip_via_builder(self):
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
        assert flat is not None
        builder = IncrementalConfigBuilder(flat)
        regenerated = builder.build()
        reparsed = parse_clang_format_options(regenerated)
        assert flat == reparsed

    def test_set_value_then_build_matches_generate(self):
        flat = self._make_flat()
        flat["ColumnLimit"]["value"] = 120
        builder = IncrementalConfigBuilder(self._make_flat())
        _ = builder.set_value("ColumnLimit", 120)
        assert builder.build() == generate_clang_format_config(flat)

    def test_empty_flat_options(self):
        builder = IncrementalConfigBuilder({})
        assert builder.build().strip() == "{}"

    def test_path_cache_populated_on_init(self):
        flat = self._make_flat()
        builder = IncrementalConfigBuilder(flat)
        assert "ColumnLimit" in builder._path_cache  # pyright: ignore[reportPrivateUsage]
        assert "SortIncludes.CaseSensitive" in builder._path_cache  # pyright: ignore[reportPrivateUsage]

    def test_set_value_fallback_for_unknown_path(self):
        """set_value() computes path on the fly when not in cache."""
        flat = self._make_flat()
        builder = IncrementalConfigBuilder(flat)
        # Use a path that wasn't in the original flat dict
        _ = builder.set_value("NewOption", "value")
        assert "NewOption" in builder._path_cache  # pyright: ignore[reportPrivateUsage]

    def test_get_value_fallback_for_unknown_path(self):
        """get_value() computes path on the fly when not in cache."""
        flat = self._make_flat()
        builder = IncrementalConfigBuilder(flat)
        # Remove a cached entry to force the fallback path
        del builder._path_cache["ColumnLimit"]  # pyright: ignore[reportPrivateUsage]
        # get_value should recompute the path and still return the correct value
        assert builder.get_value("ColumnLimit") == 80
        # Verify it was re-cached
        assert "ColumnLimit" in builder._path_cache  # pyright: ignore[reportPrivateUsage]


class TestCheckpoint:
    def test_save_and_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "checkpoint.json")
        best_config = {"IndentWidth": {"type": "int", "value": 4}}
        populations = [[{"config": best_config, "fitness": 50}]]
        history = [[50.0, 45.0]]

        save_checkpoint(path, best_config, 45, 1, populations, history)
        result = load_checkpoint(path)

        assert result is not None
        assert result["best_config"] == best_config
        assert result["best_fitness"] == 45
        assert result["iteration"] == 1
        assert result["populations"] == populations
        assert result["fitness_history_per_island"] == history

    def test_load_missing_file_returns_none(self, tmp_path):
        path = str(tmp_path / "nonexistent.json")
        assert load_checkpoint(path) is None

    def test_load_invalid_json_returns_none(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            _ = f.write("not json")
        assert load_checkpoint(path) is None
