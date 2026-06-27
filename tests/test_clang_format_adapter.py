"""Tests for the clang-format adapter layer."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from src.clang_format_adapter import (
    build_search_space,
    config_to_flat_options,
    make_fitness_function,
)
from src.data_classes import GeneticAlgorithmLookups


def _make_lookups(
    json_options: dict[str, Any] | None = None,
    forced_options: dict[str, Any] | None = None,
) -> GeneticAlgorithmLookups:
    return GeneticAlgorithmLookups(
        json_options_lookup=json_options or {},
        forced_options_lookup=forced_options or {},
    )


def _make_base_options() -> dict[str, Any]:
    return {
        "IndentWidth": {"type": "int", "value": 4},
        "UseTab": {"type": "bool", "value": False},
        "Language": {"type": "str", "value": "Cpp"},
    }


class TestBuildSearchSpace:
    def test_includes_mutable_options(self):
        # Detected options are mutable, seeded with detected value.
        lookups = _make_lookups(
            json_options={"IndentWidth": {"possible_values": [2, 4]}}
        )
        analysis = {"IndentWidth": 4}
        ss = build_search_space(_make_base_options(), lookups, analysis)
        assert "IndentWidth" in ss.parameters
        assert ss.parameters["IndentWidth"].mutable
        assert ss.parameters["IndentWidth"].possible_values == [2, 4]

    def test_marks_forced_options_as_fixed(self):
        lookups = _make_lookups(forced_options={"UseTab": False})
        ss = build_search_space(_make_base_options(), lookups)
        assert ss.parameters["UseTab"].fixed is True
        assert ss.parameters["UseTab"].mutable is False

    def test_excludes_options_without_values(self):
        lookups = _make_lookups()
        ss = build_search_space(_make_base_options(), lookups)
        assert ss.parameters["IndentWidth"].mutable is False

    def test_preserves_param_type(self):
        lookups = _make_lookups()
        ss = build_search_space(_make_base_options(), lookups)
        assert ss.parameters["IndentWidth"].param_type == "int"
        assert ss.parameters["UseTab"].param_type == "bool"
        assert ss.parameters["Language"].param_type == "str"

    def test_analysis_results_seed_detected_options(self):
        # Detected options get all JSON possible_values for GA to explore.
        lookups = _make_lookups(
            json_options={"IndentWidth": {"possible_values": [2, 4, 8]}}
        )
        analysis = {"IndentWidth": 4}
        ss = build_search_space(_make_base_options(), lookups, analysis)
        assert ss.parameters["IndentWidth"].possible_values == [2, 4, 8]
        assert ss.parameters["IndentWidth"].mutable is True

    def test_undetected_options_are_fixed(self):
        # Options not detected by analyzer are fixed invariants.
        lookups = _make_lookups(
            json_options={"UseTab": {"possible_values": [True, False]}}
        )
        analysis = {"IndentWidth": 2}
        ss = build_search_space(_make_base_options(), lookups, analysis)
        assert ss.parameters["IndentWidth"].mutable is True
        assert ss.parameters["UseTab"].fixed is True
        assert ss.parameters["UseTab"].mutable is False

    def test_no_analysis_all_options_fixed(self):
        # Without analysis, all options are fixed invariants.
        lookups = _make_lookups(
            json_options={"IndentWidth": {"possible_values": [2, 4]}}
        )
        ss = build_search_space(_make_base_options(), lookups, None)
        assert ss.parameters["IndentWidth"].fixed is True
        assert ss.parameters["IndentWidth"].mutable is False

    def test_no_analysis_all_options_fixed_default(self):
        # Without analysis, all options are fixed invariants.
        lookups = _make_lookups(
            json_options={"IndentWidth": {"possible_values": [2, 4]}}
        )
        ss = build_search_space(_make_base_options(), lookups)
        assert ss.parameters["IndentWidth"].fixed is True
        assert ss.parameters["IndentWidth"].mutable is False

    def test_analysis_results_with_forced_options(self):
        lookups = _make_lookups(forced_options={"UseTab": True})
        analysis = {"IndentWidth": 4}
        ss = build_search_space(_make_base_options(), lookups, analysis)
        assert ss.parameters["UseTab"].fixed is True
        assert ss.parameters["IndentWidth"].mutable is True

    def test_analysis_results_unknown_key_ignored(self):
        lookups = _make_lookups()
        analysis = {"NonExistentOption": "value"}
        ss = build_search_space(_make_base_options(), lookups, analysis)
        assert "NonExistentOption" not in ss.parameters


class TestConfigToFlatOptions:
    def test_converts_int(self):
        base = _make_base_options()
        result = config_to_flat_options({"IndentWidth": 2}, base)
        assert result["IndentWidth"]["value"] == 2

    def test_converts_bool(self):
        base = _make_base_options()
        result = config_to_flat_options({"UseTab": True}, base)
        assert result["UseTab"]["value"] is True

    def test_keeps_unknown_keys_out(self):
        base = _make_base_options()
        result = config_to_flat_options({"UnknownOption": 1}, base)
        assert "UnknownOption" not in result

    def test_int_conversion_failure_keeps_original(self):
        base = _make_base_options()
        result = config_to_flat_options({"IndentWidth": "bad"}, base)
        assert result["IndentWidth"]["value"] == 4

    def test_converts_str(self):
        base = _make_base_options()
        result = config_to_flat_options({"Language": "C"}, base)
        assert result["Language"]["value"] == "C"

    def test_deep_copy_independence(self):
        base = _make_base_options()
        result = config_to_flat_options({"IndentWidth": 2}, base)
        result["IndentWidth"]["value"] = 99
        assert base["IndentWidth"]["value"] == 4


class TestMakeFitnessFunction:
    @patch("src.clang_format_adapter.run_clang_format_and_count_changes")
    @patch("src.clang_format_adapter.generate_clang_format_config")
    def test_returns_changes(self, mock_gen, mock_run):
        mock_gen.return_value = "config\n"
        mock_run.return_value = 42

        lookups = _make_lookups()
        base = _make_base_options()
        fitness_fn = make_fitness_function(
            repo_path="/tmp/repo",
            process_id=1,
            lookups=lookups,
            base_options=base,
        )
        result = fitness_fn({"IndentWidth": 2})
        assert result == 42

    @patch("src.clang_format_adapter.run_clang_format_and_count_changes")
    @patch("src.clang_format_adapter.generate_clang_format_config")
    def test_git_error_returns_inf(self, mock_gen, mock_run):
        mock_gen.return_value = "config\n"
        mock_run.return_value = -1

        lookups = _make_lookups()
        base = _make_base_options()
        fitness_fn = make_fitness_function(
            repo_path="/tmp/repo",
            process_id=1,
            lookups=lookups,
            base_options=base,
        )
        result = fitness_fn({"IndentWidth": 2})
        assert result == float("inf")

    @patch("src.clang_format_adapter.run_clang_format_and_count_changes")
    @patch("src.clang_format_adapter.generate_clang_format_config")
    def test_applies_forced_options(self, mock_gen, mock_run):
        mock_gen.return_value = "config\n"
        mock_run.return_value = 10

        lookups = _make_lookups(forced_options={"UseTab": True})
        base = _make_base_options()
        fitness_fn = make_fitness_function(
            repo_path="/tmp/repo",
            process_id=1,
            lookups=lookups,
            base_options=base,
        )
        _ = fitness_fn({"IndentWidth": 2})
        # Verify forced option was applied to the config
        call_args = mock_gen.call_args
        flat_opts = call_args[0][0]
        assert flat_opts["UseTab"]["value"] is True

    @patch("src.clang_format_adapter.run_clang_format_and_count_changes")
    @patch("src.clang_format_adapter.generate_clang_format_config")
    def test_converts_bool_type(self, mock_gen, mock_run):
        mock_gen.return_value = "config\n"
        mock_run.return_value = 5

        lookups = _make_lookups()
        base = _make_base_options()
        fitness_fn = make_fitness_function(
            repo_path="/tmp/repo",
            process_id=1,
            lookups=lookups,
            base_options=base,
        )
        _ = fitness_fn({"UseTab": 1})
        call_args = mock_gen.call_args
        flat_opts = call_args[0][0]
        assert flat_opts["UseTab"]["value"] is True

    @patch("src.clang_format_adapter.run_clang_format_and_count_changes")
    @patch("src.clang_format_adapter.generate_clang_format_config")
    def test_converts_str_type(self, mock_gen, mock_run):
        mock_gen.return_value = "config\n"
        mock_run.return_value = 5

        lookups = _make_lookups()
        base = _make_base_options()
        fitness_fn = make_fitness_function(
            repo_path="/tmp/repo",
            process_id=1,
            lookups=lookups,
            base_options=base,
        )
        _ = fitness_fn({"Language": "C"})
        call_args = mock_gen.call_args
        flat_opts = call_args[0][0]
        assert flat_opts["Language"]["value"] == "C"

    @patch("src.clang_format_adapter.run_clang_format_and_count_changes")
    @patch("src.clang_format_adapter.generate_clang_format_config")
    def test_debug_prints_conversion_failure(self, mock_gen, mock_run, capsys):
        mock_gen.return_value = "config\n"
        mock_run.return_value = 5

        lookups = _make_lookups()
        base = _make_base_options()
        fitness_fn = make_fitness_function(
            repo_path="/tmp/repo",
            process_id=1,
            lookups=lookups,
            base_options=base,
            debug=True,
        )
        _ = fitness_fn({"IndentWidth": "bad"})
        captured = capsys.readouterr()
        assert "Could not convert" in captured.err
