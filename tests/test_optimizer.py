"""Tests for optimizer — crossover, mutate, optimize_option_with_values."""

import copy
from unittest.mock import patch, MagicMock
import pytest

from src.optimizer import crossover, mutate, optimize_option_with_values
from src.data_classes import IslandEvolutionArgs, GeneticAlgorithmLookups, WorkerContext


def _make_lookups(json_options=None, forced_options=None):
    return GeneticAlgorithmLookups(
        json_options_lookup=json_options or {},
        forced_options_lookup=forced_options or {},
    )


def _make_island_args(lookups, debug=False):
    return IslandEvolutionArgs(
        population=[],
        island_population_size=0,
        island_index=0,
        lookups=lookups,
        debug=debug,
        file_sample_percentage=100.0,
        random_seed=42,
    )


def _make_worker_context(repo_path="/tmp/fake"):
    return WorkerContext(repo_path=repo_path, process_id=1)


class TestCrossover:
    def test_produces_child_with_all_keys(self):
        parent1 = {
            "A": {"type": "str", "value": "x"},
            "B": {"type": "int", "value": 1},
        }
        parent2 = {
            "A": {"type": "str", "value": "y"},
            "B": {"type": "int", "value": 2},
        }
        child = crossover(parent1, parent2)
        assert set(child.keys()) == {"A", "B"}

    def test_values_come_from_parents(self):
        parent1 = {"A": {"type": "str", "value": "x"}}
        parent2 = {"A": {"type": "str", "value": "y"}}
        child = crossover(parent1, parent2)
        assert child["A"]["value"] in ("x", "y")

    def test_deep_copy_independence(self):
        parent1 = {"A": {"type": "str", "value": "x"}}
        parent2 = {"A": {"type": "str", "value": "y"}}
        child = crossover(parent1, parent2)
        child["A"]["value"] = "z"
        assert parent1["A"]["value"] == "x"
        assert parent2["A"]["value"] == "y"


class TestOptimizeOptionWithValues:
    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer.generate_clang_format_config")
    def test_selects_best_value(self, mock_gen, mock_fitness):
        mock_gen.return_value = "config"
        mock_fitness.side_effect = [10, 5, 8]  # values a, b, c

        flat = {
            "UseTab": {"type": "str", "value": "Never"},
        }
        lookups = _make_lookups({"UseTab": {"type": "str", "possible_values": ["Never", "Always", "ForIndentation"]}})
        args = _make_island_args(lookups)
        ctx = _make_worker_context()

        result = optimize_option_with_values(flat, "UseTab", ["Never", "Always", "ForIndentation"], args, ctx)

        assert result == 5
        assert flat["UseTab"]["value"] == "Always"

    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer.generate_clang_format_config")
    def test_reverts_on_all_failures(self, mock_gen, mock_fitness):
        mock_gen.return_value = "config"
        mock_fitness.side_effect = [-1, -1]  # all git errors

        flat = {
            "UseTab": {"type": "str", "value": "Never"},
        }
        lookups = _make_lookups({"UseTab": {"type": "str", "possible_values": ["Never", "Always"]}})
        args = _make_island_args(lookups)
        ctx = _make_worker_context()

        result = optimize_option_with_values(flat, "UseTab", ["Never", "Always"], args, ctx)

        assert result == float("inf")
        assert flat["UseTab"]["value"] == "Never"  # reverted

    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer.generate_clang_format_config")
    def test_skips_git_errors(self, mock_gen, mock_fitness):
        mock_gen.return_value = "config"
        mock_fitness.side_effect = [-1, 7]  # first is git error, second is valid

        flat = {
            "UseTab": {"type": "str", "value": "Never"},
        }
        lookups = _make_lookups({"UseTab": {"type": "str", "possible_values": ["Never", "Always"]}})
        args = _make_island_args(lookups)
        ctx = _make_worker_context()

        result = optimize_option_with_values(flat, "UseTab", ["Never", "Always"], args, ctx)

        assert result == 7
        assert flat["UseTab"]["value"] == "Always"

    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer.generate_clang_format_config")
    def test_treats_inf_as_high_cost(self, mock_gen, mock_fitness):
        mock_gen.return_value = "config"
        mock_fitness.side_effect = [float("inf"), 3]

        flat = {
            "UseTab": {"type": "str", "value": "Never"},
        }
        lookups = _make_lookups({"UseTab": {"type": "str", "possible_values": ["Never", "Always"]}})
        args = _make_island_args(lookups)
        ctx = _make_worker_context()

        result = optimize_option_with_values(flat, "UseTab", ["Never", "Always"], args, ctx)

        assert result == 3
        assert flat["UseTab"]["value"] == "Always"

    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer.generate_clang_format_config")
    def test_bool_type_conversion(self, mock_gen, mock_fitness):
        mock_gen.return_value = "config"
        mock_fitness.side_effect = [5, 3]

        flat = {
            "BreakBeforeBraces": {"type": "bool", "value": True},
        }
        lookups = _make_lookups()
        args = _make_island_args(lookups)
        ctx = _make_worker_context()

        result = optimize_option_with_values(flat, "BreakBeforeBraces", [True, False], args, ctx)

        assert result == 3
        assert flat["BreakBeforeBraces"]["value"] is False

    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer.generate_clang_format_config")
    def test_int_type_conversion(self, mock_gen, mock_fitness):
        mock_gen.return_value = "config"
        mock_fitness.side_effect = [10, 5]

        flat = {
            "ColumnLimit": {"type": "int", "value": 80},
        }
        lookups = _make_lookups({"ColumnLimit": {"type": "int", "possible_values": ["80", "100"]}})
        args = _make_island_args(lookups)
        ctx = _make_worker_context()

        result = optimize_option_with_values(flat, "ColumnLimit", ["80", "100"], args, ctx)

        assert result == 5
        assert flat["ColumnLimit"]["value"] == 100  # converted to int


class TestMutate:
    @patch("src.optimizer.optimize_option_with_values")
    def test_mutates_one_option(self, mock_opt):
        mock_opt.return_value = 5

        config = {
            "UseTab": {"type": "str", "value": "Never"},
            "ColumnLimit": {"type": "int", "value": 80},
        }
        lookups = _make_lookups(
            json_options={"UseTab": {"type": "str", "possible_values": ["Never", "Always"]}},
            forced_options={},
        )
        args = _make_island_args(lookups)
        ctx = _make_worker_context()

        result_config, fitness = mutate(config, args, ctx)

        assert fitness == 5
        assert result_config is not config  # deep copy
        mock_opt.assert_called_once()

    @patch("src.optimizer.optimize_option_with_values")
    def test_does_not_mutate_forced_options(self, mock_opt):
        mock_opt.return_value = 5

        config = {
            "UseTab": {"type": "str", "value": "Never"},
            "ColumnLimit": {"type": "int", "value": 80},
        }
        lookups = _make_lookups(
            json_options={"UseTab": {"type": "str", "possible_values": ["Never", "Always"]}},
            forced_options={"UseTab": "Always"},
        )
        args = _make_island_args(lookups)
        ctx = _make_worker_context()

        result_config, fitness = mutate(config, args, ctx)

        # UseTab is forced, so only ColumnLimit is mutable
        assert result_config["UseTab"]["value"] == "Always"  # forced

    def test_no_mutable_options_returns_inf(self):
        config = {
            "UseTab": {"type": "str", "value": "Never"},
        }
        lookups = _make_lookups(
            json_options={},  # no possible values
            forced_options={"UseTab": "Never"},  # forced
        )
        args = _make_island_args(lookups)
        ctx = _make_worker_context()

        result_config, fitness = mutate(config, args, ctx)

        assert fitness == float("inf")

    def test_bool_options_are_mutable_without_json(self):
        """Bool options should be mutable even if not in json_options_lookup."""
        config = {
            "BreakBeforeBraces": {"type": "bool", "value": True},
        }
        lookups = _make_lookups(json_options={}, forced_options={})
        args = _make_island_args(lookups)
        ctx = _make_worker_context()

        # Should not return inf — bool is mutable
        with patch("src.optimizer.optimize_option_with_values", return_value=3) as mock_opt:
            result_config, fitness = mutate(config, args, ctx)
            assert fitness == 3
            mock_opt.assert_called_once()
