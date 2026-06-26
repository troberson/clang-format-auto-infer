"""Tests for optimizer — crossover, mutate, optimize_option_with_values."""

from typing import Any

from unittest.mock import patch

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

    def test_picks_from_parent1(self):
        p1 = {"UseTab": {"type": "str", "value": "Never"}}
        p2 = {"UseTab": {"type": "str", "value": "Always"}}
        with patch("src.optimizer.random.random", return_value=0.0):
            child = crossover(p1, p2)
        assert child["UseTab"]["value"] == "Never"

    def test_picks_from_parent2(self):
        p1 = {"UseTab": {"type": "str", "value": "Never"}}
        p2 = {"UseTab": {"type": "str", "value": "Always"}}
        with patch("src.optimizer.random.random", return_value=1.0):
            child = crossover(p1, p2)
        assert child["UseTab"]["value"] == "Always"


class TestOptimizeOptionWithValues:
    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer.generate_clang_format_config")
    def test_selects_best_value(self, mock_gen, mock_fitness):
        mock_gen.return_value = "config"
        mock_fitness.side_effect = [10, 5, 8]  # values a, b, c

        flat = {
            "UseTab": {"type": "str", "value": "Never"},
        }
        lookups = _make_lookups(
            {
                "UseTab": {
                    "type": "str",
                    "possible_values": ["Never", "Always", "ForIndentation"],
                }
            }
        )
        args = _make_island_args(lookups)
        ctx = _make_worker_context()

        result = optimize_option_with_values(
            flat, "UseTab", ["Never", "Always", "ForIndentation"], args, ctx
        )

        assert result == 5
        assert flat["UseTab"]["value"] == "Always"

    @patch("src.optimizer.run_clang_format_and_count_changes")
    def test_uses_builder_when_provided(self, mock_fitness):
        """When a builder is passed, builder.set_value() is used instead of generate_clang_format_config."""
        from src.clang_format_parser import IncrementalConfigBuilder

        mock_fitness.return_value = 10
        flat: dict[str, dict[str, str | Any]] = {
            "UseTab": {"type": "str", "value": "Never"},
            "ColumnLimit": {"type": "int", "value": 80},
        }
        lookups = _make_lookups(
            {"UseTab": {"type": "str", "possible_values": ["Never", "Always"]}}
        )
        args = _make_island_args(lookups)
        ctx = _make_worker_context()
        builder = IncrementalConfigBuilder(flat)

        with patch.object(builder, "set_value", wraps=builder.set_value) as mock_set:
            result = optimize_option_with_values(
                flat, "UseTab", ["Never", "Always"], args, ctx, builder
            )
            assert result == 10
            mock_set.assert_called()

    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer.generate_clang_format_config")
    def test_reverts_on_all_failures(self, mock_gen, mock_fitness):
        mock_gen.return_value = "config"
        mock_fitness.side_effect = [-1, -1]  # all git errors

        flat = {
            "UseTab": {"type": "str", "value": "Never"},
        }
        lookups = _make_lookups(
            {"UseTab": {"type": "str", "possible_values": ["Never", "Always"]}}
        )
        args = _make_island_args(lookups)
        ctx = _make_worker_context()

        result = optimize_option_with_values(
            flat, "UseTab", ["Never", "Always"], args, ctx
        )

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
        lookups = _make_lookups(
            {"UseTab": {"type": "str", "possible_values": ["Never", "Always"]}}
        )
        args = _make_island_args(lookups)
        ctx = _make_worker_context()

        result = optimize_option_with_values(
            flat, "UseTab", ["Never", "Always"], args, ctx
        )

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
        lookups = _make_lookups(
            {"UseTab": {"type": "str", "possible_values": ["Never", "Always"]}}
        )
        args = _make_island_args(lookups)
        ctx = _make_worker_context()

        result = optimize_option_with_values(
            flat, "UseTab", ["Never", "Always"], args, ctx
        )

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

        result = optimize_option_with_values(
            flat, "BreakBeforeBraces", [True, False], args, ctx
        )

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
        lookups = _make_lookups(
            {"ColumnLimit": {"type": "int", "possible_values": ["80", "100"]}}
        )
        args = _make_island_args(lookups)
        ctx = _make_worker_context()

        result = optimize_option_with_values(
            flat, "ColumnLimit", ["80", "100"], args, ctx
        )

        assert result == 5
        assert flat["ColumnLimit"]["value"] == 100  # converted to int

    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer.generate_clang_format_config")
    def test_debug_prints_testing(self, mock_gen, mock_run, capsys):
        mock_gen.return_value = "config\n"
        mock_run.return_value = 10
        lookups = _make_lookups(
            json_options={"IndentWidth": {"possible_values": ["2", "4"]}}
        )
        args = _make_island_args(lookups, debug=True)
        config = {
            "UseTab": {"type": "str", "value": "Never"},
            "IndentWidth": {"type": "int", "value": 4},
            "BreakBeforeBraces": {"type": "bool", "value": True},
        }
        _ = optimize_option_with_values(
            config, "IndentWidth", ["2", "4"], args, _make_worker_context()
        )
        captured = capsys.readouterr()
        assert "Testing values" in captured.err

    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer.generate_clang_format_config")
    def test_debug_prints_per_value(self, mock_gen, mock_run, capsys):
        mock_gen.return_value = "config\n"
        mock_run.return_value = 10
        lookups = _make_lookups(
            json_options={"IndentWidth": {"possible_values": ["2", "4"]}}
        )
        args = _make_island_args(lookups, debug=True)
        config = {
            "UseTab": {"type": "str", "value": "Never"},
            "IndentWidth": {"type": "int", "value": 4},
            "BreakBeforeBraces": {"type": "bool", "value": True},
        }
        _ = optimize_option_with_values(
            config, "IndentWidth", ["2", "4"], args, _make_worker_context()
        )
        captured = capsys.readouterr()
        assert "Testing 'IndentWidth'" in captured.err

    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer.generate_clang_format_config")
    def test_debug_prints_best_value(self, mock_gen, mock_run, capsys):
        mock_gen.return_value = "config\n"
        mock_run.return_value = 10
        lookups = _make_lookups(
            json_options={"IndentWidth": {"possible_values": ["2", "4"]}}
        )
        args = _make_island_args(lookups, debug=True)
        config = {
            "UseTab": {"type": "str", "value": "Never"},
            "IndentWidth": {"type": "int", "value": 4},
            "BreakBeforeBraces": {"type": "bool", "value": True},
        }
        _ = optimize_option_with_values(
            config, "IndentWidth", ["2", "4"], args, _make_worker_context()
        )
        captured = capsys.readouterr()
        assert "Best value for" in captured.err

    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer.generate_clang_format_config")
    def test_bool_string_conversion_true(self, mock_gen, mock_run):
        mock_gen.return_value = "config\n"
        mock_run.return_value = 10
        args = _make_island_args(_make_lookups())
        config: dict[str, dict[str, Any]] = {
            "UseTab": {"type": "str", "value": "Never"},
            "BreakBeforeBraces": {"type": "bool", "value": True},
        }
        _ = optimize_option_with_values(
            config,
            "BreakBeforeBraces",
            ["true"],
            args,
            _make_worker_context(),
        )
        assert config["BreakBeforeBraces"]["value"] is True

    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer.generate_clang_format_config")
    def test_bool_string_conversion_false(self, mock_gen, mock_run):
        mock_gen.return_value = "config\n"
        mock_run.return_value = 10
        args = _make_island_args(_make_lookups())
        config: dict[str, dict[str, Any]] = {
            "UseTab": {"type": "str", "value": "Never"},
            "BreakBeforeBraces": {"type": "bool", "value": True},
        }
        _ = optimize_option_with_values(
            config,
            "BreakBeforeBraces",
            ["false"],
            args,
            _make_worker_context(),
        )
        assert config["BreakBeforeBraces"]["value"] is False

    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer.generate_clang_format_config")
    def test_bool_capitalized_string_conversion(self, mock_gen, mock_run):
        mock_gen.return_value = "config\n"
        mock_run.return_value = 10
        args = _make_island_args(_make_lookups())
        config: dict[str, dict[str, Any]] = {
            "UseTab": {"type": "str", "value": "Never"},
            "BreakBeforeBraces": {"type": "bool", "value": True},
        }
        _ = optimize_option_with_values(
            config,
            "BreakBeforeBraces",
            ["True", "False"],
            args,
            _make_worker_context(),
        )
        assert isinstance(config["BreakBeforeBraces"]["value"], bool)

    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer.generate_clang_format_config")
    def test_string_true_converted_to_bool(self, mock_gen, mock_run):
        mock_gen.return_value = "config\n"
        mock_run.return_value = 10
        args = _make_island_args(_make_lookups())
        config: dict[str, dict[str, Any]] = {
            "UseTab": {"type": "str", "value": "Never"},
            "BreakBeforeBraces": {"type": "bool", "value": True},
        }
        _ = optimize_option_with_values(
            config,
            "BreakBeforeBraces",
            ["true"],
            args,
            _make_worker_context(),
        )
        assert config["BreakBeforeBraces"]["value"] is True

    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer.generate_clang_format_config")
    def test_int_conversion_failure_skips(self, mock_gen, mock_run, capsys):
        mock_gen.return_value = "config\n"
        mock_run.return_value = 10
        lookups = _make_lookups(
            json_options={"IndentWidth": {"possible_values": ["not_a_number"]}}
        )
        args = _make_island_args(lookups, debug=True)
        config = {
            "UseTab": {"type": "str", "value": "Never"},
            "IndentWidth": {"type": "int", "value": 4},
            "BreakBeforeBraces": {"type": "bool", "value": True},
        }
        _ = optimize_option_with_values(
            config, "IndentWidth", ["not_a_number"], args, _make_worker_context()
        )
        captured = capsys.readouterr()
        assert "Could not convert" in captured.err


class TestMutate:
    @patch("src.optimizer.optimize_option_with_values")
    def test_mutates_one_option(self, mock_opt):
        mock_opt.return_value = 5

        config = {
            "UseTab": {"type": "str", "value": "Never"},
            "ColumnLimit": {"type": "int", "value": 80},
        }
        lookups = _make_lookups(
            json_options={
                "UseTab": {"type": "str", "possible_values": ["Never", "Always"]}
            },
            forced_options={},
        )
        args = _make_island_args(lookups)
        ctx = _make_worker_context()

        result_config, fitness = mutate(config, args, ctx)

        assert fitness == 5
        # mutate() modifies config in-place and returns the same object
        assert result_config is config
        mock_opt.assert_called_once()

    @patch("src.optimizer.optimize_option_with_values")
    def test_does_not_mutate_forced_options(self, mock_opt):
        mock_opt.return_value = 5

        config = {
            "UseTab": {"type": "str", "value": "Never"},
            "ColumnLimit": {"type": "int", "value": 80},
        }
        lookups = _make_lookups(
            json_options={
                "UseTab": {"type": "str", "possible_values": ["Never", "Always"]}
            },
            forced_options={"UseTab": "Always"},
        )
        args = _make_island_args(lookups)
        ctx = _make_worker_context()

        result_config, _fitness = mutate(config, args, ctx)

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

        _result_config, fitness = mutate(config, args, ctx)

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
        with patch(
            "src.optimizer.optimize_option_with_values", return_value=3
        ) as mock_opt:
            _result_config, fitness = mutate(config, args, ctx)
            assert fitness == 3
            mock_opt.assert_called_once()

    @patch("src.optimizer.optimize_option_with_values")
    def test_debug_prints_no_mutable_options(self, mock_opt, capsys):
        island_args = _make_island_args(_make_lookups(), debug=True)
        config = {
            "UseTab": {"type": "str", "value": "Never"},
            "IndentWidth": {"type": "int", "value": 4},
        }
        _ = mutate(config, island_args, _make_worker_context())
        captured = capsys.readouterr()
        assert "No mutable options" in captured.err
        mock_opt.assert_not_called()

    @patch("src.optimizer.optimize_option_with_values")
    def test_debug_prints_mutating(self, mock_opt, capsys):
        mock_opt.return_value = 10
        lookups = _make_lookups(
            json_options={"IndentWidth": {"possible_values": ["2", "4"]}}
        )
        args = _make_island_args(lookups, debug=True)
        config = {
            "UseTab": {"type": "str", "value": "Never"},
            "IndentWidth": {"type": "int", "value": 4},
            "BreakBeforeBraces": {"type": "bool", "value": True},
        }
        _ = mutate(config, args, _make_worker_context())
        captured = capsys.readouterr()
        assert "Mutating" in captured.err
