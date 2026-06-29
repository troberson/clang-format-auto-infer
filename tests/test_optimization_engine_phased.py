"""Tests for the phased hybrid optimization module."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

from src.optimization_engine.phased import (
    _build_subspace,  # pyright: ignore[reportPrivateUsage]
    _run_phase,  # pyright: ignore[reportPrivateUsage]
    _split_by_type,  # pyright: ignore[reportPrivateUsage]
    PHASE_TIERS,
    run_phased_optimization,
)
from src.optimization_engine.types import (
    OptimizationResult,
    ParameterDef,
    SearchSpace,
)


def _make_fitness(
    target: dict[str, Any],
) -> Callable[[dict[str, Any]], float]:
    """Return a fitness function that returns 0 when config matches target."""

    def fitness(config: dict[str, Any]) -> float:
        score = 0
        for key, value in target.items():
            if key in config and config[key] != value:
                score += 1
        return float(score)

    return fitness


def _build_mock_ng_optimizer():
    """Build a mock nevergrad optimizer for testing."""
    mock_optimizer = MagicMock()
    mock_param = MagicMock()
    mock_param.kwargs = {}
    mock_optimizer.ask.return_value = mock_param

    mock_rec = MagicMock()
    mock_rec.kwargs = {}
    mock_optimizer.provide_recommendation.return_value = mock_rec
    return mock_optimizer


def _build_mock_executor_and_future(
    _objective: Callable[[dict[str, Any]], float],
):
    """Build a mock executor and a shared future so wait() can return it."""
    future = MagicMock()
    future.done.return_value = True
    future.cancel.return_value = True

    mock_executor = MagicMock()

    def submit_func(func, **kwargs):
        try:
            result = func(**kwargs)
        except Exception as e:
            future.result.side_effect = e
        else:
            future.result.return_value = result
        return future

    mock_executor.submit = submit_func
    return mock_executor, future


class TestSplitByType:
    def test_splits_integers_and_categoricals(self):
        params = [
            ParameterDef(name="IndentWidth", param_type="int", possible_values=[2, 4]),
            ParameterDef(
                name="BreakBeforeBraces",
                param_type="str",
                possible_values=["Attach", "Allman"],
            ),
            ParameterDef(name="TabWidth", param_type="int", possible_values=[4, 8]),
            ParameterDef(
                name="SortIncludes",
                param_type="bool",
                possible_values=[True, False],
            ),
        ]
        ints, cats = _split_by_type(params)
        assert [p.name for p in ints] == ["IndentWidth", "TabWidth"]
        assert [p.name for p in cats] == ["BreakBeforeBraces", "SortIncludes"]

    def test_all_integers(self):
        params = [
            ParameterDef(name="a", param_type="int", possible_values=[1]),
            ParameterDef(name="b", param_type="int", possible_values=[2]),
        ]
        ints, cats = _split_by_type(params)
        assert len(ints) == 2
        assert cats == []

    def test_all_categoricals(self):
        params = [
            ParameterDef(name="a", param_type="str", possible_values=["x"]),
            ParameterDef(name="b", param_type="bool", possible_values=[True]),
        ]
        ints, cats = _split_by_type(params)
        assert ints == []
        assert len(cats) == 2

    def test_empty(self):
        ints, cats = _split_by_type([])
        assert ints == []
        assert cats == []


class TestBuildSubspace:
    def test_includes_only_given_params(self):
        space = SearchSpace(
            parameters={
                "a": ParameterDef(name="a", param_type="int", possible_values=[1, 2]),
                "b": ParameterDef(name="b", param_type="str", possible_values=["x"]),
                "c": ParameterDef(name="c", param_type="int", possible_values=[3]),
            }
        )
        subspace = _build_subspace(space, [space.parameters["a"]])
        assert subspace.parameters["a"].mutable is True
        assert subspace.parameters["b"].fixed is True
        assert subspace.parameters["c"].fixed is True

    def test_empty_param_list_freezes_all(self):
        space = SearchSpace(
            parameters={
                "a": ParameterDef(name="a", param_type="int", possible_values=[1]),
            }
        )
        subspace = _build_subspace(space, [])
        assert subspace.mutable_parameters == []


class TestRunPhase:
    def test_skips_when_no_mutable_params_in_tier(self):
        space = SearchSpace(
            parameters={
                "a": ParameterDef(
                    name="a",
                    param_type="int",
                    possible_values=[1, 2],
                    tier="polish",
                ),
            }
        )
        fitness = _make_fitness({"a": 1})
        result = _run_phase(
            phase_name="resolve",
            tier="resolve",
            search_space=space,
            fitness_fn=fitness,
            initial_config={"a": 1},
            num_islands=1,
            population_size=4,
            num_workers=1,
            debug=False,
        )
        assert result.best_config == {"a": 1}

    def test_skips_when_no_mutable_params_debug_print(self, capsys):
        space = SearchSpace(
            parameters={
                "a": ParameterDef(
                    name="a",
                    param_type="int",
                    possible_values=[1, 2],
                    tier="polish",
                ),
            }
        )
        fitness = _make_fitness({"a": 1})
        _ = _run_phase(
            phase_name="resolve",
            tier="resolve",
            search_space=space,
            fitness_fn=fitness,
            initial_config={"a": 1},
            num_islands=1,
            population_size=4,
            num_workers=1,
            debug=True,
        )
        captured = capsys.readouterr()
        assert "no mutable parameters" in captured.err

    def test_runs_ga_for_integer_params(self):
        space = SearchSpace(
            parameters={
                "IndentWidth": ParameterDef(
                    name="IndentWidth",
                    param_type="int",
                    possible_values=[2, 4, 8],
                    tier="structure",
                ),
            }
        )
        target = {"IndentWidth": 4}
        fitness = _make_fitness(target)
        result = _run_phase(
            phase_name="structure",
            tier="structure",
            search_space=space,
            fitness_fn=fitness,
            initial_config={"IndentWidth": 2},
            num_islands=1,
            population_size=4,
            num_workers=1,
            debug=False,
        )
        assert result.best_fitness == 0.0
        assert result.best_config["IndentWidth"] == 4

    @patch("src.optimization_engine.nevergrad.ng.optimizers.registry")
    @patch("src.optimization_engine.nevergrad.concurrent.futures.ThreadPoolExecutor")
    @patch("src.optimization_engine.nevergrad.concurrent.futures.wait")
    def test_runs_nevergrad_for_categorical_params(
        self, mock_wait, mock_pool_cls, mock_registry
    ):
        mock_optimizer = _build_mock_ng_optimizer()
        mock_registry.__getitem__.return_value = MagicMock(return_value=mock_optimizer)

        mock_executor, mock_future = _build_mock_executor_and_future(
            _make_fitness({"BreakBeforeBraces": "Allman"})
        )
        mock_pool_cls.return_value = mock_executor
        mock_wait.return_value = ([mock_future], [])

        space = SearchSpace(
            parameters={
                "BreakBeforeBraces": ParameterDef(
                    name="BreakBeforeBraces",
                    param_type="str",
                    possible_values=["Attach", "Allman", "K&R"],
                    tier="structure",
                ),
            }
        )
        target = {"BreakBeforeBraces": "Allman"}
        fitness = _make_fitness(target)
        result = _run_phase(
            phase_name="structure",
            tier="structure",
            search_space=space,
            fitness_fn=fitness,
            initial_config={"BreakBeforeBraces": "Attach"},
            num_islands=1,
            population_size=4,
            num_workers=1,
            debug=False,
        )
        # nevergrad mock returns initial config; fitness may not be 0
        # but it should not crash and should return a result
        assert isinstance(result, OptimizationResult)

    @patch("src.optimization_engine.nevergrad.ng.optimizers.registry")
    @patch("src.optimization_engine.nevergrad.concurrent.futures.ThreadPoolExecutor")
    @patch("src.optimization_engine.nevergrad.concurrent.futures.wait")
    def test_nevergrad_debug_print_and_improvement(
        self, mock_wait, mock_pool_cls, mock_registry, capsys
    ):
        """Test nevergrad debug print and the improvement path."""
        mock_optimizer = _build_mock_ng_optimizer()
        mock_registry.__getitem__.return_value = MagicMock(return_value=mock_optimizer)

        # Mock the recommendation to return the optimal value
        mock_rec = MagicMock()
        mock_rec.kwargs = {"BreakBeforeBraces": "Allman"}
        mock_optimizer.provide_recommendation.return_value = mock_rec

        target = {"BreakBeforeBraces": "Allman"}
        fitness = _make_fitness(target)

        # Mock executor that returns fitness=0 (optimal) for all evaluations
        future = MagicMock()
        future.done.return_value = True
        future.cancel.return_value = True
        future.result.return_value = 0.0

        mock_executor = MagicMock()
        mock_executor.submit.return_value = future
        mock_pool_cls.return_value = mock_executor
        mock_wait.return_value = ([future], [])

        space = SearchSpace(
            parameters={
                "BreakBeforeBraces": ParameterDef(
                    name="BreakBeforeBraces",
                    param_type="str",
                    possible_values=["Attach", "Allman"],
                    tier="structure",
                ),
            }
        )
        result = _run_phase(
            phase_name="structure",
            tier="structure",
            search_space=space,
            fitness_fn=fitness,
            initial_config={"BreakBeforeBraces": "Attach"},
            num_islands=1,
            population_size=4,
            num_workers=1,
            debug=True,
        )
        captured = capsys.readouterr()
        assert "Running nevergrad" in captured.err
        # nevergrad found fitness=0, which is better than initial fitness=1
        assert result.best_fitness == 0.0
        assert "nevergrad improved fitness" in captured.err

    @patch("src.optimization_engine.nevergrad.ng.optimizers.registry")
    @patch("src.optimization_engine.nevergrad.concurrent.futures.ThreadPoolExecutor")
    @patch("src.optimization_engine.nevergrad.concurrent.futures.wait")
    def test_runs_both_ga_and_nevergrad(self, mock_wait, mock_pool_cls, mock_registry):
        mock_optimizer = _build_mock_ng_optimizer()
        mock_registry.__getitem__.return_value = MagicMock(return_value=mock_optimizer)

        mock_executor, mock_future = _build_mock_executor_and_future(
            _make_fitness({"IndentWidth": 4, "BreakBeforeBraces": "Allman"})
        )
        mock_pool_cls.return_value = mock_executor
        mock_wait.return_value = ([mock_future], [])

        space = SearchSpace(
            parameters={
                "IndentWidth": ParameterDef(
                    name="IndentWidth",
                    param_type="int",
                    possible_values=[2, 4, 8],
                    tier="structure",
                ),
                "BreakBeforeBraces": ParameterDef(
                    name="BreakBeforeBraces",
                    param_type="str",
                    possible_values=["Attach", "Allman"],
                    tier="structure",
                ),
            }
        )
        target = {"IndentWidth": 4, "BreakBeforeBraces": "Allman"}
        fitness = _make_fitness(target)
        result = _run_phase(
            phase_name="structure",
            tier="structure",
            search_space=space,
            fitness_fn=fitness,
            initial_config={"IndentWidth": 2, "BreakBeforeBraces": "Attach"},
            num_islands=1,
            population_size=4,
            num_workers=1,
            debug=False,
        )
        # GA should find the integer optimum; nevergrad mock returns initial
        assert result.best_fitness <= 1.0
        assert result.best_config["IndentWidth"] == 4


class TestRunPhasedOptimization:
    def test_phase_order_is_correct(self):
        assert PHASE_TIERS == ["resolve", "structure", "polish"]

    def test_skips_empty_tiers(self):
        space = SearchSpace(
            parameters={
                "a": ParameterDef(
                    name="a",
                    param_type="int",
                    possible_values=[1, 2],
                    tier="polish",
                ),
            }
        )
        fitness = _make_fitness({"a": 2})
        result = run_phased_optimization(
            search_space=space,
            fitness_fn=fitness,
            initial_config={"a": 1},
            num_islands=1,
            population_size=4,
            num_workers=1,
            debug=False,
        )
        # Should skip resolve and structure, only run polish
        assert result.best_fitness == 0.0
        assert result.best_config["a"] == 2

    def test_convergence_skips_remaining_phases(self):
        """If resolve finds the optimal config, structure and polish should be skipped."""
        space = SearchSpace(
            parameters={
                "a": ParameterDef(
                    name="a",
                    param_type="int",
                    possible_values=[1, 2],
                    tier="resolve",
                ),
                "b": ParameterDef(
                    name="b",
                    param_type="str",
                    possible_values=["x", "y"],
                    tier="structure",
                ),
            }
        )
        # Target is already at initial config for 'a', so resolve finds fitness=0
        fitness = _make_fitness({"a": 1, "b": "x"})
        result = run_phased_optimization(
            search_space=space,
            fitness_fn=fitness,
            initial_config={"a": 1, "b": "x"},
            num_islands=1,
            population_size=4,
            num_workers=1,
            debug=False,
        )
        assert result.best_fitness == 0.0

    def test_convergence_skips_remaining_phases_debug(self, capsys):
        """Debug mode should print convergence skip message."""
        space = SearchSpace(
            parameters={
                "a": ParameterDef(
                    name="a",
                    param_type="int",
                    possible_values=[1, 2],
                    tier="resolve",
                ),
                "b": ParameterDef(
                    name="b",
                    param_type="str",
                    possible_values=["x", "y"],
                    tier="structure",
                ),
            }
        )
        fitness = _make_fitness({"a": 1, "b": "x"})
        _ = run_phased_optimization(
            search_space=space,
            fitness_fn=fitness,
            initial_config={"a": 1, "b": "x"},
            num_islands=1,
            population_size=4,
            num_workers=1,
            debug=True,
        )
        captured = capsys.readouterr()
        assert "no improvement" in captured.err

    @patch("src.optimization_engine.nevergrad.ng.optimizers.registry")
    @patch("src.optimization_engine.nevergrad.concurrent.futures.ThreadPoolExecutor")
    @patch("src.optimization_engine.nevergrad.concurrent.futures.wait")
    def test_multi_tier_optimization(self, mock_wait, mock_pool_cls, mock_registry):
        """Test that phased optimization works across tiers."""
        mock_optimizer = _build_mock_ng_optimizer()
        mock_registry.__getitem__.return_value = MagicMock(return_value=mock_optimizer)

        mock_executor, mock_future = _build_mock_executor_and_future(
            _make_fitness(
                {
                    "IndentWidth": 4,
                    "BreakBeforeBraces": "Allman",
                    "SortIncludes": True,
                }
            )
        )
        mock_pool_cls.return_value = mock_executor
        mock_wait.return_value = ([mock_future], [])

        space = SearchSpace(
            parameters={
                "IndentWidth": ParameterDef(
                    name="IndentWidth",
                    param_type="int",
                    possible_values=[2, 4, 8],
                    tier="resolve",
                ),
                "BreakBeforeBraces": ParameterDef(
                    name="BreakBeforeBraces",
                    param_type="str",
                    possible_values=["Attach", "Allman"],
                    tier="structure",
                ),
                "SortIncludes": ParameterDef(
                    name="SortIncludes",
                    param_type="bool",
                    possible_values=[True, False],
                    tier="polish",
                ),
            }
        )
        target = {
            "IndentWidth": 4,
            "BreakBeforeBraces": "Allman",
            "SortIncludes": True,
        }
        fitness = _make_fitness(target)
        result = run_phased_optimization(
            search_space=space,
            fitness_fn=fitness,
            initial_config={
                "IndentWidth": 2,
                "BreakBeforeBraces": "Attach",
                "SortIncludes": False,
            },
            num_islands=1,
            population_size=4,
            num_workers=1,
            debug=False,
        )
        # GA should find IndentWidth=4; nevergrad mock may not find categoricals
        assert isinstance(result, OptimizationResult)
        assert result.best_config["IndentWidth"] == 4

    def test_all_fixed_params_returns_initial(self):
        space = SearchSpace(
            parameters={
                "a": ParameterDef(name="a", param_type="int", fixed=True),
            }
        )
        fitness = _make_fitness({"a": 1})
        result = run_phased_optimization(
            search_space=space,
            fitness_fn=fitness,
            initial_config={"a": 1},
            num_islands=1,
            population_size=4,
            num_workers=1,
            debug=False,
        )
        assert result.best_config == {"a": 1}
        assert result.best_fitness == 0.0

    def test_debug_output(self, capsys):
        """Debug mode should print phase information."""
        space = SearchSpace(
            parameters={
                "a": ParameterDef(
                    name="a",
                    param_type="int",
                    possible_values=[1, 2],
                    tier="resolve",
                ),
            }
        )
        fitness = _make_fitness({"a": 2})
        _ = run_phased_optimization(
            search_space=space,
            fitness_fn=fitness,
            initial_config={"a": 1},
            num_islands=1,
            population_size=4,
            num_workers=1,
            debug=True,
        )
        captured = capsys.readouterr()
        assert "Phased Optimization" in captured.err
        assert "resolve" in captured.err

    @patch("src.optimization_engine.phased.run_nevergrad_optimization")
    @patch("src.optimization_engine.phased.run_island_ga")
    def test_polish_tier_params_are_optimized(self, mock_ga, mock_ng):
        """Undetected options with tier='polish' are optimized in the polish phase."""
        from src.optimization_engine.types import Individual

        mock_ga.return_value = Individual(config={"a": 3, "b": "x"}, fitness=1.0)
        mock_ng.return_value = OptimizationResult(
            best_config={"a": 3, "b": "y"}, best_fitness=0.0
        )

        space = SearchSpace(
            parameters={
                "a": ParameterDef(
                    name="a",
                    param_type="int",
                    possible_values=[1, 2, 3],
                    fixed=False,
                    tier="polish",
                ),
                "b": ParameterDef(
                    name="b",
                    param_type="str",
                    possible_values=["x", "y"],
                    fixed=False,
                    tier="polish",
                ),
            }
        )
        fitness = _make_fitness({"a": 3, "b": "y"})
        result = run_phased_optimization(
            search_space=space,
            fitness_fn=fitness,
            initial_config={"a": 1, "b": "x"},
            num_islands=1,
            population_size=4,
            num_workers=1,
        )
        # Polish phase should have found the target
        assert result.best_fitness == 0
        assert result.best_config["a"] == 3
        assert result.best_config["b"] == "y"
