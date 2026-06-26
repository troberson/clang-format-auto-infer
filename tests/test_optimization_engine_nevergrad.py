"""Tests for the generic nevergrad optimization wrapper."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest

from src.optimization_engine.nevergrad import (
    build_instrumentation,
    _convert_param_types,  # pyright: ignore[reportPrivateUsage]
    run_nevergrad_optimization,
)
from src.optimization_engine.types import (
    OptimizationResult,
    ParameterDef,
    SearchSpace,
)


def _make_search_space() -> SearchSpace:
    return SearchSpace(
        parameters={
            "a": ParameterDef(name="a", param_type="int", possible_values=[1, 2, 3]),
            "b": ParameterDef(
                name="b", param_type="bool", possible_values=[True, False]
            ),
            "c": ParameterDef(name="c", param_type="str", possible_values=["x", "y"]),
            "d": ParameterDef(name="d", param_type="str", fixed=True),
        }
    )


def _simple_objective(config: dict[str, object]) -> float:
    """Simple fitness: sum of numeric values, lower is better."""
    return sum(v if isinstance(v, (int, float)) else 0 for v in config.values())


class TestBuildInstrumentation:
    def test_includes_mutable_int(self):
        ss = SearchSpace(
            parameters={
                "a": ParameterDef(name="a", param_type="int", possible_values=[1, 2]),
            }
        )
        inst = build_instrumentation(ss)
        assert "a" in inst.kwargs

    def test_includes_mutable_bool(self):
        ss = SearchSpace(
            parameters={
                "b": ParameterDef(
                    name="b", param_type="bool", possible_values=[True, False]
                ),
            }
        )
        inst = build_instrumentation(ss)
        assert "b" in inst.kwargs

    def test_includes_mutable_str(self):
        ss = SearchSpace(
            parameters={
                "c": ParameterDef(
                    name="c", param_type="str", possible_values=["x", "y"]
                ),
            }
        )
        inst = build_instrumentation(ss)
        assert "c" in inst.kwargs

    def test_excludes_fixed_parameter(self):
        ss = SearchSpace(
            parameters={
                "d": ParameterDef(name="d", param_type="str", fixed=True),
            }
        )
        inst = build_instrumentation(ss)
        assert "d" not in inst.kwargs

    def test_excludes_no_values(self):
        ss = SearchSpace(
            parameters={
                "e": ParameterDef(name="e", param_type="int"),
            }
        )
        inst = build_instrumentation(ss)
        assert "e" not in inst.kwargs

    def test_empty_search_space(self):
        ss = SearchSpace(parameters={})
        inst = build_instrumentation(ss)
        assert inst.kwargs == {}


class TestConvertParamTypes:
    def test_converts_int(self):
        ss = SearchSpace(
            parameters={
                "a": ParameterDef(name="a", param_type="int", possible_values=[1, 2]),
            }
        )
        result = _convert_param_types({"a": 2.0}, ss)
        assert result["a"] == 2

    def test_converts_bool(self):
        ss = SearchSpace(
            parameters={
                "b": ParameterDef(
                    name="b", param_type="bool", possible_values=[True, False]
                ),
            }
        )
        result = _convert_param_types({"b": 1}, ss)
        assert result["b"] is True

    def test_skips_unknown_key(self):
        ss = SearchSpace(parameters={})
        result = _convert_param_types({"x": 1}, ss)
        assert result["x"] == 1

    def test_int_conversion_failure_keeps_original(self):
        ss = SearchSpace(
            parameters={
                "a": ParameterDef(name="a", param_type="int", possible_values=[1, 2]),
            }
        )
        result = _convert_param_types({"a": "not_a_number"}, ss)
        assert result["a"] == "not_a_number"


class TestRunNevergradOptimization:
    def _build_mock_optimizer(self):
        """Build a mock nevergrad optimizer for testing."""
        mock_optimizer = MagicMock()
        mock_param = MagicMock()
        mock_param.kwargs = {"a": 1}
        mock_optimizer.ask.return_value = mock_param

        mock_rec = MagicMock()
        mock_rec.kwargs = {"a": 1}
        mock_optimizer.provide_recommendation.return_value = mock_rec
        return mock_optimizer

    def _build_mock_executor_and_future(
        self, _objective: Callable[[dict[str, object]], float]
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

    @patch("src.optimization_engine.nevergrad.ng.optimizers.registry")
    def test_budget_zero_returns_initial(self, mock_registry):
        mock_optimizer = self._build_mock_optimizer()
        mock_registry.__getitem__.return_value = MagicMock(return_value=mock_optimizer)

        ss = _make_search_space()
        result = run_nevergrad_optimization(
            search_space=ss,
            objective=_simple_objective,
            budget=0,
            num_workers=1,
            initial_config={"a": 3, "b": True},
        )
        assert result.best_config == {"a": 3, "b": True}
        assert result.best_fitness == float("inf")

    @patch("src.optimization_engine.nevergrad.ng.optimizers.registry")
    @patch("src.optimization_engine.nevergrad.concurrent.futures.ProcessPoolExecutor")
    @patch("src.optimization_engine.nevergrad.concurrent.futures.wait")
    def test_runs_and_returns_result(self, mock_wait, mock_pool_cls, mock_registry):
        mock_optimizer = self._build_mock_optimizer()
        mock_registry.__getitem__.return_value = MagicMock(return_value=mock_optimizer)

        mock_executor, mock_future = self._build_mock_executor_and_future(
            _simple_objective
        )
        mock_pool_cls.return_value = mock_executor

        # wait must return the actual future that was submitted
        mock_wait.return_value = ([mock_future], [])

        ss = SearchSpace(
            parameters={
                "a": ParameterDef(name="a", param_type="int", possible_values=[1, 2]),
            }
        )
        result = run_nevergrad_optimization(
            search_space=ss,
            objective=_simple_objective,
            budget=1,
            num_workers=1,
            initial_config={"a": 2},
        )
        assert isinstance(result, OptimizationResult)
        assert result.best_fitness <= 2

    @patch("src.optimization_engine.nevergrad.ng.optimizers.registry")
    def test_unknown_optimizer_exits(self, mock_registry, capsys):
        mock_registry.__getitem__.side_effect = KeyError("bad_opt")
        mock_registry.keys.return_value = ["DE", "CMA"]

        ss = SearchSpace(parameters={})
        with pytest.raises(SystemExit):
            _ = run_nevergrad_optimization(
                search_space=ss,
                objective=_simple_objective,
                budget=1,
                num_workers=1,
                optimizer_name="bad_opt",
            )
        captured = capsys.readouterr()
        assert "not found" in captured.err

    @patch("src.optimization_engine.nevergrad.ng.optimizers.registry")
    @patch("src.optimization_engine.nevergrad.concurrent.futures.ProcessPoolExecutor")
    @patch("src.optimization_engine.nevergrad.concurrent.futures.wait")
    def test_executor_shutdown_called(self, mock_wait, mock_pool_cls, mock_registry):
        mock_optimizer = self._build_mock_optimizer()
        mock_registry.__getitem__.return_value = MagicMock(return_value=mock_optimizer)

        mock_executor, mock_future = self._build_mock_executor_and_future(
            _simple_objective
        )
        mock_pool_cls.return_value = mock_executor
        mock_wait.return_value = ([mock_future], [])

        ss = SearchSpace(
            parameters={
                "a": ParameterDef(name="a", param_type="int", possible_values=[1, 2]),
            }
        )
        _ = run_nevergrad_optimization(
            search_space=ss,
            objective=_simple_objective,
            budget=1,
            num_workers=1,
        )
        mock_executor.shutdown.assert_called()

    @patch("src.optimization_engine.nevergrad.ng.optimizers.registry")
    @patch("src.optimization_engine.nevergrad.concurrent.futures.ProcessPoolExecutor")
    @patch("src.optimization_engine.nevergrad.concurrent.futures.wait")
    def test_debug_prints_progress(
        self, mock_wait, mock_pool_cls, mock_registry, capsys
    ):
        mock_optimizer = self._build_mock_optimizer()
        mock_registry.__getitem__.return_value = MagicMock(return_value=mock_optimizer)

        mock_executor, mock_future = self._build_mock_executor_and_future(
            _simple_objective
        )
        mock_pool_cls.return_value = mock_executor
        mock_wait.return_value = ([mock_future], [])

        ss = SearchSpace(
            parameters={
                "a": ParameterDef(name="a", param_type="int", possible_values=[1, 2]),
            }
        )
        _ = run_nevergrad_optimization(
            search_space=ss,
            objective=_simple_objective,
            budget=1,
            num_workers=1,
            debug=True,
        )
        captured = capsys.readouterr()
        assert "Submitted task" in captured.err

    @patch("src.optimization_engine.nevergrad.ng.optimizers.registry")
    @patch("src.optimization_engine.nevergrad.concurrent.futures.ProcessPoolExecutor")
    @patch("src.optimization_engine.nevergrad.concurrent.futures.wait")
    def test_none_recommendation_returns_initial(
        self, mock_wait, mock_pool_cls, mock_registry, capsys
    ):
        mock_optimizer = self._build_mock_optimizer()
        mock_optimizer.provide_recommendation.return_value = None
        mock_registry.__getitem__.return_value = MagicMock(return_value=mock_optimizer)

        mock_executor, mock_future = self._build_mock_executor_and_future(
            _simple_objective
        )
        mock_pool_cls.return_value = mock_executor
        mock_wait.return_value = ([mock_future], [])

        ss = SearchSpace(
            parameters={
                "a": ParameterDef(name="a", param_type="int", possible_values=[1, 2]),
            }
        )
        result = run_nevergrad_optimization(
            search_space=ss,
            objective=_simple_objective,
            budget=1,
            num_workers=1,
            initial_config={"a": 2},
        )
        assert result.best_config == {"a": 2}
        captured = capsys.readouterr()
        assert "No recommendation" in captured.err

    @patch("src.optimization_engine.nevergrad.ng.optimizers.registry")
    @patch("src.optimization_engine.nevergrad.concurrent.futures.ProcessPoolExecutor")
    @patch("src.optimization_engine.nevergrad.concurrent.futures.wait")
    def test_evaluation_error_told_as_inf(
        self, mock_wait, mock_pool_cls, mock_registry
    ):
        mock_optimizer = self._build_mock_optimizer()
        mock_registry.__getitem__.return_value = MagicMock(return_value=mock_optimizer)

        def failing_objective(_config: dict[str, object]) -> float:
            raise ValueError("boom")

        mock_executor, mock_future = self._build_mock_executor_and_future(
            failing_objective
        )
        mock_pool_cls.return_value = mock_executor
        mock_wait.return_value = ([mock_future], [])

        ss = SearchSpace(
            parameters={
                "a": ParameterDef(name="a", param_type="int", possible_values=[1, 2]),
            }
        )
        _ = run_nevergrad_optimization(
            search_space=ss,
            objective=failing_objective,
            budget=1,
            num_workers=1,
        )
        # The candidate from ask() is told with inf on error
        mock_optimizer.tell.assert_called_once()
        call_args = mock_optimizer.tell.call_args
        assert call_args[0][1] == float("inf")

    @patch("src.optimization_engine.nevergrad.ng.optimizers.registry")
    @patch("src.optimization_engine.nevergrad.concurrent.futures.ProcessPoolExecutor")
    @patch("src.optimization_engine.nevergrad.concurrent.futures.wait")
    def test_progress_print_every_50_evaluations(
        self, mock_wait, mock_pool_cls, mock_registry, capsys
    ):
        """The non-debug progress print fires every 50 evaluations."""
        mock_optimizer = self._build_mock_optimizer()
        mock_registry.__getitem__.return_value = MagicMock(return_value=mock_optimizer)

        mock_executor, mock_future = self._build_mock_executor_and_future(
            _simple_objective
        )
        mock_pool_cls.return_value = mock_executor

        # Simulate 50 evaluations by having wait return 50 times
        mock_wait.side_effect = lambda futures, return_when: ([mock_future], [])

        ss = SearchSpace(
            parameters={
                "a": ParameterDef(name="a", param_type="int", possible_values=[1, 2]),
            }
        )
        _ = run_nevergrad_optimization(
            search_space=ss,
            objective=_simple_objective,
            budget=50,
            num_workers=1,
            debug=False,
        )
        captured = capsys.readouterr()
        assert "--- Evaluation 50/50" in captured.err
