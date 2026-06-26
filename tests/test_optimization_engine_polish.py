"""Tests for the generic coordinate descent polish."""

from collections.abc import Mapping

from src.optimization_engine.polish import polish_coordinate_descent
from src.optimization_engine.types import (
    Individual,
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
            "c": ParameterDef(name="c", param_type="str", fixed=True),
        }
    )


def _fitness(config: Mapping[str, object]) -> float:
    """Simple fitness: sum of numeric values, lower is better."""
    return sum(v if isinstance(v, (int, float)) else 0 for v in config.values())


class TestPolishCoordinateDescent:
    def test_improves_fitness(self):
        ss = _make_search_space()
        config = {"a": 3, "b": True, "c": "x"}
        result = polish_coordinate_descent(
            config, _fitness(config), ss, _fitness, max_passes=3
        )
        assert result.config["a"] == 1
        assert result.fitness < _fitness({"a": 3, "b": True, "c": "x"})

    def test_converges_when_no_improvements(self, capsys):
        ss = _make_search_space()
        config = {"a": 1, "b": False, "c": "x"}
        _ = polish_coordinate_descent(
            config, _fitness(config), ss, _fitness, max_passes=10, debug=True
        )
        captured = capsys.readouterr()
        assert "converged after 1 pass" in captured.err

    def test_no_mutable_options_returns_original(self):
        ss = SearchSpace(
            parameters={
                "x": ParameterDef(name="x", param_type="str", fixed=True),
            }
        )
        config = {"x": "hello"}
        result = polish_coordinate_descent(config, 5.0, ss, _fitness, max_passes=3)
        assert result.config == {"x": "hello"}
        assert result.fitness == 5.0

    def test_max_passes_not_exceeded(self):
        """Even with max_passes=1, polish stops after one pass if it converges."""
        ss = SearchSpace(
            parameters={
                "a": ParameterDef(name="a", param_type="int", possible_values=[1, 2]),
            }
        )
        config = {"a": 2}
        result = polish_coordinate_descent(
            config, _fitness(config), ss, _fitness, max_passes=1
        )
        assert result.config["a"] == 1
        assert result.fitness == 1

    def test_skips_parameter_not_in_config(self):
        ss = SearchSpace(
            parameters={
                "a": ParameterDef(name="a", param_type="int", possible_values=[1, 2]),
                "z": ParameterDef(name="z", param_type="int", possible_values=[1, 2]),
            }
        )
        config = {"a": 2}
        result = polish_coordinate_descent(
            config, _fitness(config), ss, _fitness, max_passes=3
        )
        assert "z" not in result.config

    def test_debug_prints_no_mutable(self, capsys):
        ss = SearchSpace(parameters={})
        config = {"a": 1}
        _ = polish_coordinate_descent(
            config, 5.0, ss, _fitness, max_passes=3, debug=True
        )
        captured = capsys.readouterr()
        assert "No mutable options" in captured.err

    def test_debug_prints_convergence(self, capsys):
        ss = _make_search_space()
        config = {"a": 1, "b": False, "c": "x"}
        _ = polish_coordinate_descent(
            config, _fitness(config), ss, _fitness, max_passes=10, debug=True
        )
        captured = capsys.readouterr()
        assert "Starting coordinate descent polish" in captured.err
        assert "converged" in captured.err

    def test_returns_individual(self):
        ss = _make_search_space()
        config = {"a": 3, "b": True}
        result = polish_coordinate_descent(
            config, _fitness(config), ss, _fitness, max_passes=3
        )
        assert isinstance(result, Individual)
        assert result.fitness <= _fitness({"a": 3, "b": True})

    def test_max_passes_reached(self, capsys):
        """When fitness always improves, polish stops at max_passes."""
        call_count = 0

        def _always_improving(_config: Mapping[str, object]) -> float:
            nonlocal call_count
            call_count += 1
            # Return a value that always looks better than the previous pass
            return -call_count

        ss = SearchSpace(
            parameters={
                "a": ParameterDef(name="a", param_type="int", possible_values=[1, 2]),
            }
        )
        config = {"a": 1}
        _ = polish_coordinate_descent(
            config, 0.0, ss, _always_improving, max_passes=2, debug=True
        )
        captured = capsys.readouterr()
        assert "reached max passes" in captured.err
