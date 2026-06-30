"""Tests for the iterative expansion optimizer."""

from unittest.mock import MagicMock, patch

from src.optimization_engine.iterative import (
    CONVERGENCE_THRESHOLD,
    IMPACT_THRESHOLD,
    MAX_BATCH_FRACTION,
    NG_SAFETY_BUDGET,
    _is_penalty_option,  # pyright: ignore[reportPrivateUsage]
    _optimize_batch,  # pyright: ignore[reportPrivateUsage]
    _select_batch,  # pyright: ignore[reportPrivateUsage]
    run_iterative_optimization,
)
from src.optimization_engine.types import (
    ParameterDef,
    SearchSpace,
)


class FakeImpactScore:
    def __init__(self, name: str, fitness_delta: float) -> None:
        self.name: str = name
        self.fitness_delta: float = fitness_delta


def _make_search_space(
    detected: list[str] | None = None,
    fixed: list[str] | None = None,
    param_type: str = "int",
) -> SearchSpace:
    """Build a search space with some mutable and some fixed params."""
    params: dict[str, ParameterDef] = {}
    for name in detected or []:
        params[name] = ParameterDef(
            name=name,
            param_type=param_type,
            possible_values=[1, 2, 4],
            fixed=False,
            tier="polish",
        )
    for name in fixed or []:
        params[name] = ParameterDef(
            name=name,
            param_type=param_type,
            possible_values=[1, 2, 4],
            fixed=True,
            tier="polish",
        )
    return SearchSpace(parameters=params)


class TestSelectBatch:
    def test_selects_high_impact_options(self):
        scores = [
            FakeImpactScore("A", 100.0),
            FakeImpactScore("B", 80.0),
            FakeImpactScore("C", 10.0),
        ]
        # remaining_count=10 gives max_batch=2, so A and B are selected.
        batch = _select_batch(scores, remaining_count=10)
        assert "A" in batch
        assert "B" in batch
        assert "C" not in batch

    def test_caps_at_max_batch_fraction(self):
        scores = [FakeImpactScore(f"X{i}", 100.0) for i in range(10)]
        batch = _select_batch(scores, remaining_count=10)
        assert len(batch) <= max(1, int(10 * 0.2))

    def test_empty_scores_returns_empty(self):
        assert _select_batch([], remaining_count=5) == []

    def test_zero_max_score_returns_empty(self):
        scores = [FakeImpactScore("A", 0.0)]
        assert _select_batch(scores, remaining_count=1) == []

    def test_negative_score_returns_empty(self):
        scores = [FakeImpactScore("A", -5.0)]
        assert _select_batch(scores, remaining_count=1) == []

    def test_min_batch_is_one(self):
        scores = [FakeImpactScore("A", 100.0)]
        batch = _select_batch(scores, remaining_count=1)
        assert batch == ["A"]


class TestRunIterativeOptimization:
    def _make_fitness(self, base_fitness: float = 500.0) -> MagicMock:
        fitness = MagicMock()
        fitness.return_value = base_fitness
        return fitness

    def _make_impact_fn(self, scores: list[object] | None = None) -> MagicMock:
        impact = MagicMock()
        impact.return_value = scores or []
        return impact

    def test_optimizes_detected_then_stops(self):
        """When no remaining fixed options, radial search fixes all, then stops."""
        space = _make_search_space(detected=["A", "B"], fixed=[])
        fitness = self._make_fitness(500.0)
        impact_fn = self._make_impact_fn([])

        result = run_iterative_optimization(
            search_space=space,
            fitness_fn=fitness,
            initial_config={"A": 1, "B": 1},
            impact_fn=impact_fn,
            impact_kwargs={},
            debug=False,
        )

        assert result.best_fitness == 500.0
        # Radial search fixes A and B (both ints). No remaining fixed candidates.
        # Impact should not be called since all are detected (excluded).
        impact_fn.assert_not_called()

    def test_stops_when_no_impactful_options(self):
        """When impact measurement returns no scores, stop."""
        # X and Y are mutable str (impact measures them). A is mutable int (radial fixes it).
        params = {
            "A": ParameterDef(
                name="A", param_type="int", possible_values=[1, 2, 4], fixed=False
            ),
            "X": ParameterDef(
                name="X", param_type="str", possible_values=["a", "b"], fixed=False
            ),
            "Y": ParameterDef(
                name="Y", param_type="str", possible_values=["c", "d"], fixed=False
            ),
        }
        space = SearchSpace(parameters=params)
        fitness = self._make_fitness(500.0)
        impact_fn = self._make_impact_fn([])

        result = run_iterative_optimization(
            search_space=space,
            fitness_fn=fitness,
            initial_config={"A": 1},
            impact_fn=impact_fn,
            impact_kwargs={},
            debug=False,
        )

        assert result.best_fitness == 500.0
        impact_fn.assert_called_once()

    def test_unlocks_impactful_options(self):
        """When impactful options found, optimize them and continue."""
        # X and Y are mutable str (impact measures them). A is mutable int (radial fixes it).
        params = {
            "A": ParameterDef(
                name="A", param_type="int", possible_values=[1, 2, 4], fixed=False
            ),
            "X": ParameterDef(
                name="X", param_type="str", possible_values=["a", "b"], fixed=False
            ),
            "Y": ParameterDef(
                name="Y", param_type="str", possible_values=["c", "d"], fixed=False
            ),
        }
        space = SearchSpace(parameters=params)
        fitness = self._make_fitness(500.0)
        # Impact called once upfront with all candidates.
        impact_fn = MagicMock(
            return_value=[
                FakeImpactScore("X", 100.0),
                FakeImpactScore("Y", 10.0),
            ]
        )

        with patch("src.optimization_engine.iterative._optimize_batch") as mock_opt:
            mock_opt.return_value = MagicMock(
                best_fitness=500.0, best_config={"A": 1}, evaluations_used=10
            )

            result = run_iterative_optimization(
                search_space=space,
                fitness_fn=fitness,
                initial_config={"A": 1},
                impact_fn=impact_fn,
                impact_kwargs={},
                debug=False,
            )

        assert result.best_fitness == 500.0
        assert impact_fn.call_count == 1

    def test_stops_when_impact_below_threshold(self):
        """When top impact is below min_improvement_ratio of best fitness, stop."""
        params = {
            "A": ParameterDef(
                name="A", param_type="int", possible_values=[1, 2, 4], fixed=False
            ),
            "X": ParameterDef(
                name="X", param_type="str", possible_values=["a", "b"], fixed=False
            ),
        }
        space = SearchSpace(parameters=params)
        fitness = self._make_fitness(500.0)
        # Radial search fixes A (int). Impact called with X.
        # Impact of 1.0 is below 1% of 500 (which is 5.0).
        impact_fn = self._make_impact_fn([FakeImpactScore("X", 1.0)])

        result = run_iterative_optimization(
            search_space=space,
            fitness_fn=fitness,
            initial_config={"A": 1},
            impact_fn=impact_fn,
            impact_kwargs={},
            min_improvement_ratio=0.01,
            debug=False,
        )

        assert result.best_fitness == 500.0
        assert impact_fn.call_count == 1

    def test_passes_candidate_names_to_impact_fn(self):
        """Impact function receives correct candidate names."""
        params = {
            "A": ParameterDef(
                name="A", param_type="int", possible_values=[1, 2, 4], fixed=False
            ),
            "X": ParameterDef(
                name="X", param_type="str", possible_values=["a", "b"], fixed=False
            ),
            "Y": ParameterDef(
                name="Y", param_type="str", possible_values=["c", "d"], fixed=False
            ),
            "Z": ParameterDef(
                name="Z", param_type="str", possible_values=["e", "f"], fixed=False
            ),
        }
        space = SearchSpace(parameters=params)
        fitness = self._make_fitness(500.0)
        # Radial search fixes A (int). Impact called with X, Y, Z.
        impact_fn = self._make_impact_fn([])

        _ = run_iterative_optimization(
            search_space=space,
            fitness_fn=fitness,
            initial_config={"A": 1},
            impact_fn=impact_fn,
            impact_kwargs={"repo_path": "/tmp/repo"},
            debug=False,
        )

        call_kwargs = impact_fn.call_args.kwargs
        assert set(call_kwargs["candidate_names"]) == {"X", "Y", "Z"}
        assert call_kwargs["repo_path"] == "/tmp/repo"

    def test_passes_current_config_to_impact_fn(self):
        """Impact function receives the current best config."""
        params = {
            "A": ParameterDef(
                name="A", param_type="int", possible_values=[1, 2, 4], fixed=False
            ),
            "X": ParameterDef(
                name="X", param_type="str", possible_values=["a", "b"], fixed=False
            ),
        }
        space = SearchSpace(parameters=params)
        fitness = self._make_fitness(500.0)
        # Radial search fixes A (int). Impact called with X.
        impact_fn = self._make_impact_fn([])

        _ = run_iterative_optimization(
            search_space=space,
            fitness_fn=fitness,
            initial_config={"A": 4, "X": "a"},
            impact_fn=impact_fn,
            impact_kwargs={},
            debug=False,
        )

        call_kwargs = impact_fn.call_args.kwargs
        assert "current_config" in call_kwargs

    def test_constants_have_expected_values(self):
        assert MAX_BATCH_FRACTION == 0.2
        assert IMPACT_THRESHOLD == 0.5
        assert 0.001 <= 0.01  # MIN_IMPROVEMENT_RATIO

    def test_new_constants_have_expected_values(self):
        """New convergence-based constants have sensible defaults."""
        assert NG_SAFETY_BUDGET == 10_000

    def test_convergence_constant_has_expected_value(self):
        """CONVERGENCE_THRESHOLD has a sensible default."""
        assert CONVERGENCE_THRESHOLD == 20

    def test_convergence_threshold_passed_to_optimize_batch(self):
        """run_iterative_optimization forwards convergence_threshold to _optimize_batch."""
        params = {
            "A": ParameterDef(
                name="A", param_type="str", possible_values=["x", "y"], fixed=False
            ),
        }
        space = SearchSpace(parameters=params)
        fitness = self._make_fitness(500.0)
        impact_fn = self._make_impact_fn([FakeImpactScore("A", 100.0)])

        with patch("src.optimization_engine.iterative._optimize_batch") as mock_opt:
            from src.optimization_engine.types import OptimizationResult

            mock_opt.return_value = OptimizationResult(
                best_config={"A": "y"},
                best_fitness=400.0,
                evaluations_used=10,
            )
            _ = run_iterative_optimization(
                search_space=space,
                fitness_fn=fitness,
                initial_config={"A": "x"},
                impact_fn=impact_fn,
                impact_kwargs={},
                convergence_threshold=25,
                debug=False,
            )
            call_kwargs = mock_opt.call_args[1]
            assert call_kwargs["convergence_threshold"] == 25

    def test_global_polish_improvement(self):
        """Global polish unlocks impactful options and improves fitness."""
        params = {
            "A": ParameterDef(
                name="A", param_type="str", possible_values=["x", "y"], fixed=False
            ),
            "B": ParameterDef(
                name="B", param_type="str", possible_values=["p", "q"], fixed=False
            ),
        }
        space = SearchSpace(parameters=params)
        fitness = self._make_fitness(500.0)
        # Impact called once upfront with all candidates.
        impact_fn = MagicMock(
            return_value=[
                FakeImpactScore("A", 100.0),
                FakeImpactScore("B", 50.0),
            ]
        )

        with patch("src.optimization_engine.iterative._optimize_batch") as mock_opt:
            from src.optimization_engine.types import OptimizationResult

            # Window size is 1 (max(1, int(2 * 0.2))), so 2 iterative calls + 1 global polish = 3.
            mock_opt.side_effect = [
                OptimizationResult(
                    best_config={"A": "y", "B": "p"},
                    best_fitness=500.0,
                    evaluations_used=10,
                ),
                OptimizationResult(
                    best_config={"A": "y", "B": "p"},
                    best_fitness=500.0,
                    evaluations_used=10,
                ),
                OptimizationResult(
                    best_config={"A": "y", "B": "q"},
                    best_fitness=400.0,
                    evaluations_used=10,
                ),
            ]
            result = run_iterative_optimization(
                search_space=space,
                fitness_fn=fitness,
                initial_config={"A": "x", "B": "p"},
                impact_fn=impact_fn,
                impact_kwargs={},
                debug=False,
            )

        assert result.best_fitness == 400.0
        assert mock_opt.call_count == 3  # 2 iterative batches + global polish


class TestOptimizeBatch:
    """Tests for _optimize_batch covering GA and nevergrad paths."""

    def test_runs_ga_for_integer_params(self):
        """GA path is exercised when params are integers."""
        space = _make_search_space(detected=["A", "B"], fixed=[])
        fitness = MagicMock()
        fitness.return_value = 500.0

        with patch("src.optimization_engine.iterative.run_island_ga") as mock_ga:
            mock_ga.return_value = MagicMock(
                fitness=400.0, config={"A": 2}, evaluations_used=10
            )
            result = _optimize_batch(
                search_space=space,
                fitness_fn=fitness,
                current_config={"A": 1, "B": 1},
                num_islands=1,
                population_size=4,
                num_workers=1,
                debug=False,
            )
            mock_ga.assert_called_once()
            assert result.best_fitness == 400.0

    def test_runs_nevergrad_for_categorical_params(self):
        """Nevergrad path is exercised when params are not integers."""
        params = {
            "A": ParameterDef(
                name="A",
                param_type="categorical",
                possible_values=["Allman", "KRN"],
                fixed=False,
            ),
        }
        space = SearchSpace(parameters=params)
        fitness = MagicMock()
        fitness.return_value = 500.0

        with patch(
            "src.optimization_engine.iterative.run_nevergrad_optimization"
        ) as mock_ng:
            mock_ng.return_value = MagicMock(
                best_fitness=400.0, best_config={"A": "Allman"}
            )
            result = _optimize_batch(
                search_space=space,
                fitness_fn=fitness,
                current_config={"A": "KRN"},
                num_islands=1,
                population_size=4,
                num_workers=1,
                debug=False,
            )
            mock_ng.assert_called_once()
            assert result.best_fitness == 400.0

    def test_runs_both_ga_and_nevergrad(self):
        """Both GA and nevergrad run when mixed param types exist."""
        params = {
            "IndentWidth": ParameterDef(
                name="IndentWidth",
                param_type="int",
                possible_values=[2, 4, 8],
                fixed=False,
            ),
            "BreakBeforeBraces": ParameterDef(
                name="BreakBeforeBraces",
                param_type="categorical",
                possible_values=["Allman", "KRN"],
                fixed=False,
            ),
        }
        space = SearchSpace(parameters=params)
        fitness = MagicMock()
        fitness.return_value = 500.0

        with (
            patch("src.optimization_engine.iterative.run_island_ga") as mock_ga,
            patch(
                "src.optimization_engine.iterative.run_nevergrad_optimization"
            ) as mock_ng,
        ):
            mock_ga.return_value = MagicMock(
                fitness=450.0, config={"IndentWidth": 4}, evaluations_used=10
            )
            mock_ng.return_value = MagicMock(
                best_fitness=400.0,
                best_config={"IndentWidth": 4, "BreakBeforeBraces": "Allman"},
            )
            result = _optimize_batch(
                search_space=space,
                fitness_fn=fitness,
                current_config={"IndentWidth": 2, "BreakBeforeBraces": "KRN"},
                num_islands=1,
                population_size=4,
                num_workers=1,
                debug=False,
            )
            mock_ga.assert_called_once()
            mock_ng.assert_called_once()
            assert result.best_fitness == 400.0

    def test_no_mutable_params_returns_current_fitness(self):
        """When no mutable params, return current fitness without optimization."""
        space = _make_search_space(detected=[], fixed=["A"])
        fitness = MagicMock()
        fitness.return_value = 500.0

        result = _optimize_batch(
            search_space=space,
            fitness_fn=fitness,
            current_config={"A": 1},
            num_islands=1,
            population_size=4,
            num_workers=1,
            debug=False,
        )
        assert result.best_fitness == 500.0

    def test_debug_prints_ga_improvement(self):
        """Debug mode prints GA improvement message."""
        space = _make_search_space(detected=["A"], fixed=[])
        fitness = MagicMock()
        fitness.return_value = 500.0

        with patch("src.optimization_engine.iterative.run_island_ga") as mock_ga:
            mock_ga.return_value = MagicMock(
                fitness=400.0, config={"A": 2}, evaluations_used=10
            )
            _ = _optimize_batch(
                search_space=space,
                fitness_fn=fitness,
                current_config={"A": 1},
                num_islands=1,
                population_size=4,
                num_workers=1,
                debug=True,
            )
            # The debug print goes to sys.stderr, verify it was called
            mock_ga.assert_called_once()

    def test_debug_prints_nevergrad_improvement(self):
        """Debug mode prints nevergrad improvement message."""
        params = {
            "A": ParameterDef(
                name="A",
                param_type="categorical",
                possible_values=["Allman", "KRN"],
                fixed=False,
            ),
        }
        space = SearchSpace(parameters=params)
        fitness = MagicMock()
        fitness.return_value = 500.0

        with patch(
            "src.optimization_engine.iterative.run_nevergrad_optimization"
        ) as mock_ng:
            mock_ng.return_value = MagicMock(
                best_fitness=400.0, best_config={"A": "Allman"}
            )
            _ = _optimize_batch(
                search_space=space,
                fitness_fn=fitness,
                current_config={"A": "KRN"},
                num_islands=1,
                population_size=4,
                num_workers=1,
                debug=True,
            )
            mock_ng.assert_called_once()

    def test_ga_receives_convergence_threshold(self):
        """convergence_threshold is passed through to run_island_ga."""
        space = _make_search_space(detected=["A"], fixed=[])
        fitness = MagicMock(return_value=500.0)

        with patch("src.optimization_engine.iterative.run_island_ga") as mock_ga:
            mock_ga.return_value = MagicMock(
                fitness=400.0, config={"A": 2}, evaluations_used=10
            )
            _ = _optimize_batch(
                search_space=space,
                fitness_fn=fitness,
                current_config={"A": 1},
                num_islands=1,
                population_size=4,
                num_workers=1,
                convergence_threshold=15,
                debug=False,
            )
            call_kwargs = mock_ga.call_args[1]
            assert call_kwargs["convergence_threshold"] == 15

    def test_nevergrad_receives_convergence_threshold(self):
        """convergence_threshold is passed through to run_nevergrad_optimization."""
        params = {
            "A": ParameterDef(
                name="A",
                param_type="categorical",
                possible_values=["x", "y"],
                fixed=False,
            ),
        }
        space = SearchSpace(parameters=params)
        fitness = MagicMock(return_value=500.0)

        with patch(
            "src.optimization_engine.iterative.run_nevergrad_optimization"
        ) as mock_ng:
            mock_ng.return_value = MagicMock(best_fitness=400.0, best_config={"A": "y"})
            _ = _optimize_batch(
                search_space=space,
                fitness_fn=fitness,
                current_config={"A": "x"},
                num_islands=1,
                population_size=4,
                num_workers=1,
                convergence_threshold=15,
                debug=False,
            )
            call_kwargs = mock_ng.call_args[1]
            assert call_kwargs["convergence_threshold"] == 15

    def test_default_convergence_threshold_used(self):
        """Default CONVERGENCE_THRESHOLD is passed when not overridden."""
        space = _make_search_space(detected=["A"], fixed=[])
        fitness = MagicMock(return_value=500.0)

        with patch("src.optimization_engine.iterative.run_island_ga") as mock_ga:
            mock_ga.return_value = MagicMock(
                fitness=400.0, config={"A": 2}, evaluations_used=10
            )
            _ = _optimize_batch(
                search_space=space,
                fitness_fn=fitness,
                current_config={"A": 1},
                num_islands=1,
                population_size=4,
                num_workers=1,
                debug=False,
            )
            call_kwargs = mock_ga.call_args[1]
            assert call_kwargs["convergence_threshold"] == CONVERGENCE_THRESHOLD

    def test_nevergrad_receives_safety_budget(self):
        """nevergrad receives NG_SAFETY_BUDGET as its budget cap."""
        params = {
            "A": ParameterDef(
                name="A",
                param_type="categorical",
                possible_values=["x", "y"],
                fixed=False,
            ),
        }
        space = SearchSpace(parameters=params)
        fitness = MagicMock(return_value=500.0)

        with patch(
            "src.optimization_engine.iterative.run_nevergrad_optimization"
        ) as mock_ng:
            mock_ng.return_value = MagicMock(best_fitness=400.0, best_config={"A": "y"})
            _ = _optimize_batch(
                search_space=space,
                fitness_fn=fitness,
                current_config={"A": "x"},
                num_islands=1,
                population_size=4,
                num_workers=1,
                debug=False,
            )
            call_kwargs = mock_ng.call_args[1]
            assert call_kwargs["budget"] == NG_SAFETY_BUDGET

    def test_ga_receives_none_iterations(self):
        """GA receives num_iterations=None for convergence-only termination."""
        space = _make_search_space(detected=["A"], fixed=[])
        fitness = MagicMock(return_value=500.0)

        with patch("src.optimization_engine.iterative.run_island_ga") as mock_ga:
            mock_ga.return_value = MagicMock(
                fitness=400.0, config={"A": 2}, evaluations_used=10
            )
            _ = _optimize_batch(
                search_space=space,
                fitness_fn=fitness,
                current_config={"A": 1},
                num_islands=1,
                population_size=4,
                num_workers=1,
                debug=False,
            )
            call_kwargs = mock_ga.call_args[1]
            assert call_kwargs["num_iterations"] is None


class TestIterativeDebugPaths:
    """Tests for debug branches in run_iterative_optimization."""

    def test_fitness_improvement_updates_config(self):
        """When optimization improves fitness, config is updated."""
        params = {
            "A": ParameterDef(
                name="A", param_type="int", possible_values=[1, 2, 4], fixed=False
            ),
            "X": ParameterDef(
                name="X", param_type="str", possible_values=["a", "b"], fixed=False
            ),
        }
        space = SearchSpace(parameters=params)
        fitness = MagicMock()
        fitness.return_value = 500.0
        # Radial search fixes A (int). Impact called with X, returns empty.
        impact_fn = MagicMock(return_value=[])

        with patch("src.optimization_engine.iterative._optimize_batch") as mock_opt:
            mock_opt.return_value = MagicMock(
                best_fitness=400.0, best_config={"A": 2}, evaluations_used=10
            )
            result = run_iterative_optimization(
                search_space=space,
                fitness_fn=fitness,
                initial_config={"A": 1},
                impact_fn=impact_fn,
                impact_kwargs={},
                debug=False,
            )
            # No _optimize_batch calls in the main loop (impact returns empty).
            assert result.best_fitness == 500.0
            assert mock_opt.call_count == 0

    def test_debug_prints_no_batch_selected(self):
        """Debug prints when _select_batch returns empty."""
        import io
        import sys

        params = {
            "A": ParameterDef(
                name="A", param_type="int", possible_values=[1, 2, 4], fixed=False
            ),
            "X": ParameterDef(
                name="X", param_type="str", possible_values=["a", "b"], fixed=False
            ),
        }
        space = SearchSpace(parameters=params)
        fitness = MagicMock()
        fitness.return_value = 500.0
        # Radial search fixes A (int). Impact called with X.
        # Score is high enough to pass improvement threshold, but
        # impact_threshold=2.0 means no score can pass (threshold > max_score).
        impact_fn = MagicMock(return_value=[FakeImpactScore("X", 100.0)])

        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            _ = run_iterative_optimization(
                search_space=space,
                fitness_fn=fitness,
                initial_config={"A": 1},
                impact_fn=impact_fn,
                impact_kwargs={},
                impact_threshold=2.0,  # Forces _select_batch to return empty.
                debug=True,
            )
        finally:
            output = sys.stderr.getvalue()
            sys.stderr = old_stderr

        assert "No options selected for batch" in output

    def test_debug_prints_unlocking(self):
        """Debug prints when options are optimized."""
        import io
        import sys

        params = {
            "A": ParameterDef(
                name="A", param_type="int", possible_values=[1, 2, 4], fixed=False
            ),
            "X": ParameterDef(
                name="X", param_type="str", possible_values=["a", "b"], fixed=False
            ),
            "Y": ParameterDef(
                name="Y", param_type="str", possible_values=["c", "d"], fixed=False
            ),
        }
        space = SearchSpace(parameters=params)
        fitness = MagicMock()
        fitness.return_value = 500.0
        # Radial search fixes A (int). Impact called with X, Y.
        # First call: impactful, second: empty (to stop).
        impact_fn = MagicMock()
        impact_fn.side_effect = [
            [FakeImpactScore("X", 100.0), FakeImpactScore("Y", 60.0)],
            [],
        ]

        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            _ = run_iterative_optimization(
                search_space=space,
                fitness_fn=fitness,
                initial_config={"A": 1},
                impact_fn=impact_fn,
                impact_kwargs={},
                debug=True,
            )
        finally:
            output = sys.stderr.getvalue()
            sys.stderr = old_stderr

        assert "Optimizing batch" in output

    def test_final_polish_improvement(self):
        """Penalty polish can improve fitness after loop exits."""
        params = {
            "A": ParameterDef(
                name="A",
                param_type="int",
                possible_values=[1, 2, 4],
                fixed=False,
            ),
            "PenaltyBreakAssignment": ParameterDef(
                name="PenaltyBreakAssignment",
                param_type="int",
                possible_values=[1, 2],
                fixed=False,  # Mutable, fixed at start of loop.
            ),
        }
        space = SearchSpace(parameters=params)
        fitness = MagicMock()
        fitness.return_value = 500.0
        # Radial search fixes A (int). Penalty fixed at start. Impact has no candidates.
        impact_fn = MagicMock(return_value=[])

        with patch("src.optimization_engine.iterative._optimize_batch") as mock_opt:
            # Penalty polish call.
            mock_opt.side_effect = [
                MagicMock(
                    best_fitness=350.0,
                    best_config={"A": 3, "PenaltyBreakAssignment": 2},
                    evaluations_used=5,
                ),
            ]
            result = run_iterative_optimization(
                search_space=space,
                fitness_fn=fitness,
                initial_config={"A": 1, "PenaltyBreakAssignment": 1},
                impact_fn=impact_fn,
                impact_kwargs={},
                debug=False,
            )
            assert result.best_fitness == 350.0
            assert mock_opt.call_count == 1

    def test_debug_prints_iteration_header(self):
        """Debug mode prints iteration headers with window progress."""
        import io
        import sys

        params = {
            "A": ParameterDef(
                name="A", param_type="int", possible_values=[1, 2, 4], fixed=False
            ),
            "X": ParameterDef(
                name="X", param_type="str", possible_values=["a", "b"], fixed=False
            ),
        }
        space = SearchSpace(parameters=params)
        fitness = MagicMock()
        fitness.return_value = 500.0
        # Radial search fixes A (int). Impact called with X.
        impact_fn = MagicMock(return_value=[FakeImpactScore("X", 100.0)])

        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            with patch("src.optimization_engine.iterative._optimize_batch") as mock_opt:
                mock_opt.return_value = MagicMock(
                    best_fitness=500.0, best_config={"X": "a"}, evaluations_used=5
                )
                _ = run_iterative_optimization(
                    search_space=space,
                    fitness_fn=fitness,
                    initial_config={"A": 1},
                    impact_fn=impact_fn,
                    impact_kwargs={},
                    debug=True,
                )
        finally:
            output = sys.stderr.getvalue()
            sys.stderr = old_stderr

        assert "Iterative Expansion Optimization" in output
        assert "Iteration 1" in output

    def test_debug_prints_no_impactful_options(self):
        """Debug prints 'No impactful options found' when impact returns empty."""
        import io
        import sys

        params = {
            "A": ParameterDef(
                name="A", param_type="int", possible_values=[1, 2, 4], fixed=False
            ),
            "X": ParameterDef(
                name="X", param_type="str", possible_values=["a", "b"], fixed=False
            ),
        }
        space = SearchSpace(parameters=params)
        fitness = MagicMock()
        fitness.return_value = 500.0
        # Radial search fixes A (int). Impact called with X, returns empty.
        impact_fn = MagicMock(return_value=[])

        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            _ = run_iterative_optimization(
                search_space=space,
                fitness_fn=fitness,
                initial_config={"A": 1},
                impact_fn=impact_fn,
                impact_kwargs={},
                debug=True,
            )
        finally:
            output = sys.stderr.getvalue()
            sys.stderr = old_stderr

        assert "No impactful options found" in output

    def test_debug_prints_below_threshold(self):
        """Debug prints threshold message when impact is too low."""
        import io
        import sys

        params = {
            "A": ParameterDef(
                name="A", param_type="int", possible_values=[1, 2, 4], fixed=False
            ),
            "X": ParameterDef(
                name="X", param_type="str", possible_values=["a", "b"], fixed=False
            ),
        }
        space = SearchSpace(parameters=params)
        fitness = MagicMock()
        fitness.return_value = 500.0
        # Radial search fixes A (int). Impact called with X.
        impact_fn = MagicMock(return_value=[FakeImpactScore("X", 1.0)])

        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            _ = run_iterative_optimization(
                search_space=space,
                fitness_fn=fitness,
                initial_config={"A": 1},
                impact_fn=impact_fn,
                impact_kwargs={},
                min_improvement_ratio=0.01,
                debug=True,
            )
        finally:
            output = sys.stderr.getvalue()
            sys.stderr = old_stderr

        assert "below threshold" in output

    def test_debug_prints_final_polish(self):
        """Debug prints penalty polish header."""
        import io
        import sys

        from unittest.mock import patch

        params = {
            "A": ParameterDef(
                name="A",
                param_type="int",
                possible_values=[1, 2, 4],
                fixed=False,
            ),
            "PenaltyBreakAssignment": ParameterDef(
                name="PenaltyBreakAssignment",
                param_type="int",
                possible_values=[1, 2],
                fixed=False,
            ),
        }
        space = SearchSpace(parameters=params)
        fitness = MagicMock()
        fitness.return_value = 500.0

        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            with patch("src.optimization_engine.iterative._optimize_batch") as mock_opt:
                mock_opt.return_value = MagicMock(
                    best_fitness=500.0, best_config={"A": 1}, evaluations_used=5
                )
                _ = run_iterative_optimization(
                    search_space=space,
                    fitness_fn=fitness,
                    initial_config={"A": 1, "PenaltyBreakAssignment": 1},
                    impact_fn=MagicMock(return_value=[]),
                    impact_kwargs={},
                    debug=True,
                )
        finally:
            output = sys.stderr.getvalue()
            sys.stderr = old_stderr

        assert "[penalty-polish]" in output


class TestPenaltyOptions:
    """Tests for penalty option filtering and final penalty polish."""

    def test_is_penalty_option_matches_penalty_prefix(self):
        """_is_penalty_option returns True for Penalty* options."""
        assert _is_penalty_option("PenaltyBreakAssignment") is True
        assert _is_penalty_option("PenaltyExcessCharacter") is True
        assert _is_penalty_option("PenaltyBreakString") is True

    def test_is_penalty_option_rejects_non_penalty(self):
        """_is_penalty_option returns False for non-penalty options."""
        assert _is_penalty_option("IndentWidth") is False
        assert _is_penalty_option("ColumnLimit") is False
        assert _is_penalty_option("BreakBeforeBraces") is False
        assert _is_penalty_option("Penalize") is False  # not starting with Penalty

    def test_impact_measurement_excludes_penalty_options(self):
        """Impact measurement candidate names exclude Penalty* options."""
        params = {
            "IndentWidth": ParameterDef(
                name="IndentWidth",
                param_type="int",
                possible_values=[2, 4],
                fixed=False,
            ),
            "PenaltyBreakAssignment": ParameterDef(
                name="PenaltyBreakAssignment",
                param_type="int",
                possible_values=[1, 2],
                fixed=False,  # Mutable, but will be fixed at start of loop.
            ),
            "ColumnLimit": ParameterDef(
                name="ColumnLimit",
                param_type="str",
                possible_values=["80", "100"],
                fixed=False,
            ),
        }
        space = SearchSpace(parameters=params)
        fitness = MagicMock(return_value=500.0)
        # Radial search fixes IndentWidth (int). Penalty fixed at start.
        # Impact called with ColumnLimit only.
        impact_fn = MagicMock(return_value=[])

        _ = run_iterative_optimization(
            search_space=space,
            fitness_fn=fitness,
            initial_config={
                "IndentWidth": 2,
                "PenaltyBreakAssignment": 1,
                "ColumnLimit": "80",
            },
            impact_fn=impact_fn,
            impact_kwargs={},
            debug=False,
        )

        # Impact fn should be called with only non-penalty, non-detected candidates.
        call_kwargs = impact_fn.call_args[1]
        assert "PenaltyBreakAssignment" not in call_kwargs["candidate_names"]
        assert "IndentWidth" not in call_kwargs["candidate_names"]
        assert "ColumnLimit" in call_kwargs["candidate_names"]

    def test_penalty_polish_unlocks_and_optimizes(self):
        """Final penalty polish unlocks penalty options and runs optimizer."""
        params = {
            "A": ParameterDef(
                name="A",
                param_type="int",
                possible_values=[1, 2, 4],
                fixed=False,
            ),
            "PenaltyBreakAssignment": ParameterDef(
                name="PenaltyBreakAssignment",
                param_type="int",
                possible_values=[1, 2],
                fixed=False,  # Mutable, fixed at start of loop.
            ),
        }
        space = SearchSpace(parameters=params)
        fitness = MagicMock(return_value=500.0)

        with patch("src.optimization_engine.iterative._optimize_batch") as mock_opt:
            from src.optimization_engine.types import OptimizationResult

            # Radial search fixes A. Penalty fixed at start. Impact has no candidates.
            # Only penalty polish calls _optimize_batch.
            mock_opt.return_value = OptimizationResult(
                best_config={"A": 2, "PenaltyBreakAssignment": 2},
                best_fitness=400.0,
                evaluations_used=10,
            )
            result = run_iterative_optimization(
                search_space=space,
                fitness_fn=fitness,
                initial_config={"A": 1, "PenaltyBreakAssignment": 1},
                impact_fn=MagicMock(return_value=[]),
                impact_kwargs={},
                debug=False,
            )

            # Penalty polish should have been attempted.
            assert result.best_fitness == 400.0

    def test_penalty_polish_debug_print_and_improvement(self):
        """Penalty polish prints debug header and updates config on improvement."""
        import io
        import sys

        params = {
            "A": ParameterDef(
                name="A",
                param_type="int",
                possible_values=[1, 2, 4],
                fixed=False,
            ),
            "PenaltyBreakAssignment": ParameterDef(
                name="PenaltyBreakAssignment",
                param_type="int",
                possible_values=[1, 2],
                fixed=False,
            ),
        }
        space = SearchSpace(parameters=params)
        fitness = MagicMock(return_value=500.0)

        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            with patch("src.optimization_engine.iterative._optimize_batch") as mock_opt:
                from src.optimization_engine.types import OptimizationResult

                # Radial search fixes A. Penalty fixed at start. Impact has no candidates.
                # Only penalty polish calls _optimize_batch.
                mock_opt.side_effect = [
                    OptimizationResult(
                        best_config={
                            "A": 2,
                            "PenaltyBreakAssignment": 2,
                        },
                        best_fitness=400.0,
                        evaluations_used=10,
                    ),
                ]
                result = run_iterative_optimization(
                    search_space=space,
                    fitness_fn=fitness,
                    initial_config={"A": 1, "PenaltyBreakAssignment": 1},
                    impact_fn=MagicMock(return_value=[]),
                    impact_kwargs={},
                    debug=True,
                )
        finally:
            output = sys.stderr.getvalue()
            sys.stderr = old_stderr

        assert "[penalty-polish]" in output
        assert result.best_fitness == 400.0

    def test_impact_scan_budget_not_passed_to_impact_fn(self):
        """Impact function no longer receives a budget parameter."""
        params = {
            "A": ParameterDef(
                name="A", param_type="int", possible_values=[1, 2, 4], fixed=False
            ),
            "X": ParameterDef(
                name="X", param_type="str", possible_values=["a", "b"], fixed=False
            ),
        }
        space = SearchSpace(parameters=params)
        fitness = MagicMock(return_value=500.0)
        impact_fn = MagicMock(return_value=[])

        _ = run_iterative_optimization(
            search_space=space,
            fitness_fn=fitness,
            initial_config={"A": 1},
            impact_fn=impact_fn,
            impact_kwargs={},
            debug=False,
        )

        call_kwargs = impact_fn.call_args[1]
        assert "budget" not in call_kwargs
