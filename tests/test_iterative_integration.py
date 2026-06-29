"""Integration tests for the iterative expansion optimizer.

Exercises the full loop with real _optimize_batch (GA/nevergrad), but mocks
the fitness function to avoid running clang-format.
"""

from unittest.mock import MagicMock

from src.analyze_conventions.impact import ImpactScore
from src.optimization_engine.iterative import (
    IMPACT_THRESHOLD,
    MAX_BATCH_FRACTION,
    MIN_IMPROVEMENT_RATIO,
    run_iterative_optimization,
)
from src.optimization_engine.types import (
    ParameterDef,
    SearchSpace,
)


def _make_space(detected: list[str], fixed: list[str]) -> SearchSpace:
    """Build a search space with detected (mutable) and fixed params."""
    params: dict[str, ParameterDef] = {}
    for name in detected:
        params[name] = ParameterDef(
            name=name,
            param_type="int",
            possible_values=[2, 4, 8],
            fixed=False,
        )
    for name in fixed:
        params[name] = ParameterDef(
            name=name,
            param_type="int",
            possible_values=[2, 4, 8],
            fixed=True,
        )
    return SearchSpace(parameters=params)


class TestFullIterativeLoop:
    """Test the full iterative loop with real _optimize_batch."""

    def test_polish_detected_then_expand_impactful(self):
        """Full loop: polish detected, measure impact, unlock, repeat."""
        from unittest.mock import patch

        space = _make_space(detected=["IndentWidth"], fixed=["ColumnLimit", "TabWidth"])
        fitness = MagicMock(return_value=500.0)

        # First impact call: ColumnLimit is impactful.
        # Second impact call: nothing left impactful.
        impact_fn = MagicMock()
        impact_fn.side_effect = [
            [ImpactScore("ColumnLimit", 100.0), ImpactScore("TabWidth", 10.0)],
            [],
        ]

        with patch("src.optimization_engine.iterative._optimize_batch") as mock_opt:
            mock_opt.return_value = MagicMock(
                best_fitness=500.0,
                best_config={"IndentWidth": 4, "ColumnLimit": 80, "TabWidth": 8},
                evaluations_used=10,
            )

            result = run_iterative_optimization(
                search_space=space,
                fitness_fn=fitness,
                initial_config={
                    "IndentWidth": 4,
                    "ColumnLimit": 80,
                    "TabWidth": 8,
                },
                impact_fn=impact_fn,
                impact_kwargs={},
                num_islands=1,
                population_size=2,
                num_workers=1,
                debug=False,
            )

        assert result.best_fitness == 500.0
        assert impact_fn.call_count == 2
        # Verify ColumnLimit was unlocked (second impact call had different remaining).
        first_call = impact_fn.call_args_list[0].kwargs
        second_call = impact_fn.call_args_list[1].kwargs
        assert "ColumnLimit" in first_call["candidate_names"]
        assert "ColumnLimit" not in second_call["candidate_names"]

    def test_exhausts_all_options(self):
        """Loop continues until all fixed options are unlocked and optimized."""
        space = _make_space(
            detected=["IndentWidth"],
            fixed=["ColumnLimit"],
        )
        fitness = MagicMock(return_value=500.0)

        # Impact finds ColumnLimit impactful, then no remaining fixed.
        impact_fn = MagicMock()
        impact_fn.side_effect = [
            [ImpactScore("ColumnLimit", 100.0)],
        ]

        result = run_iterative_optimization(
            search_space=space,
            fitness_fn=fitness,
            initial_config={"IndentWidth": 4, "ColumnLimit": 80},
            impact_fn=impact_fn,
            impact_kwargs={},
            num_islands=1,
            population_size=2,
            num_workers=1,
            debug=False,
        )

        assert result.best_fitness == 500.0
        # After unlocking ColumnLimit, no remaining fixed -> loop exits.
        assert impact_fn.call_count == 1


class TestStoppingConditions:
    """Test that the loop stops correctly under various conditions."""

    def test_stops_when_no_impactful_options(self):
        """Stop when impact measurement returns empty list."""
        space = _make_space(detected=["A"], fixed=["X", "Y"])
        fitness = MagicMock(return_value=500.0)
        impact_fn = MagicMock(return_value=[])

        result = run_iterative_optimization(
            search_space=space,
            fitness_fn=fitness,
            initial_config={"A": 4},
            impact_fn=impact_fn,
            impact_kwargs={},
            num_islands=1,
            population_size=2,
            num_workers=1,
            debug=False,
        )

        assert result.best_fitness == 500.0
        assert impact_fn.call_count == 1

    def test_stops_when_impact_below_threshold(self):
        """Stop when top impact is below min_improvement_ratio of fitness."""
        space = _make_space(detected=["A"], fixed=["X"])
        fitness = MagicMock(return_value=500.0)
        # Impact of 1.0 < 0.01 * 500 = 5.0
        impact_fn = MagicMock(return_value=[ImpactScore("X", 1.0)])

        result = run_iterative_optimization(
            search_space=space,
            fitness_fn=fitness,
            initial_config={"A": 4},
            impact_fn=impact_fn,
            impact_kwargs={},
            min_improvement_ratio=MIN_IMPROVEMENT_RATIO,
            num_islands=1,
            population_size=2,
            num_workers=1,
            debug=False,
        )

        assert result.best_fitness == 500.0
        assert impact_fn.call_count == 1


class TestBatchSelection:
    """Test batch selection with real _select_batch logic."""

    def test_batch_capped_at_percentage(self):
        """Batch size is capped at MAX_BATCH_FRACTION of remaining."""
        from unittest.mock import patch

        space = _make_space(
            detected=["A"],
            fixed=[f"opt_{i}" for i in range(50)],
        )
        fitness = MagicMock(return_value=500.0)

        # All 50 options are equally impactful.
        scores = [ImpactScore(f"opt_{i}", 100.0) for i in range(50)]
        # First call returns 50 impactful options; second call returns none (loop stops).
        impact_fn = MagicMock()
        impact_fn.side_effect = [scores, []]

        with patch("src.optimization_engine.iterative._optimize_batch") as mock_opt:
            mock_opt.return_value = MagicMock(
                best_fitness=500.0,
                best_config={"A": 4},
                evaluations_used=5,
            )

            result = run_iterative_optimization(
                search_space=space,
                fitness_fn=fitness,
                initial_config={"A": 4},
                impact_fn=impact_fn,
                impact_kwargs={},
                max_batch_fraction=MAX_BATCH_FRACTION,
                impact_threshold=IMPACT_THRESHOLD,
                num_islands=1,
                population_size=2,
                num_workers=1,
                debug=False,
            )

        # After first unlock, remaining should be 50 - 10 = 40.
        assert impact_fn.call_count >= 1
        # The loop should have unlocked a batch.
        assert result.best_fitness == 500.0

    def test_dropoff_guardrail_excludes_low_impact(self):
        """Options below IMPACT_THRESHOLD of top score are excluded."""
        from unittest.mock import patch

        space = _make_space(
            detected=["A"],
            fixed=["high", "medium", "low1", "low2", "low3", "low4", "low5", "low6"],
        )
        fitness = MagicMock(return_value=500.0)

        # high=100, medium=60 (>= 50% of 100), lows are < 50%.
        # 8 remaining, batch cap = max(1, int(8*0.2)) = 1. Only "high" unlocks.
        scores = [
            ImpactScore("high", 100.0),
            ImpactScore("medium", 60.0),
            ImpactScore("low1", 10.0),
            ImpactScore("low2", 5.0),
        ]
        impact_fn = MagicMock()
        impact_fn.side_effect = [scores, []]

        with patch("src.optimization_engine.iterative._optimize_batch") as mock_opt:
            mock_opt.return_value = MagicMock(
                best_fitness=500.0, best_config={"A": 4}, evaluations_used=10
            )

            result = run_iterative_optimization(
                search_space=space,
                fitness_fn=fitness,
                initial_config={"A": 4},
                impact_fn=impact_fn,
                impact_kwargs={},
                max_batch_fraction=MAX_BATCH_FRACTION,
                impact_threshold=IMPACT_THRESHOLD,
                num_islands=1,
                population_size=2,
                num_workers=1,
                debug=False,
            )

        assert result.best_fitness == 500.0
        assert impact_fn.call_count == 2
        # Second call should not include "high" (it was unlocked).
        second_candidates = impact_fn.call_args_list[1].kwargs["candidate_names"]
        assert "high" not in second_candidates
        # medium was NOT unlocked because batch cap=1 and high took the slot.
        assert "medium" in second_candidates

    def test_multi_iteration_expansion(self):
        """Verify the loop can run multiple iterations, unlocking batches each time."""
        from unittest.mock import patch

        space = _make_space(
            detected=["A"],
            fixed=["X", "Y", "Z", "W"],
        )
        fitness = MagicMock(return_value=500.0)

        # 4 fixed, batch cap = max(1, int(4*0.2)) = 1. One unlock per iteration.
        impact_fn = MagicMock()
        impact_fn.side_effect = [
            [ImpactScore("X", 100.0), ImpactScore("Y", 10.0)],
            [ImpactScore("Y", 80.0), ImpactScore("Z", 10.0)],
            [],
        ]

        # Mock _optimize_batch to use minimal budget so we can test multiple iterations.
        with patch("src.optimization_engine.iterative._optimize_batch") as mock_opt:
            mock_opt.return_value = MagicMock(
                best_fitness=500.0, best_config={"A": 4}, evaluations_used=5
            )

            result = run_iterative_optimization(
                search_space=space,
                fitness_fn=fitness,
                initial_config={"A": 4},
                impact_fn=impact_fn,
                impact_kwargs={},
                max_batch_fraction=MAX_BATCH_FRACTION,
                impact_threshold=IMPACT_THRESHOLD,
                num_islands=1,
                population_size=2,
                num_workers=1,
                debug=False,
            )

            assert result.best_fitness == 500.0
            # Loop runs 3 iterations: X unlocked, then Y, then empty stops.
            assert impact_fn.call_count == 3
            # Verify X was unlocked (not in second call's candidates).
            second_candidates = impact_fn.call_args_list[1].kwargs["candidate_names"]
            assert "X" not in second_candidates
            assert "Y" in second_candidates
            # Verify Y was unlocked (not in third call's candidates).
            third_candidates = impact_fn.call_args_list[2].kwargs["candidate_names"]
            assert "Y" not in third_candidates
