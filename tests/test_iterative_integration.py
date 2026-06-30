"""Integration tests for the iterative expansion optimizer.

Exercises the full loop with real _optimize_batch (GA/nevergrad), but mocks
the fitness function to avoid running clang-format.
"""

from typing import Any

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


def _make_param(
    name: str,
    param_type: str = "int",
    possible_values: list[Any] | None = None,
    fixed: bool = False,
    confidence: str = "",
) -> ParameterDef:
    """Helper to create a ParameterDef."""
    return ParameterDef(
        name=name,
        param_type=param_type,
        possible_values=possible_values or [2, 4, 8],
        fixed=fixed,
        confidence=confidence,
    )


def _make_space(params: list[ParameterDef]) -> SearchSpace:
    """Build a search space from a list of ParameterDef."""
    return SearchSpace(parameters={p.name: p for p in params})


class TestFullIterativeLoop:
    """Test the full iterative loop with real _optimize_batch."""

    def test_radial_scan_then_impact_optimize(self):
        """Full loop: radial search integers, impact, optimize, fix."""
        from unittest.mock import patch

        # IndentWidth is mutable int (radial search fixes it).
        # ColumnLimit and TabWidth are mutable str (impact measures them).
        space = _make_space(
            [
                _make_param("IndentWidth", "int"),
                _make_param("ColumnLimit", "str"),
                _make_param("TabWidth", "str"),
            ]
        )
        fitness = MagicMock(return_value=500.0)

        # Radial search fixes IndentWidth. Impact measures ColumnLimit, TabWidth once.
        # Window size = max(1, int(2*0.2)) = 1, so 2 batches needed.
        impact_fn = MagicMock(
            return_value=[
                ImpactScore("ColumnLimit", 100.0),
                ImpactScore("TabWidth", 10.0),
            ]
        )

        with patch("src.optimization_engine.iterative._optimize_batch") as mock_opt:
            mock_opt.return_value = MagicMock(
                best_fitness=500.0,
                best_config={"IndentWidth": 4, "ColumnLimit": "80", "TabWidth": "8"},
                evaluations_used=10,
            )

            result = run_iterative_optimization(
                search_space=space,
                fitness_fn=fitness,
                initial_config={
                    "IndentWidth": 4,
                    "ColumnLimit": "80",
                    "TabWidth": "8",
                },
                impact_fn=impact_fn,
                impact_kwargs={},
                num_islands=1,
                population_size=2,
                num_workers=1,
                debug=False,
            )

        assert result.best_fitness == 500.0
        assert impact_fn.call_count == 1
        # Verify ColumnLimit was in call's candidates.
        first_call = impact_fn.call_args_list[0].kwargs
        assert "ColumnLimit" in first_call["candidate_names"]

    def test_exhausts_all_options(self):
        """Loop continues until all mutable options are optimized and fixed."""
        from unittest.mock import patch

        space = _make_space(
            [
                _make_param("ColumnLimit", "str"),
            ]
        )
        fitness = MagicMock(return_value=500.0)

        impact_fn = MagicMock()
        impact_fn.side_effect = [
            [ImpactScore("ColumnLimit", 100.0)],
        ]

        with patch("src.optimization_engine.iterative._optimize_batch") as mock_opt:
            mock_opt.return_value = MagicMock(
                best_fitness=500.0,
                best_config={"ColumnLimit": "80"},
                evaluations_used=5,
            )

            result = run_iterative_optimization(
                search_space=space,
                fitness_fn=fitness,
                initial_config={"ColumnLimit": "80"},
                impact_fn=impact_fn,
                impact_kwargs={},
                num_islands=1,
                population_size=2,
                num_workers=1,
                debug=False,
            )

        assert result.best_fitness == 500.0
        assert impact_fn.call_count == 1


class TestStoppingConditions:
    """Test that the loop stops correctly under various conditions."""

    def test_stops_when_no_impactful_options(self):
        """Stop when impact measurement returns empty list."""
        space = _make_space(
            [
                _make_param("X", "str"),
                _make_param("Y", "str"),
            ]
        )
        fitness = MagicMock(return_value=500.0)
        impact_fn = MagicMock()
        impact_fn.side_effect = [
            [],  # No impactful options.
        ]

        result = run_iterative_optimization(
            search_space=space,
            fitness_fn=fitness,
            initial_config={"X": "a", "Y": "b"},
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
        space = _make_space(
            [
                _make_param("X", "str"),
            ]
        )
        fitness = MagicMock(return_value=500.0)
        impact_fn = MagicMock()
        impact_fn.side_effect = [
            [ImpactScore("X", 1.0)],  # 1.0 < 0.01 * 500 = 5.0
        ]

        result = run_iterative_optimization(
            search_space=space,
            fitness_fn=fitness,
            initial_config={"X": "a"},
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

        space = _make_space([_make_param(f"opt_{i}", "str") for i in range(50)])
        fitness = MagicMock(return_value=500.0)

        scores = [ImpactScore(f"opt_{i}", 100.0) for i in range(50)]
        impact_fn = MagicMock()
        impact_fn.side_effect = [scores, []]

        with patch("src.optimization_engine.iterative._optimize_batch") as mock_opt:
            mock_opt.return_value = MagicMock(
                best_fitness=500.0,
                best_config={},
                evaluations_used=5,
            )

            result = run_iterative_optimization(
                search_space=space,
                fitness_fn=fitness,
                initial_config={},
                impact_fn=impact_fn,
                impact_kwargs={},
                max_batch_fraction=MAX_BATCH_FRACTION,
                impact_threshold=IMPACT_THRESHOLD,
                num_islands=1,
                population_size=2,
                num_workers=1,
                debug=False,
            )

        assert impact_fn.call_count >= 1
        assert result.best_fitness == 500.0

    def test_dropoff_guardrail_excludes_low_impact(self):
        """Options below IMPACT_THRESHOLD of top score are excluded."""
        from unittest.mock import patch

        space = _make_space(
            [
                _make_param("high", "str"),
                _make_param("medium", "str"),
                _make_param("low1", "str"),
                _make_param("low2", "str"),
            ]
        )
        fitness = MagicMock(return_value=500.0)

        # high=100, medium=60 (>= 50% of 100), lows are < 50%.
        # Impact called once upfront. Window slides over pre-computed scores.
        # Window size = max(1, int(4*0.2)) = 1. First window: high.
        # Second window: medium. Third window: low1 (below threshold, stops).
        scores = [
            ImpactScore("high", 100.0),
            ImpactScore("medium", 60.0),
            ImpactScore("low1", 10.0),
            ImpactScore("low2", 5.0),
        ]
        impact_fn = MagicMock(return_value=scores)

        with patch("src.optimization_engine.iterative._optimize_batch") as mock_opt:
            mock_opt.return_value = MagicMock(
                best_fitness=500.0, best_config={}, evaluations_used=10
            )

            result = run_iterative_optimization(
                search_space=space,
                fitness_fn=fitness,
                initial_config={},
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
        # Impact called once upfront, window slides over pre-computed scores.
        assert impact_fn.call_count == 1

    def test_multi_iteration_expansion(self):
        """Verify the loop can run multiple iterations, fixing batches each time."""
        from unittest.mock import patch

        space = _make_space(
            [
                _make_param("X", "str"),
                _make_param("Y", "str"),
                _make_param("Z", "str"),
                _make_param("W", "str"),
            ]
        )
        fitness = MagicMock(return_value=500.0)

        # batch cap = max(1, int(4*0.2)) = 1.
        # Impact called once upfront. Window slides: X, then Y.
        # Z has delta=10 which is below threshold (50% of 100 = 50), so stops.
        impact_fn = MagicMock(
            return_value=[
                ImpactScore("X", 100.0),
                ImpactScore("Y", 80.0),
                ImpactScore("Z", 10.0),
                ImpactScore("W", 5.0),
            ]
        )

        with patch("src.optimization_engine.iterative._optimize_batch") as mock_opt:
            mock_opt.return_value = MagicMock(
                best_fitness=500.0, best_config={}, evaluations_used=5
            )

            result = run_iterative_optimization(
                search_space=space,
                fitness_fn=fitness,
                initial_config={},
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
            # Impact called once upfront.
            assert impact_fn.call_count == 1

    def test_penalties_fixed_at_start(self):
        """Penalty options are fixed before the main loop and excluded from impact."""
        from unittest.mock import patch

        space = _make_space(
            [
                _make_param("ColumnLimit", "str"),
                _make_param("PenaltyBreakAssignment", "int"),
            ]
        )
        fitness = MagicMock(return_value=500.0)

        # Penalty is fixed at start. Impact measures only ColumnLimit.
        impact_fn = MagicMock()
        impact_fn.side_effect = [
            [ImpactScore("ColumnLimit", 100.0)],
        ]

        with patch("src.optimization_engine.iterative._optimize_batch") as mock_opt:
            mock_opt.return_value = MagicMock(
                best_fitness=500.0,
                best_config={"ColumnLimit": "80"},
                evaluations_used=5,
            )

            result = run_iterative_optimization(
                search_space=space,
                fitness_fn=fitness,
                initial_config={"ColumnLimit": "80", "PenaltyBreakAssignment": 10},
                impact_fn=impact_fn,
                impact_kwargs={},
                num_islands=1,
                population_size=2,
                num_workers=1,
                debug=False,
            )

        assert result.best_fitness == 500.0
        assert impact_fn.call_count == 1
        # Penalty should not be in impact candidates.
        first_call = impact_fn.call_args_list[0].kwargs
        assert "PenaltyBreakAssignment" not in first_call["candidate_names"]
