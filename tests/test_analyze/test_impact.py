"""Tests for empirical impact measurement via nevergrad."""

from typing import Any

from unittest.mock import patch

from src.analyze_conventions import DetectedOption, measure_impact
from src.analyze_conventions.impact import (
    _build_flat_options,  # pyright: ignore[reportPrivateUsage]
    _build_flat_options_from_config,  # pyright: ignore[reportPrivateUsage]
    _set_option_value,  # pyright: ignore[reportPrivateUsage]
    ImpactScore,
    measure_remaining_impact,
)
from src.data_classes import GeneticAlgorithmLookups
from src.optimization_engine.types import OptimizationResult


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
        "ColumnLimit": {"type": "int", "value": 80},
        "UseTab": {"type": "bool", "value": False},
        "BreakBeforeBraces": {"type": "str", "value": "Attach"},
        "AllowShortFunctionsOnASingleLine": {"type": "str", "value": "None"},
        "TabWidth": {"type": "int", "value": 4},
    }


class TestSetOptionValue:
    def test_sets_int(self):
        flat = {"IndentWidth": {"type": "int", "value": 4}}
        _set_option_value(flat, "IndentWidth", 8)
        assert flat["IndentWidth"]["value"] == 8

    def test_sets_bool(self):
        flat = {"UseTab": {"type": "bool", "value": False}}
        _set_option_value(flat, "UseTab", True)
        assert flat["UseTab"]["value"] is True

    def test_sets_str(self):
        flat = {"Language": {"type": "str", "value": "Cpp"}}
        _set_option_value(flat, "Language", "C")
        assert flat["Language"]["value"] == "C"

    def test_skips_missing_key(self):
        flat = {"IndentWidth": {"type": "int", "value": 4}}
        _set_option_value(flat, "NonExistent", 99)
        assert "NonExistent" not in flat

    def test_int_conversion_failure_keeps_original(self):
        flat = {"IndentWidth": {"type": "int", "value": 4}}
        _set_option_value(flat, "IndentWidth", "bad")
        assert flat["IndentWidth"]["value"] == 4


class TestBuildFlatOptions:
    def test_applies_analysis_values(self):
        base = _make_base_options()
        analysis = {"IndentWidth": 8, "UseTab": True}
        flat = _build_flat_options(base, analysis, {})
        assert flat["IndentWidth"]["value"] == 8
        assert flat["UseTab"]["value"] is True

    def test_applies_detected_option_values(self):
        base = _make_base_options()
        analysis = {
            "IndentWidth": DetectedOption(8, "detected"),
            "ColumnLimit": DetectedOption(100, "detected"),
        }
        flat = _build_flat_options(base, analysis, {})
        assert flat["IndentWidth"]["value"] == 8
        assert flat["ColumnLimit"]["value"] == 100

    def test_forced_options_override(self):
        base = _make_base_options()
        analysis = {"IndentWidth": 8}
        forced = {"UseTab": True}
        flat = _build_flat_options(base, analysis, forced)
        assert flat["IndentWidth"]["value"] == 8
        assert flat["UseTab"]["value"] is True


class TestMeasureImpact:
    @patch("src.analyze_conventions.impact.run_nevergrad_optimization")
    def test_returns_tiers(self, mock_ng):
        mock_ng.return_value = OptimizationResult(
            best_config={"IndentWidth": 8, "ColumnLimit": 80},
            best_fitness=10.0,
        )

        base = _make_base_options()
        analysis = {
            "IndentWidth": DetectedOption(4, "detected"),
            "ColumnLimit": DetectedOption(80, "detected"),
        }
        lookups = _make_lookups(
            json_options={
                "IndentWidth": {"possible_values": [2, 4, 8]},
                "ColumnLimit": {"possible_values": [80, 100, 120]},
            }
        )

        tiers = measure_impact("/tmp/repo", analysis, base, lookups, budget=10)
        # IndentWidth changed 4->8, so structure; ColumnLimit unchanged, so polish.
        assert tiers["IndentWidth"] == "structure"
        assert tiers["ColumnLimit"] == "polish"

    @patch("src.analyze_conventions.impact.run_nevergrad_optimization")
    def test_guessed_always_resolve(self, mock_ng):
        mock_ng.return_value = OptimizationResult(
            best_config={"IndentWidth": 8},
            best_fitness=10.0,
        )

        base = _make_base_options()
        analysis = {
            "IndentWidth": DetectedOption(4, "guessed", "resolve"),
        }
        lookups = _make_lookups(
            json_options={"IndentWidth": {"possible_values": [2, 4, 8]}}
        )

        tiers = measure_impact("/tmp/repo", analysis, base, lookups, budget=10)
        assert tiers["IndentWidth"] == "resolve"

    @patch("src.analyze_conventions.impact.run_nevergrad_optimization")
    def test_no_mutable_options_returns_polish(self, mock_ng):
        base = _make_base_options()
        analysis = {
            "IndentWidth": DetectedOption(4, "detected"),
        }
        # No possible_values in lookups, so only 1 value -> not mutable.
        lookups = _make_lookups(json_options={})

        tiers = measure_impact("/tmp/repo", analysis, base, lookups, budget=10)
        assert tiers["IndentWidth"] == "polish"
        # nevergrad should not be called.
        mock_ng.assert_not_called()

    @patch("src.analyze_conventions.impact.run_nevergrad_optimization")
    def test_raw_values_default_to_polish_tier(self, mock_ng):
        mock_ng.return_value = OptimizationResult(
            best_config={"IndentWidth": 4},
            best_fitness=10.0,
        )

        base = _make_base_options()
        analysis = {"IndentWidth": 4}  # raw value, no DetectedOption
        lookups = _make_lookups(
            json_options={"IndentWidth": {"possible_values": [2, 4, 8]}}
        )

        tiers = measure_impact("/tmp/repo", analysis, base, lookups, budget=10)
        assert tiers["IndentWidth"] == "polish"

    @patch("src.analyze_conventions.impact.run_nevergrad_optimization")
    def test_passes_budget_to_nevergrad(self, mock_ng):
        mock_ng.return_value = OptimizationResult(
            best_config={"IndentWidth": 8},
            best_fitness=10.0,
        )

        base = _make_base_options()
        analysis = {"IndentWidth": DetectedOption(4, "detected")}
        lookups = _make_lookups(
            json_options={"IndentWidth": {"possible_values": [2, 4, 8]}}
        )

        _ = measure_impact("/tmp/repo", analysis, base, lookups, budget=50)
        call_kwargs = mock_ng.call_args.kwargs
        assert call_kwargs["budget"] == 50

    @patch("src.analyze_conventions.impact.run_nevergrad_optimization")
    def test_unchanged_options_are_polish(self, mock_ng):
        mock_ng.return_value = OptimizationResult(
            best_config={"IndentWidth": 4, "ColumnLimit": 80},
            best_fitness=5.0,
        )

        base = _make_base_options()
        analysis = {
            "IndentWidth": DetectedOption(4, "detected"),
            "ColumnLimit": DetectedOption(80, "detected"),
        }
        lookups = _make_lookups(
            json_options={
                "IndentWidth": {"possible_values": [2, 4, 8]},
                "ColumnLimit": {"possible_values": [80, 100, 120]},
            }
        )

        tiers = measure_impact("/tmp/repo", analysis, base, lookups, budget=10)
        assert tiers["IndentWidth"] == "polish"
        assert tiers["ColumnLimit"] == "polish"

    @patch("src.analyze_conventions.impact.run_nevergrad_optimization")
    def test_changed_options_are_structure(self, mock_ng):
        mock_ng.return_value = OptimizationResult(
            best_config={"IndentWidth": 8, "ColumnLimit": 100},
            best_fitness=3.0,
        )

        base = _make_base_options()
        analysis = {
            "IndentWidth": DetectedOption(4, "detected"),
            "ColumnLimit": DetectedOption(80, "detected"),
        }
        lookups = _make_lookups(
            json_options={
                "IndentWidth": {"possible_values": [2, 4, 8]},
                "ColumnLimit": {"possible_values": [80, 100, 120]},
            }
        )

        tiers = measure_impact("/tmp/repo", analysis, base, lookups, budget=10)
        assert tiers["IndentWidth"] == "structure"
        assert tiers["ColumnLimit"] == "structure"

    @patch("src.analyze_conventions.impact.run_nevergrad_optimization")
    def test_skips_options_not_in_base_options(self, mock_ng):
        """Options in analysis_results but not in base_options are skipped."""
        mock_ng.return_value = OptimizationResult(
            best_config={},
            best_fitness=10.0,
        )

        base = _make_base_options()
        analysis = {
            "IndentWidth": DetectedOption(4, "detected"),
            "NonExistentOption": DetectedOption("foo", "detected"),
        }
        lookups = _make_lookups(
            json_options={"IndentWidth": {"possible_values": [2, 4, 8]}}
        )

        tiers = measure_impact("/tmp/repo", analysis, base, lookups, budget=10)
        # NonExistentOption is not mutable, so it defaults to polish.
        assert tiers["NonExistentOption"] == "polish"

    @patch("src.analyze_conventions.impact.run_nevergrad_optimization")
    def test_option_not_in_best_config_is_polish(self, mock_ng):
        """If an option is not in the best config, it defaults to polish."""
        mock_ng.return_value = OptimizationResult(
            best_config={"IndentWidth": 8},
            best_fitness=5.0,
        )

        base = _make_base_options()
        analysis = {
            "IndentWidth": DetectedOption(4, "detected"),
            "ColumnLimit": DetectedOption(80, "detected"),
        }
        lookups = _make_lookups(
            json_options={
                "IndentWidth": {"possible_values": [2, 4, 8]},
                "ColumnLimit": {"possible_values": [80, 100, 120]},
            }
        )

        tiers = measure_impact("/tmp/repo", analysis, base, lookups, budget=10)
        assert tiers["IndentWidth"] == "structure"
        # ColumnLimit is in parameters but not in best_config -> polish.
        assert tiers["ColumnLimit"] == "polish"


class TestBuildFlatOptionsFromConfig:
    def test_applies_config_values(self):
        base = _make_base_options()
        config = {"IndentWidth": 8, "UseTab": True}
        flat = _build_flat_options_from_config(base, config, {})
        assert flat["IndentWidth"]["value"] == 8
        assert flat["UseTab"]["value"] is True

    def test_forced_options_override(self):
        base = _make_base_options()
        config = {"IndentWidth": 8}
        forced = {"UseTab": True}
        flat = _build_flat_options_from_config(base, config, forced)
        assert flat["IndentWidth"]["value"] == 8
        assert flat["UseTab"]["value"] is True


class TestImpactScore:
    def test_construction(self):
        score = ImpactScore(name="IndentWidth", fitness_delta=50.0)
        assert score.name == "IndentWidth"
        assert score.fitness_delta == 50.0

    def test_repr(self):
        score = ImpactScore(name="PenaltyX", fitness_delta=12.5)
        assert "PenaltyX" in repr(score)


class TestMeasureRemainingImpact:
    @patch("src.analyze_conventions.impact.run_nevergrad_optimization")
    def test_returns_ranked_scores(self, mock_ng):
        mock_ng.return_value = OptimizationResult(
            best_config={"IndentWidth": 8, "ColumnLimit": 100},
            best_fitness=300.0,
        )

        base = _make_base_options()
        current_config = {"IndentWidth": 4, "ColumnLimit": 80}
        lookups = _make_lookups(
            json_options={
                "IndentWidth": {"possible_values": [2, 4, 8]},
                "ColumnLimit": {"possible_values": [80, 100, 120]},
            }
        )

        with patch("src.repo_formatter.run_clang_format_and_count_changes") as mock_cf:
            mock_cf.side_effect = [
                500,
                450,
                470,
            ]  # initial, solo IndentWidth, solo ColumnLimit
            scores = measure_remaining_impact(
                "/tmp/repo",
                ["IndentWidth", "ColumnLimit"],
                base,
                lookups,
                current_config,
                budget=10,
            )
        # Both options changed, so both should have scores.
        assert len(scores) == 2
        # Sorted by delta descending.
        assert scores[0].fitness_delta >= scores[1].fitness_delta

    @patch("src.analyze_conventions.impact.run_nevergrad_optimization")
    def test_unchanged_options_have_no_score(self, mock_ng):
        mock_ng.return_value = OptimizationResult(
            best_config={"IndentWidth": 4, "ColumnLimit": 80},
            best_fitness=500.0,
        )

        base = _make_base_options()
        current_config = {"IndentWidth": 4, "ColumnLimit": 80}
        lookups = _make_lookups(
            json_options={
                "IndentWidth": {"possible_values": [2, 4, 8]},
                "ColumnLimit": {"possible_values": [80, 100, 120]},
            }
        )

        with patch("src.repo_formatter.run_clang_format_and_count_changes") as mock_cf:
            mock_cf.return_value = 500
            scores = measure_remaining_impact(
                "/tmp/repo",
                ["IndentWidth", "ColumnLimit"],
                base,
                lookups,
                current_config,
                budget=10,
            )
        # Neither option changed, so no scores.
        assert scores == []

    def test_no_candidates_returns_empty(self):
        base = _make_base_options()
        lookups = _make_lookups(json_options={})

        scores = measure_remaining_impact(
            "/tmp/repo",
            ["IndentWidth"],
            base,
            lookups,
            {},
            budget=10,
        )
        # No possible_values, so empty.
        assert scores == []

    def test_skips_options_not_in_base(self):
        base = _make_base_options()
        lookups = _make_lookups(json_options={})

        scores = measure_remaining_impact(
            "/tmp/repo",
            ["NonExistentOption"],
            base,
            lookups,
            {},
            budget=10,
        )
        assert scores == []

    @patch("src.analyze_conventions.impact.run_nevergrad_optimization")
    def test_uses_base_options_value_when_not_in_config(self, mock_ng):
        mock_ng.return_value = OptimizationResult(
            best_config={"IndentWidth": 8},
            best_fitness=400.0,
        )

        base = _make_base_options()
        # IndentWidth not in current_config, should fall back to base_options.
        current_config: dict[str, Any] = {}
        lookups = _make_lookups(
            json_options={"IndentWidth": {"possible_values": [2, 4, 8]}}
        )

        with patch("src.repo_formatter.run_clang_format_and_count_changes") as mock_cf:
            mock_cf.side_effect = [500, 450]  # initial, solo
            scores = measure_remaining_impact(
                "/tmp/repo",
                ["IndentWidth"],
                base,
                lookups,
                current_config,
                budget=10,
            )
        assert len(scores) == 1
        assert scores[0].name == "IndentWidth"

    @patch("src.analyze_conventions.impact.run_nevergrad_optimization")
    def test_passes_budget_to_nevergrad(self, mock_ng):
        mock_ng.return_value = OptimizationResult(
            best_config={"IndentWidth": 8},
            best_fitness=400.0,
        )

        base = _make_base_options()
        lookups = _make_lookups(
            json_options={"IndentWidth": {"possible_values": [2, 4, 8]}}
        )

        with patch("src.repo_formatter.run_clang_format_and_count_changes"):
            _ = measure_remaining_impact(
                "/tmp/repo",
                ["IndentWidth"],
                base,
                lookups,
                {"IndentWidth": 4},
                budget=200,
            )
        call_kwargs = mock_ng.call_args.kwargs
        assert call_kwargs["budget"] == 200

    @patch("src.analyze_conventions.impact.run_nevergrad_optimization")
    def test_penalty_options_get_curated_values(self, mock_ng):
        """Penalty options use curated values when not in json lookup."""
        mock_ng.return_value = OptimizationResult(
            best_config={"PenaltyExcessCharacter": 10},
            best_fitness=400.0,
        )

        base = {
            "PenaltyExcessCharacter": {"type": "int", "value": 20},
        }
        # No json lookup entry for this penalty option.
        lookups = _make_lookups(json_options={})

        with patch("src.repo_formatter.run_clang_format_and_count_changes"):
            scores = measure_remaining_impact(
                "/tmp/repo",
                ["PenaltyExcessCharacter"],
                base,
                lookups,
                {"PenaltyExcessCharacter": 20},
                budget=10,
            )
        # Should have been unlocked with curated values.
        assert len(scores) == 1

    @patch("src.analyze_conventions.impact.run_nevergrad_optimization")
    def test_skips_single_value_options(self, mock_ng):
        """Options with only one possible value are skipped."""
        base = _make_base_options()
        lookups = _make_lookups(json_options={"IndentWidth": {"possible_values": [4]}})

        scores = measure_remaining_impact(
            "/tmp/repo",
            ["IndentWidth"],
            base,
            lookups,
            {"IndentWidth": 4},
            budget=10,
        )
        # Only one value, so nothing to optimize.
        assert scores == []
        mock_ng.assert_not_called()
