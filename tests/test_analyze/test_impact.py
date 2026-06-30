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

        tiers = measure_impact("/tmp/repo", analysis, base, lookups)
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

        tiers = measure_impact("/tmp/repo", analysis, base, lookups)
        assert tiers["IndentWidth"] == "resolve"

    @patch("src.analyze_conventions.impact.run_nevergrad_optimization")
    def test_no_mutable_options_returns_polish(self, mock_ng):
        base = _make_base_options()
        analysis = {
            "IndentWidth": DetectedOption(4, "detected"),
        }
        # No possible_values in lookups, so only 1 value -> not mutable.
        lookups = _make_lookups(json_options={})

        tiers = measure_impact("/tmp/repo", analysis, base, lookups)
        assert tiers["IndentWidth"] == "polish"
        # nevergrad should not be called.
        mock_ng.assert_not_called()

    @patch("src.analyze_conventions.impact.run_nevergrad_optimization")
    def test_no_mutable_options_preserves_analyzer_tiers(self, mock_ng):
        """When no options have possible_values, preserve analyzer tiers."""
        base = _make_base_options()
        analysis = {
            "IndentWidth": DetectedOption(4, "guessed", "resolve"),
            "ColumnLimit": DetectedOption(72, "detected", "structure"),
            "ReflowComments": DetectedOption("Never", "detected", "polish"),
        }
        lookups = _make_lookups(json_options={})

        tiers = measure_impact("/tmp/repo", analysis, base, lookups)
        assert tiers["IndentWidth"] == "resolve"
        assert tiers["ColumnLimit"] == "structure"
        assert tiers["ReflowComments"] == "polish"
        mock_ng.assert_not_called()

    @patch("src.analyze_conventions.impact.run_nevergrad_optimization")
    def test_no_mutable_options_raw_value_defaults_polish(self, mock_ng):
        """Raw values without tier attribute default to polish."""
        base = _make_base_options()
        analysis = {
            "IndentWidth": 4,  # raw value, no DetectedOption
        }
        lookups = _make_lookups(json_options={})

        tiers = measure_impact("/tmp/repo", analysis, base, lookups)
        assert tiers["IndentWidth"] == "polish"
        mock_ng.assert_not_called()

    @patch("src.analyze_conventions.impact.run_nevergrad_optimization")
    def test_single_possible_value_skipped(self, mock_ng):
        """Options with only 1 possible_values in JSON are skipped."""
        base = _make_base_options()
        analysis = {
            "IndentWidth": DetectedOption(4, "detected"),
        }
        # Only 1 possible value, so it gets skipped.
        lookups = _make_lookups(json_options={"IndentWidth": {"possible_values": [4]}})

        tiers = measure_impact("/tmp/repo", analysis, base, lookups)
        # Falls through to preserve_analyzer_tiers because parameters is empty.
        assert tiers["IndentWidth"] == "polish"
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

        tiers = measure_impact("/tmp/repo", analysis, base, lookups)
        assert tiers["IndentWidth"] == "polish"

    @patch("src.analyze_conventions.impact.run_nevergrad_optimization")
    def test_uses_safety_budget(self, mock_ng):
        mock_ng.return_value = OptimizationResult(
            best_config={"IndentWidth": 8},
            best_fitness=10.0,
        )

        base = _make_base_options()
        analysis = {"IndentWidth": DetectedOption(4, "detected")}
        lookups = _make_lookups(
            json_options={"IndentWidth": {"possible_values": [2, 4, 8]}}
        )

        _ = measure_impact("/tmp/repo", analysis, base, lookups)
        call_kwargs = mock_ng.call_args.kwargs
        assert call_kwargs["budget"] == 10_000

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

        tiers = measure_impact("/tmp/repo", analysis, base, lookups)
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

        tiers = measure_impact("/tmp/repo", analysis, base, lookups)
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

        tiers = measure_impact("/tmp/repo", analysis, base, lookups)
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

        tiers = measure_impact("/tmp/repo", analysis, base, lookups)
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
    def test_returns_ranked_scores(self):
        """Exhaustive scan tries all values per option and ranks by best delta."""
        base = _make_base_options()
        current_config = {"IndentWidth": 4, "ColumnLimit": 80}
        lookups = _make_lookups(
            json_options={
                "IndentWidth": {"possible_values": [2, 4, 8]},
                "ColumnLimit": {"possible_values": [80, 100, 120]},
            }
        )

        # baseline=500, then IndentWidth values (2,4,8), then ColumnLimit values (80,100,120)
        with patch("src.repo_formatter.run_clang_format_and_count_changes") as mock_cf:
            mock_cf.side_effect = [
                500,  # baseline
                450,  # IndentWidth=2 (delta=50)
                500,  # IndentWidth=4 (delta=0)
                480,  # IndentWidth=8 (delta=20)
                500,  # ColumnLimit=80 (delta=0)
                460,  # ColumnLimit=100 (delta=40)
                490,  # ColumnLimit=120 (delta=10)
            ]
            scores = measure_remaining_impact(
                "/tmp/repo",
                ["IndentWidth", "ColumnLimit"],
                base,
                lookups,
                current_config,
            )
        # Both options improved, so both should have scores.
        assert len(scores) == 2
        # Sorted by delta descending: IndentWidth(50) > ColumnLimit(40)
        assert scores[0].name == "IndentWidth"
        assert scores[0].fitness_delta == 50
        assert scores[1].name == "ColumnLimit"
        assert scores[1].fitness_delta == 40

    def test_unchanged_options_have_no_score(self):
        """If no value improves over baseline, option gets no score."""
        base = _make_base_options()
        current_config = {"IndentWidth": 4, "ColumnLimit": 80}
        lookups = _make_lookups(
            json_options={
                "IndentWidth": {"possible_values": [2, 4, 8]},
                "ColumnLimit": {"possible_values": [80, 100, 120]},
            }
        )

        # baseline=500, all other values are worse or equal
        with patch("src.repo_formatter.run_clang_format_and_count_changes") as mock_cf:
            mock_cf.return_value = 500
            scores = measure_remaining_impact(
                "/tmp/repo",
                ["IndentWidth", "ColumnLimit"],
                base,
                lookups,
                current_config,
            )
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
        )
        assert scores == []

    def test_uses_base_options_value_when_not_in_config(self):
        """When option is missing from config, falls back to base_options value."""
        base = _make_base_options()
        # IndentWidth not in current_config, should fall back to base_options.
        current_config: dict[str, Any] = {}
        lookups = _make_lookups(
            json_options={"IndentWidth": {"possible_values": [2, 4, 8]}}
        )

        # baseline=500, then IndentWidth=2,4,8
        with patch("src.repo_formatter.run_clang_format_and_count_changes") as mock_cf:
            mock_cf.side_effect = [
                500,  # baseline
                450,  # IndentWidth=2 (delta=50)
                500,  # IndentWidth=4 (delta=0)
                480,  # IndentWidth=8 (delta=20)
            ]
            scores = measure_remaining_impact(
                "/tmp/repo",
                ["IndentWidth"],
                base,
                lookups,
                current_config,
            )
        assert len(scores) == 1
        assert scores[0].name == "IndentWidth"
        assert scores[0].fitness_delta == 50

    def test_penalty_options_get_curated_values(self):
        """Penalty options use curated values when not in json lookup."""
        base = {
            "PenaltyExcessCharacter": {"type": "int", "value": 20},
        }
        # No json lookup entry for this penalty option.
        lookups = _make_lookups(json_options={})

        # baseline=500, then curated penalty values (2,10,50,150,1000,10000)
        with patch("src.repo_formatter.run_clang_format_and_count_changes") as mock_cf:
            mock_cf.side_effect = [
                500,  # baseline
                450,  # PenaltyExcessCharacter=2 (delta=50)
                480,  # =10
                490,  # =50
                495,  # =150
                498,  # =1000
                499,  # =10000
            ]
            scores = measure_remaining_impact(
                "/tmp/repo",
                ["PenaltyExcessCharacter"],
                base,
                lookups,
                {"PenaltyExcessCharacter": 20},
            )
        # Should have been tested with curated values.
        assert len(scores) == 1
        assert scores[0].fitness_delta == 50

    def test_skips_single_value_options(self):
        """Options with only one possible value are skipped."""
        base = _make_base_options()
        lookups = _make_lookups(json_options={"IndentWidth": {"possible_values": [4]}})

        scores = measure_remaining_impact(
            "/tmp/repo",
            ["IndentWidth"],
            base,
            lookups,
            {"IndentWidth": 4},
        )
        # Only one value, so nothing to optimize.
        assert scores == []

    def test_baseline_error_returns_empty(self):
        """If baseline clang-format run fails, return empty scores."""
        base = _make_base_options()
        lookups = _make_lookups(
            json_options={"IndentWidth": {"possible_values": [2, 4, 8]}}
        )

        with patch("src.repo_formatter.run_clang_format_and_count_changes") as mock_cf:
            mock_cf.return_value = -1
            scores = measure_remaining_impact(
                "/tmp/repo",
                ["IndentWidth"],
                base,
                lookups,
                {"IndentWidth": 4},
            )
        assert scores == []

    def test_skips_option_not_in_initial_flat(self):
        """If option name is not in initial_flat, it is skipped."""
        base = _make_base_options()
        lookups = _make_lookups(
            json_options={"IndentWidth": {"possible_values": [2, 4, 8]}}
        )

        with patch("src.repo_formatter.run_clang_format_and_count_changes") as mock_cf:
            mock_cf.return_value = 500
            scores = measure_remaining_impact(
                "/tmp/repo",
                ["IndentWidth"],
                base,
                lookups,
                {},  # IndentWidth not in config, so not in initial_flat
            )
        assert scores == []

    def test_forced_options_override_in_loop(self):
        """Forced options are applied during each value test."""
        base = _make_base_options()
        lookups = _make_lookups(
            json_options={"IndentWidth": {"possible_values": [2, 4, 8]}},
            forced_options={"UseTab": True},
        )

        with patch("src.repo_formatter.run_clang_format_and_count_changes") as mock_cf:
            mock_cf.side_effect = [
                500,  # baseline
                450,  # IndentWidth=2
                500,  # IndentWidth=4
                480,  # IndentWidth=8
            ]
            scores = measure_remaining_impact(
                "/tmp/repo",
                ["IndentWidth"],
                base,
                lookups,
                {"IndentWidth": 4},
            )
        assert len(scores) == 1
        assert scores[0].fitness_delta == 50

    def test_value_error_continues(self):
        """If a value test returns -1, it is skipped and next value is tried."""
        base = _make_base_options()
        lookups = _make_lookups(
            json_options={"IndentWidth": {"possible_values": [2, 4, 8]}}
        )

        with patch("src.repo_formatter.run_clang_format_and_count_changes") as mock_cf:
            mock_cf.side_effect = [
                500,  # baseline
                -1,  # IndentWidth=2 (error, skip)
                500,  # IndentWidth=4 (delta=0)
                480,  # IndentWidth=8 (delta=20)
            ]
            scores = measure_remaining_impact(
                "/tmp/repo",
                ["IndentWidth"],
                base,
                lookups,
                {"IndentWidth": 4},
            )
        assert len(scores) == 1
        assert scores[0].fitness_delta == 20
