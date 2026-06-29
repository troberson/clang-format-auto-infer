"""Empirical impact measurement via lightweight nevergrad run.

Instead of flipping each option individually, run a small nevergrad
optimization over a set of candidate options. Options that the optimizer
changes to improve fitness are high-impact. Options that never move are
low-impact.

Two entry points:
- measure_impact: classifies detected options into tiers (resolve/structure/polish).
- measure_remaining_impact: scores remaining fixed options and returns a ranked list.
"""

from __future__ import annotations

import copy
from typing import Any, override

from ..clang_format_parser import generate_clang_format_config
from ..data_classes import GeneticAlgorithmLookups
from ..optimization_engine.types import ParameterDef, SearchSpace
from ..optimization_engine.nevergrad import run_nevergrad_optimization
from ..optimization_engine.iterative import NG_SAFETY_BUDGET

# Curated penalty values shared with clang_format_adapter.
CURATED_PENALTY_VALUES: list[int] = [2, 10, 50, 150, 1000, 10000]


def _preserve_analyzer_tiers(
    analysis_results: dict[str, Any],
) -> dict[str, str]:
    """Return tier assignments from the analyzer when impact cannot be measured.

    Used as a fallback when no options have possible_values, so the impact
    scan cannot run. Preserves the analyzer's resolve/structure/polish tiers
    instead of overriding everything to 'polish'.
    """
    tiers: dict[str, str] = {}
    for name, raw in analysis_results.items():
        if hasattr(raw, "tier"):
            tiers[name] = raw.tier
        else:
            tiers[name] = "polish"
    return tiers


class ImpactScore:
    """Result of measuring impact for a single option."""

    def __init__(self, name: str, fitness_delta: float) -> None:
        self.name: str = name
        # Positive delta means the option reduced fitness (good).
        self.fitness_delta: float = fitness_delta

    @override
    def __repr__(self) -> str:
        """Return string representation."""
        return f"ImpactScore({self.name!r}, delta={self.fitness_delta:.1f})"


def measure_remaining_impact(
    repo_path: str,
    candidate_names: list[str],
    base_options: dict[str, Any],
    lookups: GeneticAlgorithmLookups,
    current_config: dict[str, Any],
    process_id: int = 0,
    debug: bool = False,
    file_sample_percentage: float = 100.0,
    random_seed: int = 42,
) -> list[ImpactScore]:
    """Score the impact of remaining (fixed) options and return a ranked list.

    Builds a search space from the candidate option names, runs nevergrad,
    and measures how much each option contributed to fitness improvement by
    comparing the best config to the initial config.

    Args:
        repo_path: Path to the git repository.
        candidate_names: Names of options to score (should be currently fixed).
        base_options: Flat options dict from clang-format --dump-config.
        lookups: Contains json_options_lookup and forced_options_lookup.
        current_config: Current best config values (starting point).
        process_id: Worker process ID.
        debug: Enable debug output.
        file_sample_percentage: Percentage of files to sample.
        random_seed: Seed for random file sampling.

    Returns:
        List of ImpactScore sorted by fitness_delta descending (most impactful first).
        Empty list if no candidates have optimizable values.
    """
    # Build search space from candidate options.
    parameters: dict[str, ParameterDef] = {}

    for full_path in candidate_names:
        if full_path not in base_options:
            continue

        json_info = lookups.json_options_lookup.get(full_path, {})
        possible_values = json_info.get("possible_values")
        if not possible_values:
            # Penalty options get curated values.
            if full_path.startswith("Penalty"):
                possible_values = list(CURATED_PENALTY_VALUES)
            else:
                continue

        # Skip if only one possible value.
        if len(possible_values) <= 1:
            continue

        parameters[full_path] = ParameterDef(
            name=full_path,
            param_type=base_options[full_path]["type"],
            possible_values=list(possible_values),
            fixed=False,
        )

    if not parameters:
        return []

    search_space = SearchSpace(parameters=parameters)

    # Build initial flat options from current config.
    initial_flat = _build_flat_options_from_config(
        base_options, current_config, lookups.forced_options_lookup
    )

    # Build fitness function.
    def fitness(config: dict[str, Any]) -> float:  # pragma: no cover
        flat = copy.deepcopy(initial_flat)
        for name, val in config.items():
            if name in flat:
                target_type = flat[name].get("type", "str")
                if target_type == "int":
                    try:
                        flat[name]["value"] = int(val)
                    except (ValueError, TypeError):
                        pass
                elif target_type == "bool":
                    flat[name]["value"] = bool(val)
                else:
                    flat[name]["value"] = val

        # Apply forced options.
        for forced_path, forced_value in lookups.forced_options_lookup.items():
            if forced_path in flat:
                flat[forced_path]["value"] = forced_value

        config_string = generate_clang_format_config(flat)

        from ..repo_formatter import run_clang_format_and_count_changes

        changes = run_clang_format_and_count_changes(
            config_string,
            repo_path=repo_path,
            process_id=process_id,
            debug=debug,
            file_sample_percentage=file_sample_percentage,
            random_seed=random_seed,
        )

        if changes == -1:
            return float("inf")
        return changes

    # Build initial config from current values.
    initial_config = {}
    for name in parameters:
        if name in current_config:
            initial_config[name] = current_config[name]
        elif name in base_options:
            initial_config[name] = base_options[name].get("value")

    # Run nevergrad.
    result = run_nevergrad_optimization(
        search_space=search_space,
        objective=fitness,
        budget=NG_SAFETY_BUDGET,
        num_workers=1,
        optimizer_name="TwoPointsDE",
        debug=debug,
        initial_config=initial_config,
        convergence_threshold=10,
    )

    # Measure per-option impact by comparing best to initial.
    initial_fitness = fitness(initial_config)
    scores: list[ImpactScore] = []
    for name in parameters:
        best_val = result.best_config.get(name)
        init_val = initial_config.get(name)
        if best_val is not None and best_val != init_val:
            # Estimate impact: create a config with only this option changed.
            solo_config = copy.deepcopy(initial_config)
            solo_config[name] = best_val
            solo_fitness = fitness(solo_config)
            delta = initial_fitness - solo_fitness
            scores.append(ImpactScore(name=name, fitness_delta=delta))

    # Sort by delta descending (most impactful first).
    scores.sort(key=lambda s: s.fitness_delta, reverse=True)
    return scores


def measure_impact(
    repo_path: str,
    analysis_results: dict[str, Any],
    base_options: dict[str, Any],
    lookups: GeneticAlgorithmLookups,
    process_id: int = 0,
    debug: bool = False,
    file_sample_percentage: float = 100.0,
    random_seed: int = 42,
) -> dict[str, str]:
    """Run a lightweight nevergrad optimization to determine option impact.

    Builds a search space from detected options, runs nevergrad with a small
    budget, then compares the best config to the initial config. Options that
    changed are assigned 'structure' tier; unchanged options are 'polish'.
    Options with 'guessed' confidence are always 'resolve'.

    Args:
        repo_path: Path to the git repository.
        analysis_results: Dict from analyze() or analyze_with_metadata().
        base_options: Flat options dict from clang-format --dump-config.
        lookups: Contains json_options_lookup and forced_options_lookup.
        process_id: Worker process ID.
        debug: Enable debug output.
        file_sample_percentage: Percentage of files to sample.
        random_seed: Seed for random file sampling.

    Returns:
        Dict mapping option name to tier assignment.
    """
    # Build search space from detected options only.
    parameters: dict[str, ParameterDef] = {}

    for full_path, raw_value in analysis_results.items():
        if full_path not in base_options:
            continue

        possible_values = lookups.json_options_lookup.get(full_path, {}).get(
            "possible_values"
        )
        if not possible_values:
            continue

        possible_values = list(possible_values)

        # Skip if only one possible value — nothing to optimize.
        if len(possible_values) <= 1:
            continue

        parameters[full_path] = ParameterDef(
            name=full_path,
            param_type=base_options[full_path]["type"],
            possible_values=possible_values,
            fixed=False,
        )

    if not parameters:
        # No options have possible_values — cannot measure impact.
        # Preserve the analyzer's original tier assignments.
        return _preserve_analyzer_tiers(analysis_results)

    search_space = SearchSpace(parameters=parameters)

    # Build initial flat options from detected values.
    initial_flat = _build_flat_options(
        base_options, analysis_results, lookups.forced_options_lookup
    )

    # Build fitness function.
    def fitness(config: dict[str, Any]) -> float:  # pragma: no cover
        flat = copy.deepcopy(initial_flat)
        for name, val in config.items():
            if name in flat:
                target_type = flat[name].get("type", "str")
                if target_type == "int":
                    try:
                        flat[name]["value"] = int(val)
                    except (ValueError, TypeError):
                        pass
                elif target_type == "bool":
                    flat[name]["value"] = bool(val)
                else:
                    flat[name]["value"] = val

        # Apply forced options.
        for forced_path, forced_value in lookups.forced_options_lookup.items():
            if forced_path in flat:
                flat[forced_path]["value"] = forced_value

        config_string = generate_clang_format_config(flat)

        from ..repo_formatter import run_clang_format_and_count_changes

        changes = run_clang_format_and_count_changes(
            config_string,
            repo_path=repo_path,
            process_id=process_id,
            debug=debug,
            file_sample_percentage=file_sample_percentage,
            random_seed=random_seed,
        )

        if changes == -1:
            return float("inf")
        return changes

    # Build initial config from detected values.
    initial_config = {}
    for name, raw_value in analysis_results.items():
        if name in parameters:
            if hasattr(raw_value, "value"):
                initial_config[name] = raw_value.value
            else:
                initial_config[name] = raw_value

    # Run nevergrad with small budget.
    result = run_nevergrad_optimization(
        search_space=search_space,
        objective=fitness,
        budget=NG_SAFETY_BUDGET,
        num_workers=1,
        optimizer_name="TwoPointsDE",
        debug=debug,
        initial_config=initial_config,
        convergence_threshold=10,
    )

    # Compare best config to initial config to determine impact.
    tiers: dict[str, str] = {}
    for name in analysis_results:
        raw = analysis_results[name]
        if hasattr(raw, "confidence") and raw.confidence == "guessed":
            tiers[name] = "resolve"
            continue

        if name in result.best_config and name in initial_config:
            if result.best_config[name] != initial_config[name]:
                tiers[name] = "structure"
            else:
                tiers[name] = "polish"
        else:
            tiers[name] = "polish"

    return tiers


def _build_flat_options(
    base_options: dict[str, Any],
    analysis_results: dict[str, Any],
    forced_options: dict[str, Any],
) -> dict[str, Any]:
    """Build a flat options dict from base + analysis + forced overrides."""
    flat = copy.deepcopy(base_options)

    for name, raw_value in analysis_results.items():
        if hasattr(raw_value, "value"):
            value = raw_value.value
        else:
            value = raw_value
        _set_option_value(flat, name, value)

    for name, value in forced_options.items():
        _set_option_value(flat, name, value)

    return flat


def _build_flat_options_from_config(
    base_options: dict[str, Any],
    config: dict[str, Any],
    forced_options: dict[str, Any],
) -> dict[str, Any]:
    """Build a flat options dict from base + config values + forced overrides."""
    flat = copy.deepcopy(base_options)

    for name, value in config.items():
        _set_option_value(flat, name, value)

    for name, value in forced_options.items():
        _set_option_value(flat, name, value)

    return flat


def _set_option_value(
    flat: dict[str, Any],
    name: str,
    value: Any,
) -> None:
    """Set a value in a flat options dict, respecting type hints."""
    if name not in flat:
        return
    target_type = flat[name].get("type", "str")
    if target_type == "int":
        try:
            flat[name]["value"] = int(value)
        except (ValueError, TypeError):
            pass
    elif target_type == "bool":
        flat[name]["value"] = bool(value)
    else:
        flat[name]["value"] = value
