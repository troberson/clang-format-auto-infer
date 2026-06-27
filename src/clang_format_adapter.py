"""Clang-format adapter for the generic optimization engine.

Provides the bridge between clang-format domain objects and the generic
optimization_engine types (SearchSpace, Individual, etc.).
"""

from __future__ import annotations

import copy
import sys
from collections.abc import Callable
from typing import Any

from .clang_format_parser import generate_clang_format_config
from .data_classes import GeneticAlgorithmLookups
from .optimization_engine.types import ParameterDef, SearchSpace
from .repo_formatter import run_clang_format_and_count_changes


def build_search_space(
    base_options: dict[str, Any],
    lookups: GeneticAlgorithmLookups,
    analysis_results: dict[str, Any] | None = None,
) -> SearchSpace:
    """Build a SearchSpace from clang-format base options and lookups.

    Args:
        base_options: Flat options dict from clang-format --dump-config.
        lookups: Contains json_options_lookup and forced_options_lookup.
        analysis_results: Optional dict from analyze_conventions.analyze().
            When provided, detected values become the only possible_values
            for matching parameters, pruning the search space.

    Returns:
        SearchSpace with all tunable parameters.
    """
    parameters: dict[str, ParameterDef] = {}

    for full_path, option_info in base_options.items():
        # Forced options are fixed
        if full_path in lookups.forced_options_lookup:
            parameters[full_path] = ParameterDef(
                name=full_path,
                param_type=option_info["type"],
                possible_values=[],
                fixed=True,
            )
            continue

        # Analysis results override JSON lookup values
        if analysis_results and full_path in analysis_results:
            value = analysis_results[full_path]
            parameters[full_path] = ParameterDef(
                name=full_path,
                param_type=option_info["type"],
                possible_values=[value],
                fixed=False,
            )
            continue

        # Get possible values from JSON lookup
        possible_values = []
        if full_path in lookups.json_options_lookup:
            possible_values = list(
                lookups.json_options_lookup[full_path].get("possible_values", [])
            )

        parameters[full_path] = ParameterDef(
            name=full_path,
            param_type=option_info["type"],
            possible_values=possible_values,
            fixed=False,
        )

    return SearchSpace(parameters=parameters)


def make_fitness_function(
    repo_path: str,
    process_id: int,
    lookups: GeneticAlgorithmLookups,
    base_options: dict[str, Any],
    debug: bool = False,
    file_sample_percentage: float = 100.0,
    random_seed: int = 42,
) -> Callable[[dict[str, Any]], float]:
    """Create a fitness function for clang-format optimization.

    The returned callable takes a config dict (parameter name → value) and
    returns the number of changes clang-format would make (lower is better).

    Args:
        repo_path: Path to the git repository to format.
        process_id: Worker process ID.
        lookups: Contains forced_options_lookup.
        base_options: Base flat options dict (template for config generation).
        debug: Enable debug output.
        file_sample_percentage: Percentage of files to sample.
        random_seed: Seed for random file sampling.

    Returns:
        Callable that evaluates a config and returns fitness.
    """

    def fitness(config: dict[str, Any]) -> float:
        # Start from base template and apply config values
        flat_options = copy.deepcopy(base_options)

        for name, value in config.items():
            if name in flat_options:
                target_type = flat_options[name]["type"]
                if target_type == "int":
                    try:
                        flat_options[name]["value"] = int(value)
                    except (ValueError, TypeError):
                        if debug:
                            print(
                                f"Worker {process_id}: Could not convert '{value}' to int for '{name}'. Skipping.",
                                file=sys.stderr,
                            )
                        continue
                elif target_type == "bool":
                    flat_options[name]["value"] = bool(value)
                else:
                    flat_options[name]["value"] = value

        # Apply forced options
        for forced_path, forced_value in lookups.forced_options_lookup.items():
            if forced_path in flat_options:
                flat_options[forced_path]["value"] = forced_value

        config_string = generate_clang_format_config(flat_options)

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

    return fitness


def config_to_flat_options(
    config: dict[str, Any],
    base_options: dict[str, Any],
) -> dict[str, Any]:
    """Convert a generic config dict to clang-format flat options format.

    Args:
        config: Generic config with parameter name → value.
        base_options: Base flat options template.

    Returns:
        Flat options dict ready for generate_clang_format_config.
    """
    flat = copy.deepcopy(base_options)
    for name, value in config.items():
        if name in flat:
            target_type = flat[name]["type"]
            if target_type == "int":
                try:
                    flat[name]["value"] = int(value)
                except (ValueError, TypeError):
                    pass
            elif target_type == "bool":
                flat[name]["value"] = bool(value)
            else:
                flat[name]["value"] = value
    return flat
