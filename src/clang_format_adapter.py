"""Clang-format adapter for the generic optimization engine.

Provides the bridge between clang-format domain objects and the generic
optimization_engine types (SearchSpace, Individual, etc.).
"""

from __future__ import annotations

import copy
import sys
import threading
from collections.abc import Callable
from typing import Any

from .analyze_conventions import DetectedOption
from .clang_format_parser import generate_clang_format_config
from .data_classes import GeneticAlgorithmLookups
from .optimization_engine.types import ParameterDef, SearchSpace
from .repo_formatter import run_clang_format_and_count_changes


CURATED_PENALTY_VALUES = [2, 10, 50, 150, 1000, 10000]


def build_search_space(
    base_options: dict[str, Any],
    lookups: GeneticAlgorithmLookups,
    analysis_results: dict[str, Any] | None = None,
    polish_undetect: bool = False,
) -> SearchSpace:
    """Build a SearchSpace from clang-format base options and lookups.

    Args:
        base_options: Flat options dict from clang-format --dump-config.
        lookups: Contains json_options_lookup and forced_options_lookup.
        analysis_results: Optional dict from analyze_conventions.analyze().
            When provided, detected values become the only possible_values
            for matching parameters, pruning the search space.
        polish_undetect: When True, undetected options that have
            possible_values are made mutable instead of fixed. This allows
            the optimizer to tune options the analyzer didn't detect.

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

        # Analysis results: detected conventions are mutable (GA optimizes them).
        # Use the detected value as the seed, but allow mutation to other values.
        # Exception: penalty options are always fixed — they must be optimized as a group.
        if analysis_results and full_path in analysis_results:
            raw = analysis_results[full_path]
            # Handle both old format (raw values) and new format (DetectedOption).
            if isinstance(raw, DetectedOption):
                value = raw.value
                tier = raw.tier
                # Forced options are fixed — the analyzer is certain of the value.
                is_forced = raw.confidence == "forced"
            else:
                value = raw
                tier = "polish"
                is_forced = False
            possible_values = lookups.json_options_lookup.get(full_path, {}).get(
                "possible_values"
            ) or [value]
            possible_values = list(possible_values)
            # Force penalty options to be fixed, even if detected.
            is_penalty = full_path.startswith("Penalty")
            parameters[full_path] = ParameterDef(
                name=full_path,
                param_type=option_info["type"],
                possible_values=possible_values,
                fixed=is_penalty or is_forced,
                tier=tier,
            )
            continue

        # Undetected options.
        json_info = lookups.json_options_lookup.get(full_path, {})
        json_values = json_info.get("possible_values")

        # Penalty options get curated values but start fixed.
        # They are relative weights that only make sense when optimized together.
        # The iterative optimizer unlocks them as a group in a final polish step.
        if full_path.startswith("Penalty"):
            parameters[full_path] = ParameterDef(
                name=full_path,
                param_type=option_info["type"],
                possible_values=list(CURATED_PENALTY_VALUES),
                fixed=True,
                tier="polish",
            )
            continue

        # If polish_undetect is enabled and the option has possible_values,
        # make it mutable so the optimizer can tune it.
        if polish_undetect and json_values:
            parameters[full_path] = ParameterDef(
                name=full_path,
                param_type=option_info["type"],
                possible_values=list(json_values),
                fixed=False,
                tier="polish",
            )
            continue

        # Default: undetected options are fixed at their dump-config values.
        parameters[full_path] = ParameterDef(
            name=full_path,
            param_type=option_info["type"],
            possible_values=[],
            fixed=True,
        )

    return SearchSpace(parameters=parameters)


class FitnessEvaluator:
    """Picklable fitness evaluator for clang-format optimization.

    Must be a module-level class so ProcessPoolExecutor can pickle it.
    """

    _repo_paths: list[str]
    _process_id: int
    _forced_options_lookup: dict[str, Any]
    _base_options: dict[str, Any]
    _debug: bool
    _file_sample_percentage: float
    _random_seed: int
    _repo_index: int
    _repo_lock: threading.Lock
    _repo_busy_locks: list[threading.Lock]

    def __init__(
        self,
        repo_paths: list[str],
        process_id: int,
        lookups: GeneticAlgorithmLookups,
        base_options: dict[str, Any],
        debug: bool = False,
        file_sample_percentage: float = 100.0,
        random_seed: int = 42,
    ) -> None:
        self._repo_paths = repo_paths
        self._process_id = process_id
        self._forced_options_lookup = lookups.forced_options_lookup
        self._base_options = base_options
        self._debug = debug
        self._file_sample_percentage = file_sample_percentage
        self._random_seed = random_seed
        self._repo_index = 0
        self._repo_lock = threading.Lock()
        self._repo_busy_locks = [threading.Lock() for _ in repo_paths]

    def _get_repo_path(self) -> tuple[str, int]:
        """Select a repo path using round-robin to avoid contention.

        Uses a thread-safe counter so each parallel evaluation gets a
        distinct repo copy. Blocks if the selected repo is still in use
        by another thread, preventing race conditions on git operations.
        Works for both ThreadPoolExecutor and ProcessPoolExecutor.

        Returns:
            Tuple of (repo_path, repo_index) so the caller can release the lock.
        """
        with self._repo_lock:
            idx = self._repo_index % len(self._repo_paths)
            self._repo_index += 1
        # Acquire the per-repo lock to prevent concurrent access.
        # This blocks until the repo is free, ensuring no two threads
        # operate on the same repo simultaneously.
        _ = self._repo_busy_locks[idx].acquire()
        return self._repo_paths[idx], idx

    def __call__(self, config: dict[str, Any]) -> float:
        # Get repo path (acquires per-repo lock to prevent concurrent access)
        repo_path, repo_idx = self._get_repo_path()
        try:
            # Start from base template and apply config values
            flat_options = copy.deepcopy(self._base_options)

            for name, value in config.items():
                if name in flat_options:
                    target_type = flat_options[name]["type"]
                    if target_type == "int":
                        try:
                            flat_options[name]["value"] = int(value)
                        except (ValueError, TypeError):
                            if self._debug:
                                print(
                                    f"Worker {self._process_id}: Could not convert '{value}' to int for '{name}'. Skipping.",
                                    file=sys.stderr,
                                )
                            continue
                    elif target_type == "bool":
                        flat_options[name]["value"] = bool(value)
                    else:
                        flat_options[name]["value"] = value

            # Apply forced options
            for forced_path, forced_value in self._forced_options_lookup.items():
                if forced_path in flat_options:
                    flat_options[forced_path]["value"] = forced_value

            config_string = generate_clang_format_config(flat_options)

            changes = run_clang_format_and_count_changes(
                config_string,
                repo_path=repo_path,
                process_id=self._process_id,
                debug=self._debug,
                file_sample_percentage=self._file_sample_percentage,
                random_seed=self._random_seed,
            )

            if changes == -1:
                return float("inf")
            return changes
        finally:
            # Release the per-repo lock so other threads can use this repo
            self._repo_busy_locks[repo_idx].release()


def make_fitness_function(
    repo_paths: list[str],
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
        repo_paths: Paths to temporary git repositories. Each worker process
            selects one based on its PID to avoid contention.
        process_id: Worker process ID.
        lookups: Contains forced_options_lookup.
        base_options: Base flat options dict (template for config generation).
        debug: Enable debug output.
        file_sample_percentage: Percentage of files to sample.
        random_seed: Seed for random file sampling.

    Returns:
        Callable that evaluates a config and returns fitness.
    """
    return FitnessEvaluator(
        repo_paths=repo_paths,
        process_id=process_id,
        lookups=lookups,
        base_options=base_options,
        debug=debug,
        file_sample_percentage=file_sample_percentage,
        random_seed=random_seed,
    )


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
