"""Generic coordinate descent polish.

Iterates over mutable parameters, exhaustively testing all values for each.
Repeats passes until no improvements are found or max_passes is reached.
"""

from __future__ import annotations

import copy
import sys
from typing import Any, Callable

from .types import Individual, SearchSpace

FitnessFn = Callable[[dict[str, Any]], float]


def polish_coordinate_descent(
    config: dict[str, Any],
    initial_fitness: float,
    search_space: SearchSpace,
    fitness_fn: FitnessFn,
    max_passes: int,
    debug: bool = False,
) -> Individual:
    """Run coordinate descent polish on the given configuration.

    For each mutable parameter, test all possible values and keep the best.
    Repeat until no improvements or max_passes reached.

    Args:
        config: Starting configuration (param_name -> value).
        initial_fitness: Fitness of the starting configuration.
        search_space: Defines parameters and their possible values.
        fitness_fn: Callable that scores a config dict (lower is better).
        max_passes: Maximum number of full passes over all parameters.
        debug: Print verbose output.

    Returns:
        Individual with the best config and fitness found.
    """
    mutable = search_space.mutable_parameters
    if not mutable:
        if debug:
            print("  No mutable options for polish. Skipping.", file=sys.stderr)
        return Individual(config=config, fitness=initial_fitness)

    current_fitness = initial_fitness
    if debug:
        print(
            f"  Starting coordinate descent polish. {len(mutable)} mutable options.",
            file=sys.stderr,
        )
        print(f"  Initial fitness: {current_fitness}", file=sys.stderr)

    for pass_num in range(1, max_passes + 1):
        improvements = 0

        for param in mutable:
            if param.name not in config:
                continue

            old_fitness = current_fitness
            best_fitness = old_fitness
            best_value = config[param.name]

            for value in param.possible_values:
                if value == config[param.name]:
                    continue
                config_copy = copy.deepcopy(config)
                config_copy[param.name] = value
                fitness = fitness_fn(config_copy)
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_value = value

            if best_fitness < old_fitness:
                config[param.name] = best_value
                current_fitness = best_fitness
                improvements += 1

        if debug:
            print(
                f"  Polish pass {pass_num}: {improvements} improvements, fitness: {current_fitness}",
                file=sys.stderr,
            )

        if improvements == 0:
            if debug:
                print(
                    f"  Coordinate descent converged after {pass_num} pass(es).",
                    file=sys.stderr,
                )
            break
    else:
        if debug:
            print(
                f"  Coordinate descent reached max passes ({max_passes}).",
                file=sys.stderr,
            )

    return Individual(config=config, fitness=current_fitness)
