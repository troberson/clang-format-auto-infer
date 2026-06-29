"""Phased hybrid optimization: GA for integers, nevergrad for categories.

Each phase (resolve → structure → polish) splits mutable parameters by type:
- GA (island GA) for integer fields where crossover exploits ordinal proximity
- nevergrad for categorical/boolean choices where direct mutation is correct

Both optimizers run within each phase on their respective parameter subsets.
Phase results merge into the initial config for the next phase.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

from ..utils import dbg
from .ga import run_island_ga
from .nevergrad import run_nevergrad_optimization
from .types import OptimizationResult, ParameterDef, SearchSpace

# Fitness function signature: takes a config dict, returns a float (lower is better).
FitnessFn = Callable[[dict[str, Any]], float]

# Phase order and their tier names.
PHASE_TIERS = ["resolve", "structure", "polish"]

# Safety cap for nevergrad evaluations. Convergence fires first in practice.
NG_SAFETY_BUDGET = 10_000

# Default convergence threshold for phase sub-runs.
CONVERGENCE_THRESHOLD = 20


def _split_by_type(
    params: list[ParameterDef],
) -> tuple[list[ParameterDef], list[ParameterDef]]:
    """Split parameters into integer (GA) and categorical (nevergrad) groups.

    Integers benefit from GA crossover (ordinal proximity).
    Categoricals and booleans benefit from nevergrad's direct mutation.
    """
    integer_params: list[ParameterDef] = []
    categorical_params: list[ParameterDef] = []

    for p in params:
        if p.param_type == "int":
            integer_params.append(p)
        else:
            categorical_params.append(p)

    return integer_params, categorical_params


def _build_subspace(
    parent_space: SearchSpace,
    params: list[ParameterDef],
) -> SearchSpace:
    """Build a SearchSpace containing only the given parameters.

    All parameters not in the list are fixed.
    """
    active_names = {p.name for p in params}
    filtered: dict[str, ParameterDef] = {}
    for name, param in parent_space.parameters.items():
        if name in active_names:
            filtered[name] = param
        else:
            # Freeze everything else
            filtered[name] = ParameterDef(
                name=param.name,
                param_type=param.param_type,
                possible_values=[],
                fixed=True,
            )
    return SearchSpace(parameters=filtered)


def _run_phase(
    phase_name: str,
    tier: str,
    search_space: SearchSpace,
    fitness_fn: FitnessFn,
    initial_config: dict[str, Any],
    num_islands: int,
    population_size: int,
    num_workers: int,
    convergence_threshold: int = CONVERGENCE_THRESHOLD,
    debug: bool = False,
) -> OptimizationResult:
    """Run a single optimization phase.

    Splits mutable parameters by type and runs GA for integers and
    nevergrad for categoricals/booleans. Termination is convergence-based.

    Args:
        phase_name: Human-readable name (e.g. "resolve", "structure").
        tier: Tier name to filter parameters (resolve/structure/polish).
        search_space: Full search space definition.
        fitness_fn: Fitness evaluation function.
        initial_config: Starting configuration for this phase.
        num_islands: GA island count.
        population_size: GA population size.
        num_workers: nevergrad worker count.
        convergence_threshold: Stop sub-runs when no improvement.
        debug: Enable verbose output.

    Returns:
        OptimizationResult with merged best config and fitness.
    """
    tier_params = search_space.mutable_by_tier(tier)
    if not tier_params:
        if debug:
            dbg(phase_name, "no mutable parameters, skipping.")
        return OptimizationResult(
            best_config=copy.deepcopy(initial_config),
            best_fitness=fitness_fn(copy.deepcopy(initial_config)),
        )

    if debug:
        dbg(phase_name, f"=== Phase '{phase_name}' (tier={tier}) ===", summary=True)
        dbg(phase_name, f"Mutable parameters: {len(tier_params)}")

    integer_params, categorical_params = _split_by_type(tier_params)

    if debug:
        dbg(phase_name, f"Integer (GA): {len(integer_params)}")
        dbg(phase_name, f"Categorical (nevergrad): {len(categorical_params)}")

    current_config = copy.deepcopy(initial_config)
    best_fitness = fitness_fn(copy.deepcopy(current_config))

    # Run GA for integer parameters
    if integer_params:
        ga_space = _build_subspace(search_space, integer_params)
        if debug:
            dbg(
                phase_name, f"Running GA (islands={num_islands}, pop={population_size})"
            )

        best_ga = run_island_ga(
            initial_config=current_config,
            search_space=ga_space,
            fitness_fn=fitness_fn,
            num_islands=num_islands,
            population_size=population_size,
            num_iterations=None,
            debug=debug,
            convergence_threshold=convergence_threshold,
            num_workers=num_workers,
            tag=phase_name,
        )

        if best_ga.fitness < best_fitness:
            current_config = copy.deepcopy(best_ga.config)
            best_fitness = best_ga.fitness
            if debug:
                dbg(phase_name, f"GA improved fitness: {best_fitness}")

    # Run nevergrad for categorical parameters
    if categorical_params:
        ng_space = _build_subspace(search_space, categorical_params)
        if debug:
            dbg(phase_name, f"Running nevergrad (workers={num_workers})")

        best_ng = run_nevergrad_optimization(
            search_space=ng_space,
            objective=fitness_fn,
            budget=NG_SAFETY_BUDGET,
            num_workers=num_workers,
            debug=debug,
            initial_config=current_config,
            convergence_threshold=convergence_threshold,
            tag=phase_name,
        )

        if best_ng.best_fitness < best_fitness:
            current_config = copy.deepcopy(best_ng.best_config)
            best_fitness = best_ng.best_fitness
            if debug:
                dbg(phase_name, f"nevergrad improved fitness: {best_fitness}")

    return OptimizationResult(
        best_config=current_config,
        best_fitness=best_fitness,
    )


def run_phased_optimization(
    search_space: SearchSpace,
    fitness_fn: FitnessFn,
    initial_config: dict[str, Any],
    num_islands: int = 1,
    population_size: int = 4,
    num_workers: int = 1,
    max_restarts: int = 1,  # pyright: ignore[reportUnusedParameter]
    convergence_threshold: int = CONVERGENCE_THRESHOLD,
    debug: bool = False,
) -> OptimizationResult:
    """Run the full phased hybrid optimization.

    Phases: resolve → structure → polish.
    Each phase splits parameters by type (GA for integers, nevergrad for categoricals).
    Termination is convergence-based — no budget limits.

    Convergence: if a phase produces no improvement, skip remaining phases.

    Args:
        search_space: Full search space definition.
        fitness_fn: Fitness evaluation function.
        initial_config: Starting configuration.
        num_islands: GA island count.
        population_size: GA population size.
        num_workers: nevergrad worker count.
        max_restarts: Maximum restarts per phase on stagnation.
        convergence_threshold: Stop sub-runs when no improvement.
        debug: Enable verbose output.

    Returns:
        OptimizationResult with best config and fitness across all phases.
    """
    if debug:
        dbg("phased", "=== Phased Optimization ===", summary=True)

    current_config = copy.deepcopy(initial_config)
    best_fitness = fitness_fn(copy.deepcopy(current_config))
    previous_best = best_fitness

    for tier in PHASE_TIERS:
        if len(search_space.mutable_by_tier(tier)) == 0:
            continue

        if debug:
            dbg(tier, f"--- Starting phase '{tier}' ---", summary=True)

        result = _run_phase(
            phase_name=tier,
            tier=tier,
            search_space=search_space,
            fitness_fn=fitness_fn,
            initial_config=current_config,
            num_islands=num_islands,
            population_size=population_size,
            num_workers=num_workers,
            convergence_threshold=convergence_threshold,
            debug=debug,
        )

        if result.best_fitness < best_fitness:
            current_config = copy.deepcopy(result.best_config)
            best_fitness = result.best_fitness

        # Convergence check: if no improvement, skip remaining phases
        if best_fitness == previous_best:
            if debug:
                dbg(
                    tier,
                    f"Phase '{tier}' produced no improvement. Skipping remaining phases.",
                )
            break

        previous_best = best_fitness

    return OptimizationResult(
        best_config=current_config,
        best_fitness=best_fitness,
    )
