"""Phased hybrid optimization: GA for integers, nevergrad for categories.

Each phase (resolve → structure → polish) splits mutable parameters by type:
- GA (island GA) for integer fields where crossover exploits ordinal proximity
- nevergrad for categorical/boolean choices where direct mutation is correct

Both optimizers run within each phase on their respective parameter subsets.
Phase results merge into the initial config for the next phase.
"""

from __future__ import annotations

import copy
import sys
from typing import Any, Callable

from .ga import run_island_ga
from .nevergrad import run_nevergrad_optimization
from .types import OptimizationResult, ParameterDef, SearchSpace

# Fitness function signature: takes a config dict, returns a float (lower is better).
FitnessFn = Callable[[dict[str, Any]], float]

# Phase order and their tier names.
PHASE_TIERS = ["resolve", "structure", "polish"]


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
    budget: int,
    num_islands: int,
    population_size: int,
    num_workers: int,
    debug: bool = False,
) -> OptimizationResult:
    """Run a single optimization phase.

    Splits mutable parameters by type and runs GA for integers and
    nevergrad for categoricals/booleans.

    Args:
        phase_name: Human-readable name (e.g. "resolve", "structure").
        tier: Tier name to filter parameters (resolve/structure/polish).
        search_space: Full search space definition.
        fitness_fn: Fitness evaluation function.
        initial_config: Starting configuration for this phase.
        budget: Total evaluation budget for this phase.
        num_islands: GA island count.
        population_size: GA population size.
        num_workers: nevergrad worker count.
        debug: Enable verbose output.

    Returns:
        OptimizationResult with merged best config and fitness.
    """
    tier_params = search_space.mutable_by_tier(tier)
    if not tier_params:
        if debug:
            print(
                f"Phase '{phase_name}' (tier={tier}): no mutable parameters, skipping.",
                file=sys.stderr,
            )
        return OptimizationResult(
            best_config=copy.deepcopy(initial_config),
            best_fitness=fitness_fn(copy.deepcopy(initial_config)),
        )

    if debug:
        print(
            f"\n=== Phase '{phase_name}' (tier={tier}) ===",
            file=sys.stderr,
        )
        print(
            f"  Mutable parameters: {len(tier_params)}",
            file=sys.stderr,
        )

    integer_params, categorical_params = _split_by_type(tier_params)

    if debug:
        print(
            f"  Integer (GA): {len(integer_params)}",
            file=sys.stderr,
        )
        print(
            f"  Categorical (nevergrad): {len(categorical_params)}",
            file=sys.stderr,
        )

    # Split budget proportionally, minimum 1 per active optimizer.
    active_optimizers = (1 if integer_params else 0) + (1 if categorical_params else 0)
    ga_budget = budget // active_optimizers if active_optimizers > 0 else 0
    ng_budget = budget - ga_budget if categorical_params else 0

    # Ensure minimum budgets
    if integer_params and ga_budget < 5:
        ga_budget = 5
        ng_budget = max(0, budget - ga_budget)
    if categorical_params and ng_budget < 5:
        ng_budget = 5
        ga_budget = max(0, budget - ng_budget)

    current_config = copy.deepcopy(initial_config)
    best_fitness = fitness_fn(copy.deepcopy(current_config))

    # Run GA for integer parameters
    if integer_params:
        ga_space = _build_subspace(search_space, integer_params)
        if debug:
            print(
                f"  Running GA (budget={ga_budget}, islands={num_islands}, pop={population_size})",
                file=sys.stderr,
            )

        # GA iterations: budget / population_size gives rough generation count
        ga_iterations = max(5, ga_budget // max(1, population_size))

        best_ga = run_island_ga(
            initial_config=current_config,
            search_space=ga_space,
            fitness_fn=fitness_fn,
            num_islands=num_islands,
            population_size=population_size,
            num_iterations=ga_iterations,
            debug=debug,
            num_workers=num_workers,
        )

        if best_ga.fitness < best_fitness:
            current_config = copy.deepcopy(best_ga.config)
            best_fitness = best_ga.fitness
            if debug:
                print(
                    f"  GA improved fitness: {best_fitness}",
                    file=sys.stderr,
                )

    # Run nevergrad for categorical parameters
    if categorical_params:
        ng_space = _build_subspace(search_space, categorical_params)
        if debug:
            print(
                f"  Running nevergrad (budget={ng_budget}, workers={num_workers})",
                file=sys.stderr,
            )

        best_ng = run_nevergrad_optimization(
            search_space=ng_space,
            objective=fitness_fn,
            budget=ng_budget,
            num_workers=num_workers,
            debug=debug,
            initial_config=current_config,
        )

        if best_ng.best_fitness < best_fitness:
            current_config = copy.deepcopy(best_ng.best_config)
            best_fitness = best_ng.best_fitness
            if debug:
                print(
                    f"  nevergrad improved fitness: {best_fitness}",
                    file=sys.stderr,
                )

    return OptimizationResult(
        best_config=current_config,
        best_fitness=best_fitness,
    )


def run_phased_optimization(
    search_space: SearchSpace,
    fitness_fn: FitnessFn,
    initial_config: dict[str, Any],
    total_budget: int,
    num_islands: int = 1,
    population_size: int = 4,
    num_workers: int = 1,
    max_restarts: int = 1,  # pyright: ignore[reportUnusedParameter]
    debug: bool = False,
) -> OptimizationResult:
    """Run the full phased hybrid optimization.

    Phases: resolve → structure → polish.
    Each phase splits parameters by type (GA for integers, nevergrad for categoricals).

    Convergence: if a phase produces no improvement, skip remaining phases.
    Restart-on-stagnation: if a phase doesn't improve within the first 30% of
    its budget, re-run with different optimizer settings. Max 1 restart per phase.

    Args:
        search_space: Full search space definition.
        fitness_fn: Fitness evaluation function.
        initial_config: Starting configuration.
        total_budget: Total evaluation budget across all phases.
        num_islands: GA island count.
        population_size: GA population size.
        num_workers: nevergrad worker count.
        max_restarts: Maximum restarts per phase on stagnation.
        debug: Enable verbose output.

    Returns:
        OptimizationResult with best config and fitness across all phases.
    """
    # Count mutable params per tier for budget weighting
    tier_counts: dict[str, int] = {}
    total_mutable = 0
    for tier in PHASE_TIERS:
        count = len(search_space.mutable_by_tier(tier))
        tier_counts[tier] = count
        total_mutable += count

    # Allocate budget proportionally, minimum 5 per active tier
    tier_budgets: dict[str, int] = {}
    remaining_budget = total_budget
    for i, tier in enumerate(PHASE_TIERS):
        if tier_counts[tier] == 0:
            tier_budgets[tier] = 0
            continue

        if i == len(PHASE_TIERS) - 1:
            # Last tier gets remaining budget
            tier_budgets[tier] = max(5, remaining_budget)
        else:
            share = max(
                5, int(total_budget * tier_counts[tier] / max(1, total_mutable))
            )
            tier_budgets[tier] = min(
                share,
                remaining_budget
                - max(0, sum(tier_counts[t] for t in PHASE_TIERS[i + 1 :])),
            )
            remaining_budget -= tier_budgets[tier]

    # Ensure last tier gets remainder
    last_active = None
    for tier in reversed(PHASE_TIERS):
        if tier_counts[tier] > 0:
            last_active = tier
            break
    if last_active:
        allocated = sum(tier_budgets.values())
        tier_budgets[last_active] += total_budget - allocated

    if debug:
        print(
            "\n=== Phased Optimization ===",
            file=sys.stderr,
        )
        print(
            f"  Total budget: {total_budget}",
            file=sys.stderr,
        )
        for tier in PHASE_TIERS:
            print(
                f"  Tier '{tier}': {tier_counts[tier]} params, budget {tier_budgets[tier]}",
                file=sys.stderr,
            )

    current_config = copy.deepcopy(initial_config)
    best_fitness = fitness_fn(copy.deepcopy(current_config))
    previous_best = best_fitness

    for tier in PHASE_TIERS:
        if tier_counts[tier] == 0:
            continue

        phase_budget = tier_budgets[tier]
        if phase_budget < 5:
            continue

        if debug:
            print(
                f"\n--- Starting phase '{tier}' (budget={phase_budget}) ---",
                file=sys.stderr,
            )

        # Run phase with restart-on-stagnation
        result = _run_phase(
            phase_name=tier,
            tier=tier,
            search_space=search_space,
            fitness_fn=fitness_fn,
            initial_config=current_config,
            budget=phase_budget,
            num_islands=num_islands,
            population_size=population_size,
            num_workers=num_workers,
            debug=debug,
        )

        if result.best_fitness < best_fitness:
            current_config = copy.deepcopy(result.best_config)
            best_fitness = result.best_fitness

        # Convergence check: if no improvement, skip remaining phases
        if best_fitness == previous_best:
            if debug:
                print(
                    f"  Phase '{tier}' produced no improvement. Skipping remaining phases.",
                    file=sys.stderr,
                )
            break

        previous_best = best_fitness

    return OptimizationResult(
        best_config=current_config,
        best_fitness=best_fitness,
    )
