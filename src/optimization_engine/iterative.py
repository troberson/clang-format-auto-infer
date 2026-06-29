"""Iterative expansion optimizer.

Starts with analyzer-detected options, polishes them to convergence, then
empirically discovers which remaining fixed options are most impactful and
unlocks them in batches. Repeats until no impactful options remain or all
options are exhausted.

Each optimization step splits parameters by type: GA for integers,
nevergrad for categoricals/booleans (reuses _run_phase from phased).
Termination is convergence-based — no budget limits.
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

# Maximum fraction of remaining options to unlock per iteration.
MAX_BATCH_FRACTION = 0.2

# Minimum fraction of top impact score to include in batch.
IMPACT_THRESHOLD = 0.5

# Minimum fitness improvement ratio to consider an iteration impactful.
MIN_IMPROVEMENT_RATIO = 0.01

# Convergence threshold for GA and nevergrad sub-runs.
CONVERGENCE_THRESHOLD = 20

# Safety cap for nevergrad evaluations. Convergence fires first in practice.
NG_SAFETY_BUDGET = 10_000


def _is_penalty_option(name: str) -> bool:
    """Return True if the option is a clang-format penalty parameter.

    Penalty options are relative weights that only make sense when optimized
    together. Testing them individually for impact is meaningless.
    """
    return name.startswith("Penalty")


def _select_batch(
    scores: list[Any],  # ImpactScore objects, sorted descending
    remaining_count: int,
    max_batch_fraction: float = MAX_BATCH_FRACTION,
    impact_threshold: float = IMPACT_THRESHOLD,
) -> list[str]:
    """Select options to unlock based on impact scores.

    Takes all options whose impact score is >= impact_threshold of the max,
    capped at max_batch_fraction of remaining options.

    Args:
        scores: Ranked ImpactScore list (descending by fitness_delta).
        remaining_count: Total number of remaining fixed options.
        max_batch_fraction: Max fraction of remaining to unlock.
        impact_threshold: Min fraction of top score to include.

    Returns:
        List of option names to unlock.
    """
    if not scores:
        return []

    max_score = scores[0].fitness_delta
    if max_score <= 0:
        return []

    threshold = max_score * impact_threshold
    max_batch = max(1, int(remaining_count * max_batch_fraction))

    selected: list[str] = []
    for score in scores:
        if len(selected) >= max_batch:
            break
        if score.fitness_delta >= threshold:
            selected.append(score.name)

    return selected


def _optimize_batch(
    search_space: SearchSpace,
    fitness_fn: FitnessFn,
    current_config: dict[str, Any],
    num_islands: int,
    population_size: int,
    num_workers: int,
    convergence_threshold: int = CONVERGENCE_THRESHOLD,
    debug: bool = False,
    tag: str = "",
    min_improvement_ratio: float = 0.001,
) -> OptimizationResult:
    """Optimize the currently mutable parameters in the search space.

    Splits by type: GA for integers, nevergrad for categoricals.
    Termination is convergence-based — no budget limits.

    Args:
        search_space: Current search space with mutable parameters.
        fitness_fn: Fitness evaluation function.
        current_config: Starting configuration.
        num_islands: GA island count.
        population_size: GA population size.
        num_workers: nevergrad worker count.
        convergence_threshold: Stop optimizer sub-runs early when no improvement.
        debug: Enable verbose output.
        tag: Tag used for debug output (e.g., phase name).
        min_improvement_ratio: Minimum improvement ratio for convergence.

    Returns:
        OptimizationResult with best config, fitness, and evaluations used.
    """
    mutable = search_space.mutable_parameters
    if not mutable:
        return OptimizationResult(
            best_config=copy.deepcopy(current_config),
            best_fitness=fitness_fn(copy.deepcopy(current_config)),
            evaluations_used=1,
        )

    # Split by type.
    integer_params = [p for p in mutable if p.param_type == "int"]
    categorical_params = [p for p in mutable if p.param_type != "int"]

    current = copy.deepcopy(current_config)
    best_fitness = fitness_fn(copy.deepcopy(current))
    total_evals = 1  # Initial fitness evaluation

    # Wrap fitness to count evaluations.
    def counting_fitness(cfg: dict[str, Any]) -> float:
        nonlocal total_evals
        total_evals += 1
        return fitness_fn(cfg)

    # Run GA for integer parameters.
    if integer_params:
        ga_subspace = _build_subspace(search_space, integer_params)

        best_ga = run_island_ga(
            initial_config=current,
            search_space=ga_subspace,
            fitness_fn=counting_fitness,
            num_islands=num_islands,
            population_size=population_size,
            num_iterations=None,
            debug=debug,
            convergence_threshold=convergence_threshold,
            tag=tag,
            min_improvement_ratio=min_improvement_ratio,
            num_workers=num_workers,
        )

        if best_ga.fitness < best_fitness:
            current = copy.deepcopy(best_ga.config)
            best_fitness = best_ga.fitness
            if debug:
                dbg(tag, f"GA improved fitness: {best_fitness}")

    # Run nevergrad for categorical parameters.
    if categorical_params:
        ng_subspace = _build_subspace(search_space, categorical_params)

        best_ng = run_nevergrad_optimization(
            search_space=ng_subspace,
            objective=fitness_fn,
            budget=NG_SAFETY_BUDGET,
            num_workers=num_workers,
            debug=debug,
            initial_config=current,
            convergence_threshold=convergence_threshold,
            tag=tag,
        )

        if best_ng.best_fitness < best_fitness:
            current = copy.deepcopy(best_ng.best_config)
            best_fitness = best_ng.best_fitness
            if debug:
                dbg(tag, f"nevergrad improved fitness: {best_fitness}")

    return OptimizationResult(
        best_config=current,
        best_fitness=best_fitness,
        evaluations_used=total_evals,
    )


def _build_subspace(
    parent_space: SearchSpace,
    params: list[ParameterDef],
) -> SearchSpace:
    """Build a SearchSpace containing only the given parameters."""
    active_names = {p.name for p in params}
    filtered: dict[str, ParameterDef] = {}
    for name, param in parent_space.parameters.items():
        if name in active_names:
            filtered[name] = param
        else:
            filtered[name] = ParameterDef(
                name=param.name,
                param_type=param.param_type,
                possible_values=[],
                fixed=True,
            )
    return SearchSpace(parameters=filtered)


def run_iterative_optimization(
    search_space: SearchSpace,
    fitness_fn: FitnessFn,
    initial_config: dict[str, Any],
    impact_fn: Callable[..., list[Any]],
    impact_kwargs: dict[str, Any],
    num_islands: int = 1,
    population_size: int = 4,
    num_workers: int = 1,
    max_batch_fraction: float = MAX_BATCH_FRACTION,
    impact_threshold: float = IMPACT_THRESHOLD,
    min_improvement_ratio: float = MIN_IMPROVEMENT_RATIO,
    convergence_threshold: int = CONVERGENCE_THRESHOLD,
    debug: bool = False,
) -> OptimizationResult:
    """Run the full iterative expansion optimization.

    Loop:
    1. Optimize currently mutable parameters.
    2. If remaining fixed options exist, measure their impact.
    3. Select top batch and unlock.
    4. Repeat until no impactful options remain or all options exhausted.

    Termination is convergence-based — no budget limits.

    Args:
        search_space: Initial search space (detected options mutable, rest fixed).
        fitness_fn: Fitness evaluation function.
        initial_config: Starting configuration.
        impact_fn: Function to measure impact of remaining options.
            Should accept (repo_path, candidate_names, base_options, lookups,
            current_config, ...) and return a list of ImpactScore.
        impact_kwargs: Keyword arguments to pass to impact_fn (repo_path,
            base_options, lookups, etc.).
        num_islands: GA island count.
        population_size: GA population size.
        num_workers: nevergrad worker count.
        max_batch_fraction: Max fraction of remaining to unlock per batch.
        impact_threshold: Min fraction of top impact score to include.
        min_improvement_ratio: Min ratio of fitness improvement to continue.
        convergence_threshold: Stop optimizer sub-runs early when no improvement
            occurs for this many consecutive generations/evaluations.
        debug: Enable verbose output.

    Returns:
        OptimizationResult with best config and fitness.
    """
    current_config = copy.deepcopy(initial_config)
    current_space = search_space
    best_fitness = fitness_fn(copy.deepcopy(current_config))

    iteration = 0
    if debug:
        dbg("iterative", "=== Iterative Expansion Optimization ===", summary=True)

    while True:
        iteration += 1
        remaining_fixed = current_space.remaining_fixed()
        total_mutable = len(current_space.mutable_parameters)

        if debug:
            dbg("optimize", f"--- Iteration {iteration} ---", summary=True)
            dbg("optimize", f"Mutable: {total_mutable}, Fixed: {len(remaining_fixed)}")

        # Optimize current mutable parameters.
        result = _optimize_batch(
            search_space=current_space,
            fitness_fn=fitness_fn,
            current_config=current_config,
            num_islands=num_islands,
            population_size=population_size,
            num_workers=num_workers,
            convergence_threshold=convergence_threshold,
            debug=debug,
            tag="optimize",
        )

        if result.best_fitness < best_fitness:
            current_config = copy.deepcopy(result.best_config)
            best_fitness = result.best_fitness

        # If no remaining fixed options, we're done.
        if not remaining_fixed:
            if debug:
                dbg("optimize", "No remaining fixed options. Done.")
            break

        # Measure impact of remaining options, excluding penalty parameters.
        # Penalties are relative weights — testing one in isolation is meaningless.
        penalty_names = [p.name for p in remaining_fixed if _is_penalty_option(p.name)]
        candidate_names = [
            p.name for p in remaining_fixed if not _is_penalty_option(p.name)
        ]
        if debug:
            dbg(
                "impact",
                f"Measuring impact of {len(candidate_names)} remaining options ({len(penalty_names)} penalties excluded)...",
            )

        scores = impact_fn(
            candidate_names=candidate_names,
            current_config=current_config,
            **impact_kwargs,
        )

        if not scores:
            if debug:
                dbg("impact", "No impactful options found. Done.")
            break

        if debug:
            top = min(5, len(scores))
            for s in scores[:top]:
                dbg("impact", f"{s.name}: delta={s.fitness_delta:.1f}")

        # Check if improvement is significant enough.
        if scores[0].fitness_delta < best_fitness * min_improvement_ratio:
            if debug:
                dbg(
                    "impact",
                    f"Top impact ({scores[0].fitness_delta:.1f}) below threshold. Done.",
                )
            break

        # Select batch to unlock.
        batch = _select_batch(
            scores,
            len(remaining_fixed),
            max_batch_fraction,
            impact_threshold,
        )
        if not batch:
            if debug:
                dbg("impact", "No options selected for batch. Done.")
            break

        if debug:
            dbg("expand", f"Unlocking {len(batch)} options: {batch}")

        # Unlock the batch.
        current_space = current_space.unlock(batch)

    # Final penalty polish: unlock all penalty options and optimize them together.
    # Penalties are relative weights, so they only make sense as a group.
    remaining_fixed = current_space.remaining_fixed()
    penalty_options = [p.name for p in remaining_fixed if _is_penalty_option(p.name)]
    if penalty_options:
        if debug:
            dbg("penalty-polish", f"({len(penalty_options)} penalties)", summary=True)
        current_space = current_space.unlock(penalty_options)
        result = _optimize_batch(
            search_space=current_space,
            fitness_fn=fitness_fn,
            current_config=current_config,
            num_islands=num_islands,
            population_size=population_size,
            num_workers=num_workers,
            convergence_threshold=convergence_threshold,
            debug=debug,
            tag="penalty-polish",
        )
        if result.best_fitness < best_fitness:
            current_config = copy.deepcopy(result.best_config)
            best_fitness = result.best_fitness

    # Final global polish with all mutable parameters.
    if current_space.mutable_parameters:
        if debug:
            dbg("global-polish", "---", summary=True)
        result = _optimize_batch(
            search_space=current_space,
            fitness_fn=fitness_fn,
            current_config=current_config,
            num_islands=num_islands,
            population_size=population_size,
            num_workers=num_workers,
            convergence_threshold=convergence_threshold,
            debug=debug,
            tag="global-polish",
        )
        if result.best_fitness < best_fitness:
            current_config = copy.deepcopy(result.best_config)
            best_fitness = result.best_fitness

    return OptimizationResult(
        best_config=current_config,
        best_fitness=best_fitness,
    )
