"""Iterative expansion optimizer.

Starts with analyzer-detected options, polishes them to convergence, then
empirically discovers which remaining fixed options are most impactful and
unlocks them in batches. Repeats until no impactful options remain or all
options are exhausted.

Each optimization step splits parameters by type: GA for integers,
nevergrad for categoricals/booleans (reuses _run_phase from phased).
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

# Maximum fraction of remaining options to unlock per iteration.
MAX_BATCH_FRACTION = 0.2

# Minimum fraction of top impact score to include in batch.
IMPACT_THRESHOLD = 0.5

# Minimum fitness improvement ratio to consider an iteration impactful.
MIN_IMPROVEMENT_RATIO = 0.01

# Maximum budget to spend on a single impact measurement.
MAX_IMPACT_BUDGET = 100

# Fraction of remaining budget to allocate for impact measurement.
IMPACT_BUDGET_FRACTION = 3

# Minimum budget to reserve for the next iteration.
MIN_NEXT_ITERATION_BUDGET = 20

# Minimum budget required to enter the optimization loop.
MIN_LOOP_BUDGET = 10

# Minimum budget to allocate for an optimizer (GA or nevergrad) sub-run.
MIN_OPT_SUB_BUDGET = 5

# Convergence threshold for GA and nevergrad sub-runs.
CONVERGENCE_THRESHOLD = 20

# Minimum budget required to attempt a penalty polish pass.
MIN_PENALTY_POLISH_BUDGET = 30


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
    budget: int,
    num_islands: int,
    population_size: int,
    num_workers: int,
    min_sub_budget: int = MIN_OPT_SUB_BUDGET,
    convergence_threshold: int = CONVERGENCE_THRESHOLD,
    debug: bool = False,
    debug_prefix: str = "",
    min_improvement_ratio: float = 0.001,
) -> OptimizationResult:
    """Optimize the currently mutable parameters in the search space.

    Splits by type: GA for integers, nevergrad for categoricals.

    Args:
        search_space: Current search space with mutable parameters.
        fitness_fn: Fitness evaluation function.
        current_config: Starting configuration.
        budget: Evaluation budget for this step.
        num_islands: GA island count.
        population_size: GA population size.
        num_workers: nevergrad worker count.
        debug: Enable verbose output.

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

    # Wrap fitness to count actual evaluations.
    eval_count = [0]

    def counting_fitness(cfg: dict[str, Any]) -> float:
        eval_count[0] += 1
        return fitness_fn(cfg)

    # Split by type.
    integer_params = [p for p in mutable if p.param_type == "int"]
    categorical_params = [p for p in mutable if p.param_type != "int"]

    # Split budget proportionally.
    active = (1 if integer_params else 0) + (1 if categorical_params else 0)
    ga_budget = budget // active if active > 0 else 0
    ng_budget = budget - ga_budget if categorical_params else 0

    # Ensure minimum budgets.
    if integer_params and ga_budget < min_sub_budget:
        ga_budget = min_sub_budget
        ng_budget = max(0, budget - ga_budget)
    if categorical_params and ng_budget < min_sub_budget:
        ng_budget = min_sub_budget
        ga_budget = max(0, budget - ng_budget)

    current = copy.deepcopy(current_config)
    best_fitness = counting_fitness(copy.deepcopy(current))

    # Run GA for integer parameters.
    if integer_params:
        ga_subspace = _build_subspace(search_space, integer_params)
        ga_iterations = max(5, ga_budget // max(1, population_size))

        best_ga = run_island_ga(
            initial_config=current,
            search_space=ga_subspace,
            fitness_fn=counting_fitness,
            num_islands=num_islands,
            population_size=population_size,
            num_iterations=ga_iterations,
            debug=debug,
            budget=ga_budget,
            convergence_threshold=convergence_threshold,
            debug_prefix=debug_prefix,
            min_improvement_ratio=min_improvement_ratio,
            num_workers=num_workers,
        )

        if best_ga.fitness < best_fitness:
            current = copy.deepcopy(best_ga.config)
            best_fitness = best_ga.fitness
            if debug:
                print(f"  GA improved fitness: {best_fitness}", file=sys.stderr)

    # Run nevergrad for categorical parameters.
    if categorical_params:
        ng_subspace = _build_subspace(search_space, categorical_params)

        best_ng = run_nevergrad_optimization(
            search_space=ng_subspace,
            objective=counting_fitness,
            budget=ng_budget,
            num_workers=num_workers,
            debug=debug,
            initial_config=current,
            convergence_threshold=convergence_threshold,
            debug_prefix=debug_prefix,
        )

        if best_ng.best_fitness < best_fitness:
            current = copy.deepcopy(best_ng.best_config)
            best_fitness = best_ng.best_fitness
            if debug:
                print(
                    f"  nevergrad improved fitness: {best_fitness}",
                    file=sys.stderr,
                )

    return OptimizationResult(
        best_config=current,
        best_fitness=best_fitness,
        evaluations_used=eval_count[0],
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
    total_budget: int,
    impact_fn: Callable[..., list[Any]],
    impact_kwargs: dict[str, Any],
    num_islands: int = 1,
    population_size: int = 4,
    num_workers: int = 1,
    max_batch_fraction: float = MAX_BATCH_FRACTION,
    impact_threshold: float = IMPACT_THRESHOLD,
    min_improvement_ratio: float = MIN_IMPROVEMENT_RATIO,
    max_impact_budget: int = MAX_IMPACT_BUDGET,
    impact_budget_fraction: int = IMPACT_BUDGET_FRACTION,
    min_next_iteration_budget: int = MIN_NEXT_ITERATION_BUDGET,
    min_loop_budget: int = MIN_LOOP_BUDGET,
    min_opt_sub_budget: int = MIN_OPT_SUB_BUDGET,
    min_penalty_polish_budget: int = MIN_PENALTY_POLISH_BUDGET,
    convergence_threshold: int = CONVERGENCE_THRESHOLD,
    debug: bool = False,
) -> OptimizationResult:
    """Run the full iterative expansion optimization.

    Loop:
    1. Optimize currently mutable parameters.
    2. If remaining fixed options exist, measure their impact.
    3. Select top batch and unlock.
    4. Repeat until no impactful options or budget exhausted.

    Args:
        search_space: Initial search space (detected options mutable, rest fixed).
        fitness_fn: Fitness evaluation function.
        initial_config: Starting configuration.
        total_budget: Total evaluation budget across all iterations.
        impact_fn: Function to measure impact of remaining options.
            Should accept (repo_path, candidate_names, base_options, lookups,
            current_config, budget, ...) and return a list of ImpactScore.
        impact_kwargs: Keyword arguments to pass to impact_fn (repo_path,
            base_options, lookups, etc.).
        num_islands: GA island count.
        population_size: GA population size.
        num_workers: nevergrad worker count.
        max_batch_fraction: Max fraction of remaining to unlock per batch.
        impact_threshold: Min fraction of top impact score to include.
        min_improvement_ratio: Min ratio of fitness improvement to continue.
        max_impact_budget: Max budget for a single impact measurement.
        impact_budget_fraction: Divisor for impact budget allocation.
        min_next_iteration_budget: Budget to reserve for next iteration.
        min_loop_budget: Minimum budget to continue the loop.
        min_opt_sub_budget: Minimum budget per optimizer sub-run.
        min_penalty_polish_budget: Minimum budget required to attempt a final
            penalty polish pass.
        convergence_threshold: Stop optimizer sub-runs early when no improvement
            occurs for this many consecutive generations/evaluations.
        debug: Enable verbose output.

    Returns:
        OptimizationResult with best config and fitness.
    """
    current_config = copy.deepcopy(initial_config)
    current_space = search_space
    best_fitness = fitness_fn(copy.deepcopy(current_config))
    remaining_budget = total_budget

    iteration = 0
    if debug:
        print("\n=== Iterative Expansion Optimization ===", file=sys.stderr)

    while remaining_budget >= min_loop_budget:
        iteration += 1
        remaining_fixed = current_space.remaining_fixed()
        total_mutable = len(current_space.mutable_parameters)

        if debug:
            print(
                f"\n--- [optimize] Iteration {iteration} ---",
                file=sys.stderr,
            )
            print(
                f"  Mutable: {total_mutable}, Fixed: {len(remaining_fixed)}, Budget remaining: {remaining_budget}",
                file=sys.stderr,
            )

        # Allocate budget: reserve for impact measurement and next iteration.
        # Scale impact budget with the number of candidates (at least 2 evals each).
        num_candidates = len(remaining_fixed)
        impact_budget = min(
            max_impact_budget,
            max(num_candidates * 2, remaining_budget // impact_budget_fraction),
        )
        # Cap opt budget to leave room for impact + next iteration.
        opt_reserved = impact_budget + min_next_iteration_budget
        opt_budget = max(min_loop_budget, remaining_budget - opt_reserved)
        # Ensure opt budget doesn't consume more than 60% of remaining.
        opt_budget = min(opt_budget, int(remaining_budget * 0.6))

        # Optimize current mutable parameters.
        result = _optimize_batch(
            search_space=current_space,
            fitness_fn=fitness_fn,
            current_config=current_config,
            budget=opt_budget,
            num_islands=num_islands,
            population_size=population_size,
            num_workers=num_workers,
            min_sub_budget=min_opt_sub_budget,
            convergence_threshold=convergence_threshold,
            debug=debug,
            debug_prefix="[optimize] ",
        )

        if result.best_fitness < best_fitness:
            current_config = copy.deepcopy(result.best_config)
            best_fitness = result.best_fitness

        # Deduct actual evaluations used, not the allocated budget.
        remaining_budget -= result.evaluations_used

        # If no remaining fixed options, we're done.
        if not remaining_fixed:
            if debug:
                print("  [optimize] No remaining fixed options. Done.", file=sys.stderr)
            break

        # Measure impact of remaining options, excluding penalty parameters.
        # Penalties are relative weights — testing one in isolation is meaningless.
        penalty_names = [p.name for p in remaining_fixed if _is_penalty_option(p.name)]
        candidate_names = [
            p.name for p in remaining_fixed if not _is_penalty_option(p.name)
        ]
        if debug:
            msg = (
                f"  [impact] Measuring impact of {len(candidate_names)} remaining options "
                f"({len(penalty_names)} penalties excluded)..."
            )
            print(msg, file=sys.stderr)

        scores = impact_fn(
            candidate_names=candidate_names,
            current_config=current_config,
            budget=impact_budget,
            **impact_kwargs,
        )

        remaining_budget -= impact_budget

        if not scores:
            if debug:
                print("  [impact] No impactful options found. Done.", file=sys.stderr)
            break

        if debug:
            top = min(5, len(scores))
            for s in scores[:top]:
                print(
                    f"  [impact]   {s.name}: delta={s.fitness_delta:.1f}",
                    file=sys.stderr,
                )

        # Check if improvement is significant enough.
        if scores[0].fitness_delta < best_fitness * min_improvement_ratio:
            if debug:
                print(
                    f"  [impact] Top impact ({scores[0].fitness_delta:.1f}) below threshold. Done.",
                    file=sys.stderr,
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
                print(
                    "  [impact] No options selected for batch. Done.", file=sys.stderr
                )
            break

        if debug:
            print(
                f"  [expand] Unlocking {len(batch)} options: {batch}", file=sys.stderr
            )

        # Unlock the batch.
        current_space = current_space.unlock(batch)

    # Final penalty polish: unlock all penalty options and optimize them together.
    # Penalties are relative weights, so they only make sense as a group.
    remaining_fixed = current_space.remaining_fixed()
    penalty_options = [p.name for p in remaining_fixed if _is_penalty_option(p.name)]
    if penalty_options and remaining_budget >= min_penalty_polish_budget:
        if debug:
            msg = (
                f"\n--- [penalty-polish] ({len(penalty_options)} penalties, "
                f"budget={remaining_budget}) ---"
            )
            print(msg, file=sys.stderr)
        current_space = current_space.unlock(penalty_options)
        result = _optimize_batch(
            search_space=current_space,
            fitness_fn=fitness_fn,
            current_config=current_config,
            budget=remaining_budget,
            num_islands=num_islands,
            population_size=population_size,
            num_workers=num_workers,
            min_sub_budget=min_opt_sub_budget,
            convergence_threshold=convergence_threshold,
            debug=debug,
            debug_prefix="[penalty-polish] ",
        )
        if result.best_fitness < best_fitness:
            current_config = copy.deepcopy(result.best_config)
            best_fitness = result.best_fitness

    # Final global polish with remaining budget.
    if remaining_budget >= min_loop_budget and current_space.mutable_parameters:
        if debug:
            print(
                f"\n--- [global-polish] (budget={remaining_budget}) ---",
                file=sys.stderr,
            )
        result = _optimize_batch(
            search_space=current_space,
            fitness_fn=fitness_fn,
            current_config=current_config,
            budget=remaining_budget,
            num_islands=num_islands,
            population_size=population_size,
            num_workers=num_workers,
            min_sub_budget=min_opt_sub_budget,
            convergence_threshold=convergence_threshold,
            debug=debug,
            debug_prefix="[global-polish] ",
        )
        if result.best_fitness < best_fitness:
            current_config = copy.deepcopy(result.best_config)
            best_fitness = result.best_fitness

    return OptimizationResult(
        best_config=current_config,
        best_fitness=best_fitness,
    )
