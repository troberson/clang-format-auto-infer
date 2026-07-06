"""Iterative expansion optimizer.

Flow:
1. Analyzer fixes deterministic values.
2. Radial search processes integer options from default outward, fixing best.
3. Impact scan measures all remaining options once.
4. Slide a window over pre-computed impact scores.
5. Select batch from window and optimize with GA/nevergrad.
6. Advance window by batch size and repeat until no impactful options remain.
7. NG penalty polish for penalty options as a group.

Termination is convergence-based -- no budget limits.
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

# Minimum fitness improvement ratio to consider an iteration impactful.
MIN_IMPROVEMENT_RATIO = 0.01

# Convergence threshold for GA and nevergrad sub-runs.
CONVERGENCE_THRESHOLD = 20

# Safety cap for nevergrad evaluations. Convergence fires first in practice.
NG_SAFETY_BUDGET = 10_000

# Batch selection limits (used by _select_batch, kept for testing).
MAX_BATCH_FRACTION = 0.2
IMPACT_THRESHOLD = 0.5

# Options that only apply to non-Cpp languages.
# clang-format has options for Java, JavaScript, ObjC, Verilog, etc.
# that are meaningless for C/C++ projects and should be excluded.
NON_CPP_OPTIONS: set[str] = {
    "SortJavaStaticImport",
    "JavaScriptQuotes",
    "JavaScriptWrapImports",
    "InsertTrailingCommas",
    "ObjCBinPackProtocolList",
    "ObjCBlockIndentWidth",
    "ObjCBreakBeforeNestedBlockParam",
    "ObjCSpaceAfterProperty",
    "ObjCSpaceBeforeProtocolList",
    "VerilogBreakBetweenInstancePorts",
}


def _is_penalty_option(name: str) -> bool:
    """Return True if the option is a clang-format penalty parameter.

    Penalty options are relative weights that only make sense when optimized
    together. Testing them individually for impact is meaningless.
    """
    return name.startswith("Penalty")


def _select_batch(  # pyright: ignore[reportUnusedFunction]
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
    Termination is convergence-based -- no budget limits.

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
    def counting_fitness(
        cfg: dict[str, Any],
    ) -> float:  # pragma: no cover -- mocked in tests
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


def _radial_search_integers(
    search_space: SearchSpace,
    fitness_fn: FitnessFn,
    current_config: dict[str, Any],
    debug: bool = False,
) -> tuple[dict[str, Any], SearchSpace]:
    """Radial search for integer options from default outward, fixing the best.

    For each mutable integer option, tests values in radial order around
    the default. The further from default, the more friction -- we stop
    when fitness stops improving. The best value is then fixed in the
    search space, reducing the search space for subsequent stages.

    Args:
        search_space: Current search space with mutable integer parameters.
        fitness_fn: Fitness evaluation function.
        current_config: Starting configuration.
        debug: Enable verbose output.

    Returns:
        Tuple of (updated config, updated search space with integers fixed).
    """
    mutable_ints = [p for p in search_space.mutable_parameters if p.param_type == "int"]
    if not mutable_ints:
        return current_config, search_space

    if debug:
        dbg(
            "radial-search",
            f"Scanning {len(mutable_ints)} integer options...",
            summary=True,
        )

    config = copy.deepcopy(current_config)
    improved = True
    while improved:
        improved = False
        for param in mutable_ints:
            if (
                param.fixed
            ):  # pragma: no cover -- radial search only sees mutable params
                continue
            values = param.possible_values
            if not values:  # pragma: no cover -- mutable params always have values
                continue

            # Get default value from config or base.
            default = config.get(param.name)
            if default is None:  # pragma: no cover -- defensive fallback
                default = values[0] if values else None
            if default is None:  # pragma: no cover -- defensive fallback
                continue

            # Sort values by distance from default (radial order).
            assert default is not None
            default_int = int(default)
            sorted_values = sorted(values, key=lambda v: abs(int(v) - default_int))

            best_val = default
            best_fit = fitness_fn(copy.deepcopy(config))

            for val in sorted_values:
                if val == default:
                    continue
                config[param.name] = val
                fit = fitness_fn(copy.deepcopy(config))
                if (
                    fit < best_fit
                ):  # pragma: no cover -- hard to trigger with mock fitness
                    best_fit = fit
                    best_val = val
                    improved = True
                else:
                    # Stop early if fitness worsens -- further values are worse.
                    break

            # Restore best value.
            config[param.name] = best_val
            if debug:
                dbg(
                    "radial-search",
                    f"  {param.name}: {default} -> {best_val} (fitness: {best_fit})",
                )

    # Fix all scanned integers in the search space.
    new_params: dict[str, ParameterDef] = {}
    scanned_names = {p.name for p in mutable_ints}
    for name, param in search_space.parameters.items():
        if name in scanned_names and not param.fixed:
            new_params[name] = ParameterDef(
                name=param.name,
                param_type=param.param_type,
                possible_values=[],
                fixed=True,
                tier=param.tier,
                confidence="detected",  # Radial search detected the best value.
            )
        else:
            new_params[name] = param

    return config, SearchSpace(parameters=new_params)


def _get_impact_candidates(
    remaining_fixed: list[ParameterDef],
) -> tuple[list[str], list[str], list[str]]:
    """Filter remaining fixed options into impact candidates.

    Excludes penalty parameters (relative weights, meaningless individually),
    forced options (certain, must not change), detected options
    (analyzer found them deterministically), and non-Cpp options
    (language-specific options that don't apply to C/C++ projects).

    Args:
        remaining_fixed: List of currently fixed parameters.

    Returns:
        Tuple of (candidate_names, penalty_names, excluded_names).
    """
    penalty_names = [p.name for p in remaining_fixed if _is_penalty_option(p.name)]
    excluded_confidences = {"forced", "detected"}
    excluded_names = [
        p.name
        for p in remaining_fixed
        if p.confidence in excluded_confidences or p.name in NON_CPP_OPTIONS
    ]
    candidate_names = [
        p.name
        for p in remaining_fixed
        if not _is_penalty_option(p.name)
        and p.confidence not in excluded_confidences
        and p.name not in NON_CPP_OPTIONS
    ]
    return candidate_names, penalty_names, excluded_names


def run_iterative_optimization(
    search_space: SearchSpace,
    fitness_fn: FitnessFn,
    initial_config: dict[str, Any],
    impact_fn: Callable[..., list[Any]],
    impact_kwargs: dict[str, Any],
    num_islands: int = 1,
    population_size: int = 4,
    num_workers: int = 1,
    min_improvement_ratio: float = MIN_IMPROVEMENT_RATIO,
    convergence_threshold: int = CONVERGENCE_THRESHOLD,
    debug: bool = False,
) -> OptimizationResult:
    """Run the full iterative expansion optimization.

    Loop:
    1. Radial search integer options from default outward, fix best.
    2. Measure impact of remaining fixed options.
    3. Select top impactful batch and unlock.
    4. Optimize unlocked batch.
    5. Repeat from 2 until no impactful options remain.
    6. Final penalty polish.

    Termination is convergence-based -- no budget limits.

    Args:
        search_space: Initial search space (detected options fixed, rest mutable).
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

    if debug:
        dbg("iterative", "=== Iterative Expansion Optimization ===", summary=True)

    # Stage 1: Radial search integer options.
    current_config, current_space = _radial_search_integers(
        search_space=current_space,
        fitness_fn=fitness_fn,
        current_config=current_config,
        debug=debug,
    )
    radial_fitness = fitness_fn(copy.deepcopy(current_config))
    if (
        radial_fitness < best_fitness
    ):  # pragma: no cover -- hard to trigger with mock fitness
        best_fitness = radial_fitness

    # Fix penalty options so they don't participate in the main loop.
    # They are relative weights that only make sense when optimized together.
    penalty_names = [
        p.name for p in current_space.mutable_parameters if _is_penalty_option(p.name)
    ]
    if penalty_names:  # pragma: no cover -- penalty fix debug path
        current_space = current_space.fix(penalty_names)
        if debug:
            dbg(
                "iterative",
                f"Fixed {len(penalty_names)} penalty options for final polish.",
                summary=True,
            )

    # Stage 2: Compute impact once for all remaining mutable options.
    remaining_mutable = current_space.mutable_parameters
    candidate_names, penalty_names, excluded_names = _get_impact_candidates(
        remaining_mutable
    )
    if debug:
        dbg(
            "impact",
            f"Measuring impact of {len(candidate_names)} remaining options ({len(penalty_names)} penalties, {len(excluded_names)} forced/detected excluded)...",
        )

    all_impact_scores: list[Any] = []
    if candidate_names:
        all_impact_scores = impact_fn(
            candidate_names=candidate_names,
            current_config=current_config,
            **impact_kwargs,
        )

    # Track all options that showed any impact for global polish.
    impacted_options: list[str] = [s.name for s in all_impact_scores]

    if debug and all_impact_scores:
        top = min(5, len(all_impact_scores))
        for s in all_impact_scores[:top]:
            dbg("impact", f"{s.name}: delta={s.fitness_delta:.1f}")

    if not all_impact_scores:
        if debug:
            dbg("impact", "No impactful options found. Done.")
    else:
        # Stage 3-5: Optimize all impactful options in one batch.
        # The impact scan already filtered to options that matter, so we
        # let the optimizer find the best combination of all of them.
        batch_names = [s.name for s in all_impact_scores]

        # Check if top impact is significant enough.
        if all_impact_scores[0].fitness_delta < best_fitness * min_improvement_ratio:
            if debug:
                dbg(
                    "impact",
                    f"Top impact ({all_impact_scores[0].fitness_delta:.1f}) below threshold. Done.",
                )
        else:
            if debug:
                dbg(
                    "iterative",
                    f"--- Iteration 1 --- Batch: {len(batch_names)} impactful options",
                    summary=True,
                )
                dbg("expand", f"Optimizing batch: {batch_names}")

            # Optimize the batch.
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

            # Fix the optimized batch.
            current_space = current_space.fix(batch_names)

    # Stage 6: Final penalty polish.
    # Penalties were fixed at the start, so we unlock them for group optimization.
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

        # Re-fix penalties so global-polish only refines impacted options.
        current_space = current_space.fix(penalty_options)

    # Stage 7: Global polish.
    # Unlock all options that showed non-zero impact and let nevergrad
    # refine their interactions. Starts from the best config found so far.
    if impacted_options:
        if debug:
            dbg(
                "global-polish",
                f"({len(impacted_options)} impactful options)",
                summary=True,
            )
        current_space = current_space.unlock(impacted_options)
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
