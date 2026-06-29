"""Generic island-model genetic algorithm.

No domain-specific dependencies. The caller provides:
- A SearchSpace defining parameters and their possible values
- A fitness function that scores a candidate configuration
"""

from __future__ import annotations

import concurrent.futures
import copy
import random
from typing import Any, Callable

from ..utils import dbg
from .types import Individual, ParameterDef, SearchSpace

# Fitness function signature: takes a config dict, returns a float (lower is better).
FitnessFn = Callable[[dict[str, Any]], float]


def crossover(
    parent1_config: dict[str, Any],
    parent2_config: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    """Uniform crossover: each key is randomly picked from one parent."""
    child_config = {}
    for key in parent1_config:
        child_config[key] = copy.deepcopy(
            parent1_config[key] if rng.random() < 0.5 else parent2_config[key]
        )
    return child_config


def _get_mutable_options(
    config: dict[str, Any],
    search_space: SearchSpace,
) -> list[ParameterDef]:
    """Return mutable parameters whose name exists in config."""
    return [p for p in search_space.mutable_parameters if p.name in config]


def mutate(
    individual: Individual,
    search_space: SearchSpace,
    fitness_fn: FitnessFn,
    rng: random.Random,
    debug: bool = False,
    tag: str = "",
) -> Individual:
    """Mutate one random mutable parameter by exhaustively testing all values.

    Returns a new Individual with the best value found.
    """
    mutable = _get_mutable_options(individual.config, search_space)
    if not mutable:
        dbg(tag, "No mutable options for mutation.")
        return Individual(
            config=copy.deepcopy(individual.config),
            fitness=float("inf"),
        )

    param = rng.choice(mutable)
    if debug:
        current = individual.config.get(param.name)
        dbg(tag, f"Mutating '{param.name}' (current: {current})...")

    best_fitness = float("inf")
    best_values: list[Any] = []

    for value in param.possible_values:
        config_copy = copy.deepcopy(individual.config)
        config_copy[param.name] = value
        fitness = fitness_fn(config_copy)

        if debug:
            dbg(tag, f"  '{param.name}'={value} -> fitness: {fitness}")

        if fitness < best_fitness:
            best_fitness = fitness
            best_values = [value]
        elif fitness == best_fitness:
            best_values.append(value)

    best_value = rng.choice(best_values)
    mutated_config = copy.deepcopy(individual.config)
    mutated_config[param.name] = best_value

    if debug:
        dbg(
            tag,
            f"Best value for '{param.name}': {best_value} (fitness: {best_fitness})",
        )

    return Individual(config=mutated_config, fitness=best_fitness)


def evolve_island_generation(
    population: list[Individual],
    island_size: int,
    search_space: SearchSpace,
    fitness_fn: FitnessFn,
    rng: random.Random,
    debug: bool = False,
    tag: str = "",
) -> list[Individual]:
    """Evolve one island for one generation.

    Steps: elitism, crossover, mutation, selection.
    Returns a new population of `island_size` individuals.
    """
    if not population:
        return []

    # Elitism: keep the best individual
    best = min(population, key=lambda ind: ind.fitness)
    new_candidates: list[Individual] = [best]

    num_to_generate = max(0, island_size - len(new_candidates))

    for _ in range(num_to_generate):
        if len(population) < 2:
            child_config = copy.deepcopy(population[0].config)
            if debug:
                dbg(
                    tag,
                    "Warning: population too small for crossover. Mutating copy of best.",
                )
        else:
            p1 = rng.choice(population).config
            p2 = rng.choice(population).config
            child_config = crossover(p1, p2, rng)

        child = Individual(config=child_config)
        mutated = mutate(child, search_space, fitness_fn, rng, debug, tag)
        new_candidates.append(mutated)

    # Selection: keep the best `island_size`
    new_candidates.sort(key=lambda ind: ind.fitness)
    return new_candidates[:island_size]


def perform_migration(
    populations: list[list[Individual]],
    rng: random.Random,
    debug: bool = False,
    tag: str = "ga",
) -> None:
    """Migrate best individuals between islands.

    Each island sends its best individual to a random other island,
    replacing a random individual there.
    """
    if len(populations) < 2:
        dbg(tag, "Skipping migration: less than 2 islands.")
        return

    migrants: list[tuple[int, Individual]] = []
    for i, pop in enumerate(populations):
        if pop:
            best = min(pop, key=lambda ind: ind.fitness)
            migrants.append((i, best))
        elif debug:
            dbg(tag, f"Warning: island {i} is empty, cannot select migrant.")

    if debug:
        dbg(tag, f"Performing migration with {len(migrants)} migrants.")

    for source_idx, migrant in migrants:
        target_idx = source_idx
        while target_idx == source_idx:
            target_idx = rng.randrange(len(populations))

        target_pop = populations[target_idx]
        if not target_pop:
            target_pop.append(migrant)
            if debug:
                dbg(
                    tag,
                    f"  Migrant from island {source_idx} added to empty island {target_idx}.",
                )
        else:
            to_replace = rng.choice(target_pop)
            target_pop.remove(to_replace)
            target_pop.append(migrant)
            if debug:
                dbg(
                    tag,
                    f"  Migrant from island {source_idx} replaced an individual in island {target_idx}.",
                )


def _evolve_island_task(
    island_idx: int,
    population: list[Individual],
    island_size: int,
    search_space: SearchSpace,
    fitness_fn: FitnessFn,
    rng: random.Random,
    debug: bool,
    tag: str,
) -> tuple[int, list[Individual], float]:
    """Evolve one island and return (index, new_population, best_fitness)."""
    island_tag = f"{tag}/island-{island_idx}"
    new_pop = evolve_island_generation(
        population, island_size, search_space, fitness_fn, rng, debug, island_tag
    )
    best_fitness = min(ind.fitness for ind in new_pop) if new_pop else float("inf")
    return island_idx, new_pop, best_fitness


# Maximum number of mutable parameters to randomize per individual during
# initial population diversity injection. The effective diversity rate is
# capped at MAX_MUTABLE_RANDOMIZED / num_mutable_params so that repos with
# many options don't end up randomizing half their config.
MAX_MUTABLE_RANDOMIZED = 3


def _effective_diversity_rate(
    diversity_rate: float,
    num_mutable: int,
) -> float:
    """Compute an effective diversity rate that scales with problem size.

    Caps the rate so that at most MAX_MUTABLE_RANDOMIZED parameters are
    randomized per individual, regardless of how many mutable parameters
    exist. This prevents excessive noise when the search space is large.

    Args:
        diversity_rate: User-provided base rate (0.0-1.0).
        num_mutable: Number of mutable parameters in the search space.

    Returns:
        Capped diversity rate.
    """
    if num_mutable <= 0:
        return 0.0
    return min(diversity_rate, MAX_MUTABLE_RANDOMIZED / num_mutable)


def _initialize_population(
    initial_config: dict[str, Any],
    search_space: SearchSpace,
    fitness_fn: FitnessFn,
    island_size: int,
    rng: random.Random,
    diversity_rate: float = 0.5,
) -> list[Individual]:
    """Create a diverse initial population for one island.

    Individual 0 is an exact copy of *initial_config* (anchor baseline).
    Each subsequent individual randomizes a fraction of mutable parameters
    from their *possible_values*, keeping the rest from *initial_config*.

    The effective diversity rate is capped so that at most
    MAX_MUTABLE_RANDOMIZED parameters are randomized per individual,
    preventing excessive noise when the search space is large.

    Args:
        initial_config: Starting configuration.
        search_space: Defines mutable parameters and their possible values.
        fitness_fn: Fitness evaluation function.
        island_size: Number of individuals to create.
        rng: Random number generator for reproducibility.
        diversity_rate: Base fraction of mutable parameters to randomize per
            individual (0.0 = all identical, 1.0 = all randomized).
            Actually capped by MAX_MUTABLE_RANDOMIZED.

    Returns:
        List of individuals.
    """
    mutable = _get_mutable_options(initial_config, search_space)
    effective_rate = _effective_diversity_rate(diversity_rate, len(mutable))
    pop: list[Individual] = []

    for i in range(island_size):
        if i == 0 or not mutable or effective_rate <= 0.0:
            # Anchor: exact copy of initial config.
            config = copy.deepcopy(initial_config)
        else:
            config = copy.deepcopy(initial_config)
            for param in mutable:
                if rng.random() < effective_rate:
                    config[param.name] = rng.choice(param.possible_values)

        fitness = fitness_fn(config)
        pop.append(Individual(config=config, fitness=fitness))

    return pop


def run_island_ga(  # noqa: PLR0913
    initial_config: dict[str, Any],
    search_space: SearchSpace,
    fitness_fn: FitnessFn,
    num_islands: int,
    population_size: int,
    num_iterations: int | None,
    migration_interval: int = 15,
    debug: bool = False,
    random_seed: int | None = None,
    convergence_threshold: int | None = None,
    tag: str = "",
    min_improvement_ratio: float = 0.001,
    num_workers: int = 1,
    diversity_rate: float = 0.5,
) -> Individual:
    """Run the full island-model GA.

    Args:
        initial_config: Starting configuration (param_name -> value).
        search_space: Defines parameters and their possible values.
        fitness_fn: Callable that scores a config dict (lower is better).
        num_islands: Number of parallel islands.
        population_size: Total population across all islands.
        num_iterations: Number of generations to evolve. None means run until
            convergence (requires convergence_threshold to be set).
        migration_interval: Generations between migrations.
        debug: Print verbose output.
        random_seed: Seed for reproducibility.
        convergence_threshold: If set, stop early when no improvement occurs
            for this many consecutive generations. None means run full iterations.
        tag: Tag used for debug output (e.g., phase name).
        min_improvement_ratio: Minimum fractional improvement over current best
            to reset the convergence counter. Improvements smaller than this
            threshold are treated as noise and do not reset the counter.
        num_workers: Maximum number of islands to evolve in parallel per
            generation. Set to 1 for sequential execution (default). Values
            greater than 1 use ThreadPoolExecutor to parallelize island evolution.
        diversity_rate: Fraction of mutable parameters to randomize in the
            initial population per individual (0.0 = all identical clones,
            1.0 = all randomized). Default 0.5 gives a balance of exploration
            and exploitation from generation 0.

    Returns:
        The best individual found.
    """
    # Safety cap: if num_iterations is None, use a large default.
    # Convergence is the real limiter; this is a runaway guard.
    if num_iterations is None:  # pragma: no cover -- always set by caller
        num_iterations = 10_000

    rng = random.Random(random_seed)

    num_islands = max(1, num_islands)
    island_size = max(5, population_size // num_islands)
    if island_size * num_islands > population_size:
        population_size = island_size * num_islands

    # Initialize diverse populations -- each island gets unique starting individuals.
    populations: list[list[Individual]] = []
    best_init_fitness = float("inf")
    best_init_config: dict[str, Any] | None = None

    for _ in range(num_islands):
        pop = _initialize_population(
            initial_config, search_space, fitness_fn, island_size, rng, diversity_rate
        )
        populations.append(pop)
        for ind in pop:
            if ind.fitness < best_init_fitness:
                best_init_fitness = ind.fitness
                best_init_config = copy.deepcopy(ind.config)

    # island_size is always >= 5, so best_init_config is guaranteed non-None.
    assert best_init_config is not None
    best_overall = Individual(
        config=best_init_config,
        fitness=best_init_fitness,
    )

    fitness_history: list[list[float]] = [
        [min(ind.fitness for ind in pop)] for pop in populations
    ]
    no_improve_count = 0

    max_workers = min(num_workers, num_islands)
    executor: concurrent.futures.ThreadPoolExecutor | None = None
    if max_workers > 1:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    try:
        for iteration in range(num_iterations):
            prev_best = best_overall.fitness

            # Seed per-island RNGs from the main RNG to ensure deterministic behavior.
            # Each island gets its own RNG instance to avoid thread-safety issues.
            island_seeds = [rng.random() for _ in range(num_islands)]
            island_rngs = [random.Random(s) for s in island_seeds]

            if executor is not None:
                futures = {
                    executor.submit(
                        _evolve_island_task,
                        i,
                        pop,
                        island_size,
                        search_space,
                        fitness_fn,
                        island_rngs[i],
                        debug,
                        tag,
                    ): i
                    for i, pop in enumerate(populations)
                }
                results = []
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())

                # Update populations and track best
                for idx, new_pop, _ in results:
                    populations[idx] = new_pop
                    best_in_island = min(new_pop, key=lambda ind: ind.fitness)
                    fitness_history[idx].append(best_in_island.fitness)
                    if best_in_island.fitness < best_overall.fitness:
                        best_overall = Individual(
                            config=copy.deepcopy(best_in_island.config),
                            fitness=best_in_island.fitness,
                        )
            else:
                # Sequential execution
                for i, pop in enumerate(populations):
                    new_pop = evolve_island_generation(
                        pop,
                        island_size,
                        search_space,
                        fitness_fn,
                        island_rngs[i],
                        debug,
                        tag,
                    )
                    populations[i] = new_pop

                    best_in_island = min(new_pop, key=lambda ind: ind.fitness)
                    fitness_history[i].append(best_in_island.fitness)

                    if best_in_island.fitness < best_overall.fitness:
                        best_overall = Individual(
                            config=copy.deepcopy(best_in_island.config),
                            fitness=best_in_island.fitness,
                        )

            # Convergence detection
            if best_overall.fitness == prev_best:
                no_improve_count += 1
            else:
                improvement = (prev_best - best_overall.fitness) / max(
                    abs(prev_best), 1
                )
                if improvement < min_improvement_ratio:
                    no_improve_count += 1
                else:
                    no_improve_count = 0

            # Single compact debug line per iteration: stage, iteration, fitness, convergence.
            if debug:
                conv_str = (
                    f" conv {no_improve_count}/{convergence_threshold}"
                    if convergence_threshold is not None
                    else ""
                )
                dbg(
                    tag,
                    f"Iter {iteration + 1} fitness={best_overall.fitness}{conv_str}",
                    summary=True,
                )

            # Early termination on perfect fitness
            if best_overall.fitness == 0:
                break

            if (
                convergence_threshold is not None
                and no_improve_count >= convergence_threshold
            ):
                if debug:
                    dbg(tag, f"Converged after {iteration + 1} iterations.")
                break

            # Migration
            if num_islands > 1 and (iteration + 1) % migration_interval == 0:
                perform_migration(populations, rng, debug, tag)

                # Re-check overall best after migration
                all_inds = [ind for pop in populations for ind in pop]
                if all_inds:
                    post_migration_best = min(all_inds, key=lambda ind: ind.fitness)
                    if (
                        post_migration_best.fitness < best_overall.fitness
                    ):  # pragma: no cover - migration only moves existing individuals; best_overall already tracks global minimum
                        best_overall = Individual(
                            config=copy.deepcopy(post_migration_best.config),
                            fitness=post_migration_best.fitness,
                        )
    finally:
        if executor:
            executor.shutdown(wait=True)

    return best_overall
