"""Generic island-model genetic algorithm.

No domain-specific dependencies. The caller provides:
- A SearchSpace defining parameters and their possible values
- A fitness function that scores a candidate configuration
"""

from __future__ import annotations

import copy
import random
import sys
from typing import Any, Callable

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
    debug_prefix: str = "",
) -> Individual:
    """Mutate one random mutable parameter by exhaustively testing all values.

    Returns a new Individual with the best value found.
    """
    mutable = _get_mutable_options(individual.config, search_space)
    if not mutable:
        if debug:
            print(
                f"{debug_prefix}No mutable options for mutation.",
                file=sys.stderr,
            )
        return Individual(
            config=copy.deepcopy(individual.config),
            fitness=float("inf"),
        )

    param = rng.choice(mutable)
    if debug:
        current = individual.config.get(param.name)
        print(
            f"{debug_prefix}Mutating '{param.name}' (current: {current})...",
            file=sys.stderr,
        )

    best_fitness = float("inf")
    best_values: list[Any] = []

    for value in param.possible_values:
        config_copy = copy.deepcopy(individual.config)
        config_copy[param.name] = value
        fitness = fitness_fn(config_copy)

        if debug:
            print(
                f"{debug_prefix}  '{param.name}'={value} -> fitness: {fitness}",
                file=sys.stderr,
            )

        if fitness < best_fitness:
            best_fitness = fitness
            best_values = [value]
        elif fitness == best_fitness:
            best_values.append(value)

    best_value = rng.choice(best_values)
    mutated_config = copy.deepcopy(individual.config)
    mutated_config[param.name] = best_value

    if debug:
        print(
            f"{debug_prefix}Best value for '{param.name}': {best_value} (fitness: {best_fitness})",
            file=sys.stderr,
        )

    return Individual(config=mutated_config, fitness=best_fitness)


def evolve_island_generation(
    population: list[Individual],
    island_size: int,
    search_space: SearchSpace,
    fitness_fn: FitnessFn,
    rng: random.Random,
    debug: bool = False,
    debug_prefix: str = "",
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
                print(
                    f"{debug_prefix}Warning: population too small for crossover. Mutating copy of best.",
                    file=sys.stderr,
                )
        else:
            p1 = rng.choice(population).config
            p2 = rng.choice(population).config
            child_config = crossover(p1, p2, rng)

        child = Individual(config=child_config)
        mutated = mutate(child, search_space, fitness_fn, rng, debug, debug_prefix)
        new_candidates.append(mutated)

    # Selection: keep the best `island_size`
    new_candidates.sort(key=lambda ind: ind.fitness)
    return new_candidates[:island_size]


def perform_migration(
    populations: list[list[Individual]],
    rng: random.Random,
    debug: bool = False,
) -> None:
    """Migrate best individuals between islands.

    Each island sends its best individual to a random other island,
    replacing a random individual there.
    """
    if len(populations) < 2:
        if debug:
            print("Skipping migration: less than 2 islands.", file=sys.stderr)
        return

    migrants: list[tuple[int, Individual]] = []
    for i, pop in enumerate(populations):
        if pop:
            best = min(pop, key=lambda ind: ind.fitness)
            migrants.append((i, best))
        elif debug:
            print(
                f"Warning: island {i} is empty, cannot select migrant.", file=sys.stderr
            )

    if debug:
        print(f"Performing migration with {len(migrants)} migrants.", file=sys.stderr)

    for source_idx, migrant in migrants:
        target_idx = source_idx
        while target_idx == source_idx:
            target_idx = rng.randrange(len(populations))

        target_pop = populations[target_idx]
        if not target_pop:
            target_pop.append(migrant)
            if debug:
                print(
                    f"  Migrant from island {source_idx} added to empty island {target_idx}.",
                    file=sys.stderr,
                )
        else:
            to_replace = rng.choice(target_pop)
            target_pop.remove(to_replace)
            target_pop.append(migrant)
            if debug:
                print(
                    f"  Migrant from island {source_idx} replaced an individual in island {target_idx}.",
                    file=sys.stderr,
                )


def run_island_ga(
    initial_config: dict[str, Any],
    search_space: SearchSpace,
    fitness_fn: FitnessFn,
    num_islands: int,
    population_size: int,
    num_iterations: int,
    migration_interval: int = 15,
    debug: bool = False,
    random_seed: int | None = None,
) -> Individual:
    """Run the full island-model GA.

    Args:
        initial_config: Starting configuration (param_name -> value).
        search_space: Defines parameters and their possible values.
        fitness_fn: Callable that scores a config dict (lower is better).
        num_islands: Number of parallel islands.
        population_size: Total population across all islands.
        num_iterations: Number of generations to evolve.
        migration_interval: Generations between migrations.
        debug: Print verbose output.
        random_seed: Seed for reproducibility.

    Returns:
        The best individual found.
    """
    rng = random.Random(random_seed)

    num_islands = max(1, num_islands)
    island_size = max(5, population_size // num_islands)
    if island_size * num_islands > population_size:
        population_size = island_size * num_islands

    # Evaluate initial fitness
    initial_fitness = fitness_fn(copy.deepcopy(initial_config))

    # Initialize populations
    populations: list[list[Individual]] = []
    for _ in range(num_islands):
        pop = [
            Individual(config=copy.deepcopy(initial_config), fitness=initial_fitness)
            for _ in range(island_size)
        ]
        populations.append(pop)

    best_overall = Individual(
        config=copy.deepcopy(initial_config),
        fitness=initial_fitness,
    )

    fitness_history: list[list[float]] = [[initial_fitness] for _ in range(num_islands)]

    for iteration in range(num_iterations):
        if debug:
            print(
                f"\n--- Iteration {iteration + 1}/{num_iterations} ---",
                file=sys.stderr,
            )

        # Evolve each island sequentially (caller handles parallelism)
        for i, pop in enumerate(populations):
            debug_prefix = f"Island {i + 1}: "
            new_pop = evolve_island_generation(
                pop, island_size, search_space, fitness_fn, rng, debug, debug_prefix
            )
            populations[i] = new_pop

            best_in_island = min(new_pop, key=lambda ind: ind.fitness)
            fitness_history[i].append(best_in_island.fitness)

            if debug:
                print(
                    f"  Island {i + 1} best fitness: {best_in_island.fitness}",
                    file=sys.stderr,
                )

            if best_in_island.fitness < best_overall.fitness:
                best_overall = Individual(
                    config=copy.deepcopy(best_in_island.config),
                    fitness=best_in_island.fitness,
                )
                if debug:
                    print(
                        f"  New overall best fitness: {best_overall.fitness}",
                        file=sys.stderr,
                    )

        # Early termination on perfect fitness
        if best_overall.fitness == 0:
            if debug:
                print(
                    f"\nPerfect configuration found (fitness=0) at iteration {iteration + 1}.",
                    file=sys.stderr,
                )
            break

        # Migration
        if num_islands > 1 and (iteration + 1) % migration_interval == 0:
            perform_migration(populations, rng, debug)

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

    return best_overall
