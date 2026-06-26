import sys
import copy
import random
import multiprocessing
from multiprocessing.pool import Pool
from typing import TYPE_CHECKING, Any, cast, final, override

if TYPE_CHECKING:
    pass

# Import formatter and config generator
from .repo_formatter import run_clang_format_and_count_changes
from .clang_format_parser import (
    IncrementalConfigBuilder,
    generate_clang_format_config,
    load_checkpoint,
    save_checkpoint,
)
from .data_classes import (
    GeneticOptimizationConfig,
    GeneticAlgorithmLookups,
    IslandEvolutionArgs,
    WorkerContext,
)
from .base_optimizer import BaseOptimizer  # New import

# Initialize plt to None to prevent UnboundLocalError warnings from static analyzers
plt: Any = None
matplotlib_available = False
try:
    import matplotlib.pyplot as plt  # type: ignore[assignment]

    matplotlib_available = True
except ImportError:  # pragma: no cover
    print(
        "Warning: matplotlib not found. Fitness plotting will be disabled. Install with 'pip install matplotlib' to enable.",
        file=sys.stderr,
    )


def optimize_option_with_values(
    flat_options_info: dict[str, Any],
    full_option_path: str,
    possible_values: list[Any],
    island_args: IslandEvolutionArgs,
    worker_context: WorkerContext,
    builder: IncrementalConfigBuilder | None = None,
) -> float:
    """
    Optimizes a single option by testing each value in the provided list.
    If multiple values yield the same minimum changes, one is randomly selected.
    Modifies `flat_options_info` in place with the best found value.

    Args:
        flat_options_info (dict): The flat dictionary containing all options.
                                  This dictionary is modified in place.
        full_option_path (str): The dot-separated full name of the option.
        possible_values (list): A list of values to test for this option.
        island_args (IslandEvolutionArgs): Dataclass containing common arguments for the island.
        worker_context (WorkerContext): Dataclass containing worker-specific arguments like
                                        repo_path and process_id.

    Returns:
        float: The minimum number of changes (fitness) achieved by the best value for this option.
               Returns float('inf') if no valid configuration could be found for any tested value.
    """
    option_info = flat_options_info[full_option_path]
    original_value = option_info["value"]

    if island_args.debug:
        print(
            f"Island {island_args.island_index} (Worker {worker_context.process_id}): Optimizing '{full_option_path}' (current: {original_value})...",
            file=sys.stderr,
        )
        print(
            f"Island {island_args.island_index} (Worker {worker_context.process_id}):   Testing values: {possible_values}",
            file=sys.stderr,
        )

    min_changes = float("inf")
    best_values_candidates = []

    for value_to_test in possible_values:
        # Ensure the value type matches the expected type from dump-config
        # Simple type conversion for common types
        if option_info["type"] == "bool":
            # Convert string 'true'/'false' to boolean
            if isinstance(value_to_test, str):
                if value_to_test.lower() == "true":
                    value_to_test = True
                elif value_to_test.lower() == "false":
                    value_to_test = False
                # else: keep as string, might be an enum value represented as string

        elif option_info["type"] == "int":
            try:
                value_to_test = int(value_to_test)
            except (ValueError, TypeError):
                print(
                    f"Island {island_args.island_index} (Worker {worker_context.process_id}): Warning: Could not convert value '{value_to_test}' to int for option '{full_option_path}'. Skipping.",
                    file=sys.stderr,
                )
                continue  # Skip this value

        # Add other type conversions if necessary (e.g., float, list, etc.)
        # For now, assume other types (like strings for enums) can be used directly

        flat_options_info[full_option_path]["value"] = value_to_test
        if builder is not None:
            config_string = builder.set_value(full_option_path, value_to_test)
        else:
            config_string = generate_clang_format_config(
                flat_options_info
            )  # Pass the flat dict

        # run_clang_format_and_count_changes will now exit on critical clang-format error,
        # return float('inf') on invalid config error, or return >= 0 on success, or -1 on git error.
        changes = run_clang_format_and_count_changes(
            config_string,
            repo_path=worker_context.repo_path,
            process_id=worker_context.process_id,
            debug=island_args.debug,
            file_sample_percentage=island_args.file_sample_percentage,
            random_seed=island_args.random_seed,
        )

        # We now consider float('inf') as a valid (but high) result, not an error to skip
        if changes != -1:  # Only skip if it's a git-related error (-1)
            # Treat float('inf') as a very high change count
            if island_args.debug:
                print(
                    f"Island {island_args.island_index} (Worker {worker_context.process_id}):   Testing '{full_option_path}'='{value_to_test}' -> Changes: {changes}",
                    file=sys.stderr,
                )
            if changes < min_changes:
                min_changes = changes
                best_values_candidates = [value_to_test]  # New best found, reset list
            elif changes == min_changes:
                best_values_candidates.append(
                    value_to_test
                )  # Another value with same best changes
        else:
            # An error occurred in run_clang_format_and_count_changes (e.g., git diff failed)
            # The error message is already printed by that function.
            # We just need to skip this value and continue with the next one.
            pass  # Error message already printed, continue loop

    # --- Decide Best Value ---
    if not best_values_candidates:
        # All tested values resulted in git error (-1) or invalid config (inf).
        # Revert to original value.
        print(
            f"Island {island_args.island_index} (Worker {worker_context.process_id}): All tests failed or resulted in invalid configurations for '{full_option_path}'. Keeping original value: {original_value}",
            file=sys.stderr,
        )
        flat_options_info[full_option_path]["value"] = original_value
        return float("inf")
    else:
        best_value = random.choice(best_values_candidates)
        if island_args.debug:
            print(
                f"Island {island_args.island_index} (Worker {worker_context.process_id}): Best value for '{full_option_path}': {best_value} (changes: {min_changes})",
                file=sys.stderr,
            )
        flat_options_info[full_option_path]["value"] = best_value
        return min_changes


def crossover(
    parent1_config: dict[str, Any], parent2_config: dict[str, Any]
) -> dict[str, Any]:
    """
    Performs a uniform crossover between two parent configurations.
    For each option, randomly picks the value from parent1 or parent2.
    """
    child_config = {}
    # Assuming both parents have the same set of options and structure
    for option_path in parent1_config.keys():
        # Ensure we copy the dictionary for the option's info, not just reference it
        if random.random() < 0.5:
            child_config[option_path] = copy.deepcopy(parent1_config[option_path])
        else:
            child_config[option_path] = copy.deepcopy(parent2_config[option_path])
    return child_config


def mutate(
    individual_config: dict[str, Any],
    island_args: IslandEvolutionArgs,
    worker_context: WorkerContext,
) -> tuple[dict[str, Any], float]:
    """
    Mutates an individual by selecting one random mutable option and optimizing its value
    by testing all possible values using optimize_option_with_values.

    Args:
        individual_config (dict): The configuration of the individual to mutate.
        island_args (IslandEvolutionArgs): Dataclass containing common arguments for the island.
        worker_context (WorkerContext): Dataclass containing worker-specific arguments like
                                        repo_path and process_id.

    Returns:
        tuple[dict, float]: A tuple containing the mutated configuration and its fitness.
    """
    # Apply forced options to the mutated child (ensure they are always respected)
    for forced_path, forced_value in island_args.lookups.forced_options_lookup.items():
        if forced_path in individual_config:
            individual_config[forced_path]["value"] = forced_value

    # Find mutable options (not forced, has possible values or is boolean)
    mutable_options = []
    for full_option_path, option_info in individual_config.items():
        if full_option_path in island_args.lookups.forced_options_lookup:
            continue  # Skip forced options, they are not mutable by the GA

        if (
            full_option_path in island_args.lookups.json_options_lookup
            and island_args.lookups.json_options_lookup[full_option_path][
                "possible_values"
            ]
        ) or (option_info["type"] == "bool"):
            mutable_options.append(full_option_path)

    if not mutable_options:
        if island_args.debug:
            print(
                f"Island {island_args.island_index} (Worker {worker_context.process_id}): No mutable options found for mutation. Returning original config and its fitness.",
                file=sys.stderr,
            )
        return copy.deepcopy(individual_config), float("inf")

    option_to_mutate_path = random.choice(mutable_options)
    option_info = individual_config[option_to_mutate_path]

    possible_values = []
    if (
        option_to_mutate_path in island_args.lookups.json_options_lookup
        and island_args.lookups.json_options_lookup[option_to_mutate_path][
            "possible_values"
        ]
    ):
        possible_values = island_args.lookups.json_options_lookup[
            option_to_mutate_path
        ]["possible_values"]
    elif option_info["type"] == "bool":
        possible_values = [True, False]

    if island_args.debug:
        print(
            f"Island {island_args.island_index} (Worker {worker_context.process_id}): Mutating '{option_to_mutate_path}' (current: {option_info['value']})...",
            file=sys.stderr,
        )

    builder = IncrementalConfigBuilder(individual_config)
    mutated_fitness = optimize_option_with_values(
        individual_config,
        option_to_mutate_path,
        possible_values,
        island_args,
        worker_context,
        builder,
    )

    return individual_config, mutated_fitness


def _evolve_island_generation_task(
    island_args: IslandEvolutionArgs, worker_context: WorkerContext
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Helper function to evolve a single island for one generation.
    This function is called by the multiprocessing pool.

    Args:
        island_args (IslandEvolutionArgs): Dataclass containing all arguments
                                           for this island's evolution task.
        worker_context (WorkerContext): Dataclass containing worker-specific arguments.

    Returns:
        tuple: (new_population, best_individual_in_generation)
    """
    # Unpack arguments from the dataclass
    population = island_args.population
    island_population_size = island_args.island_population_size

    new_generation_candidates = []

    # Elitism: Keep the best individual from the current population
    if population:
        best_current_individual = min(population, key=lambda x: x["fitness"])
        new_generation_candidates.append(best_current_individual)
    else:
        # This case should ideally not happen if initial population is correctly created
        return [], {"config": {}, "fitness": float("inf")}

    # Generate new individuals through crossover and mutation
    num_to_generate = island_population_size - len(new_generation_candidates)
    if num_to_generate < 0:  # pragma: no cover
        num_to_generate = 0

    for _ in range(num_to_generate):
        # Selection: Simple random selection for parents from the current population
        if len(population) < 2:
            parent1 = population[0]["config"] if population else {}
            child_config = copy.deepcopy(parent1)
            if island_args.debug:
                print(
                    f"Island {island_args.island_index} (Worker {worker_context.process_id}): Warning: Island population too small for crossover. Mutating a copy of the best individual.",
                    file=sys.stderr,
                )
        else:
            parent1 = random.choice(population)["config"]
            parent2 = random.choice(population)["config"]
            child_config = crossover(parent1, parent2)

        # Mutation: Mutate one random option in the child
        # The mutate function now handles applying forced options internally
        mutated_child_config, child_fitness = mutate(
            child_config,
            island_args,  # Pass the IslandEvolutionArgs object
            worker_context,  # Pass the WorkerContext object
        )

        new_generation_candidates.append(
            {"config": mutated_child_config, "fitness": child_fitness}
        )

    # Selection for next generation: Sort all candidates and take the top `island_population_size`
    new_population = sorted(new_generation_candidates, key=lambda x: x["fitness"])[
        :island_population_size
    ]

    # Find the best individual in this new generation
    best_in_generation = (
        min(new_population, key=lambda x: x["fitness"])
        if new_population
        else {"config": {}, "fitness": float("inf")}
    )

    return new_population, best_in_generation


def _island_evolution_task_wrapper(
    args_tuple: tuple[
        IslandEvolutionArgs, list[str], GeneticAlgorithmLookups, bool, float, int
    ],
) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    """
    Wrapper function for multiprocessing.Pool.imap_unordered.
    It unpacks arguments, determines the worker's specific repo_path and process_id,
    creates a WorkerContext, and then calls the actual island evolution logic.
    Returns (island_index, new_population, best_individual).
    """
    (
        island_args_for_this_island,
        all_repo_paths,
        lookups,
        debug,
        file_sample_percentage,
        random_seed,
    ) = args_tuple

    # Get the unique process ID for this worker
    # _identity[0] gives a unique integer for each worker process (starts from 1)
    process_id = multiprocessing.current_process()._identity[0]

    # Select the dedicated repo path for this worker based on its process ID
    # Ensure process_id - 1 is a valid index for all_repo_paths
    repo_path_for_worker = all_repo_paths[process_id - 1]

    # Create a WorkerContext for this specific worker process
    worker_context = WorkerContext(
        repo_path=repo_path_for_worker, process_id=process_id
    )

    # Derive a per-worker seed so each worker samples different files
    worker_seed = random_seed + process_id

    # Update the island_args with the lookups, debug, sampling, and random_seed
    # These were passed separately to the wrapper because they are constant across all islands
    # but need to be part of IslandEvolutionArgs for the inner functions.
    # This is a bit of a hack to avoid modifying IslandEvolutionArgs in place in the main process,
    # but it works for passing context down.
    island_args_for_this_island.lookups = lookups
    island_args_for_this_island.debug = debug
    island_args_for_this_island.file_sample_percentage = file_sample_percentage
    island_args_for_this_island.random_seed = worker_seed

    # Now call the actual island evolution function
    new_pop, best = _evolve_island_generation_task(
        island_args_for_this_island, worker_context
    )
    return (island_args_for_this_island.island_index, new_pop, best)


def _polish_single_option_task(
    args: tuple[
        dict[str, Any],
        str,
        list[Any],
        GeneticAlgorithmLookups,
        bool,
        str,
        float,
        int,
    ],
) -> tuple[str, float, Any]:
    """
    Worker function for parallel polish. Evaluates a single option and returns
    the best fitness and value found.

    Returns:
        Tuple of (option_path, best_fitness, best_value).
    """
    (
        config_copy,
        option_path,
        possible_values,
        lookups,
        debug,
        repo_path,
        file_sample_pct,
        seed,
    ) = args

    worker_context = WorkerContext(repo_path=repo_path, process_id=0)
    island_args = IslandEvolutionArgs(
        population=[],
        island_population_size=0,
        island_index=-1,
        lookups=lookups,
        debug=debug,
        file_sample_percentage=file_sample_pct,
        random_seed=seed,
    )

    builder = IncrementalConfigBuilder(config_copy)
    fitness = optimize_option_with_values(
        config_copy,
        option_path,
        possible_values,
        island_args,
        worker_context,
        builder,
    )
    best_value = config_copy[option_path]["value"]
    return option_path, fitness, best_value


def _polish_coordinate_descent(
    best_config: dict[str, Any],
    repo_path: str,
    lookups: GeneticAlgorithmLookups,
    debug: bool,
    file_sample_percentage: float,
    random_seed: int,
    max_passes: int,
    pool: Pool | None = None,
) -> tuple[dict[str, Any], float]:
    """
    Sequential coordinate descent polish on the best GA result.
    Iterates over all mutable options, optimizing each one exhaustively.
    Repeats passes until no improvements are found or max_passes is reached.

    Returns:
        tuple: (polished_config, final_fitness)
    """
    worker_context = WorkerContext(repo_path=repo_path, process_id=0)
    island_args = IslandEvolutionArgs(
        population=[],
        island_population_size=0,
        island_index=-1,
        lookups=lookups,
        debug=debug,
        file_sample_percentage=file_sample_percentage,
        random_seed=random_seed,
    )

    # Collect mutable options once (order is deterministic via sorted keys)
    mutable_options = []
    for full_option_path, option_info in sorted(best_config.items()):
        if full_option_path in lookups.forced_options_lookup:
            continue
        if (
            full_option_path in lookups.json_options_lookup
            and lookups.json_options_lookup[full_option_path]["possible_values"]
        ) or (option_info["type"] == "bool"):
            mutable_options.append(full_option_path)

    if not mutable_options:
        print("  No mutable options for polish. Skipping.", file=sys.stderr)
        return best_config, best_config.get("__fitness__", float("inf"))

    # Compute initial fitness
    builder = IncrementalConfigBuilder(best_config)
    current_fitness = run_clang_format_and_count_changes(
        builder.build(),
        repo_path=repo_path,
        process_id=worker_context.process_id,
        debug=debug,
        file_sample_percentage=file_sample_percentage,
        random_seed=random_seed,
    )

    print(
        f"  Starting coordinate descent polish. {len(mutable_options)} mutable options.",
        file=sys.stderr,
    )
    print(f"  Initial fitness: {current_fitness}", file=sys.stderr)

    for pass_num in range(1, max_passes + 1):
        improvements = 0

        if pool is not None:
            # Parallel path: evaluate each option in a separate worker.
            # Each worker gets a deep copy of the config so mutations are isolated.
            tasks: list[
                tuple[
                    dict[str, Any],
                    str,
                    list[Any],
                    GeneticAlgorithmLookups,
                    bool,
                    str,
                    float,
                    int,
                ]
            ] = []
            for full_option_path in mutable_options:
                option_info = best_config[full_option_path]
                possible_values = []
                if (
                    full_option_path in lookups.json_options_lookup
                    and lookups.json_options_lookup[full_option_path]["possible_values"]
                ):
                    possible_values = lookups.json_options_lookup[full_option_path][
                        "possible_values"
                    ]
                elif option_info["type"] == "bool":
                    possible_values = [True, False]

                tasks.append(
                    (
                        copy.deepcopy(best_config),
                        full_option_path,
                        possible_values,
                        lookups,
                        debug,
                        repo_path,
                        file_sample_percentage,
                        random_seed,
                    )
                )

            results = pool.map(_polish_single_option_task, tasks)
            for option_path, fitness, best_value in results:
                old_fitness = current_fitness
                if fitness < old_fitness:
                    best_config[option_path]["value"] = best_value
                    current_fitness = fitness
                    improvements += 1
        else:
            # Sequential path
            for full_option_path in mutable_options:
                option_info = best_config[full_option_path]
                possible_values = []
                if (
                    full_option_path in lookups.json_options_lookup
                    and lookups.json_options_lookup[full_option_path]["possible_values"]
                ):
                    possible_values = lookups.json_options_lookup[full_option_path][
                        "possible_values"
                    ]
                elif option_info["type"] == "bool":
                    possible_values = [True, False]

                old_fitness = current_fitness
                new_fitness = optimize_option_with_values(
                    best_config,
                    full_option_path,
                    possible_values,
                    island_args,
                    worker_context,
                    builder,
                )
                if new_fitness < old_fitness:
                    improvements += 1
                    current_fitness = new_fitness

        print(
            f"  Polish pass {pass_num}: {improvements} improvements, fitness: {current_fitness}",
            file=sys.stderr,
        )
        if improvements == 0:
            print(
                f"  Coordinate descent converged after {pass_num} pass(es).",
                file=sys.stderr,
            )
            break
    else:
        print(
            f"  Coordinate descent reached max passes ({max_passes}).", file=sys.stderr
        )

    return best_config, current_fitness


def _perform_migration(populations: list[list[dict[str, Any]]], debug: bool = False):
    """
    Performs migration between islands.
    Each island sends its best individual to a randomly chosen other island,
    replacing a random individual in the target island.
    """
    if len(populations) < 2:
        if debug:
            print("Skipping migration: Less than 2 islands.", file=sys.stderr)
        return

    migrants = []
    # Collect the best individual from each island
    for i, island_pop in enumerate(populations):
        if island_pop:
            best_individual = min(island_pop, key=lambda x: x["fitness"])
            migrants.append({"source_island_idx": i, "individual": best_individual})
        else:
            if debug:
                print(
                    f"Warning: Island {i} is empty, cannot select migrant.",
                    file=sys.stderr,
                )

    if debug:
        print(f"Performing migration with {len(migrants)} migrants.", file=sys.stderr)

    # Distribute migrants
    for migrant_info in migrants:
        source_idx: int = cast(int, migrant_info["source_island_idx"])
        migrant: dict[str, Any] = cast(dict[str, Any], migrant_info["individual"])

        # Choose a random target island different from the source
        target_island_idx = source_idx
        while target_island_idx == source_idx:
            target_island_idx = random.randrange(len(populations))

        target_population = populations[target_island_idx]

        if not target_population:
            # If target island is empty, just add the migrant
            target_population.append(migrant)
            if debug:
                print(
                    f"  Migrant from island {source_idx} added to empty island {target_island_idx}.",
                    file=sys.stderr,
                )
        else:
            # Replace a random individual in the target island with the migrant
            individual_to_replace = random.choice(target_population)
            target_population.remove(individual_to_replace)
            target_population.append(migrant)
            if debug:
                print(
                    f"  Migrant from island {source_idx} replaced an individual in island {target_island_idx}.",
                    file=sys.stderr,
                )


@final
class GeneticAlgorithmOptimizer(BaseOptimizer):
    """
    Implements the genetic algorithm optimization strategy with an island model.
    """

    config: GeneticOptimizationConfig  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(self, config: GeneticOptimizationConfig):
        super().__init__(config)

    def _initialize_populations(
        self,
        base_options_info: dict[str, Any],
        repo_paths: list[str],
        lookups: GeneticAlgorithmLookups,
        num_islands: int,
        total_population_size: int,
        debug: bool,
        file_sample_percentage: float,
        random_seed: int,
    ) -> tuple[
        list[list[dict[str, Any]]],
        dict[str, Any],
        list[list[float]],
        int,
        int,
    ]:
        """Initialize island populations and calculate base fitness.

        Returns:
            Tuple of (populations, best_overall_individual, fitness_history_per_island,
                      island_population_size, num_islands).
        """
        MIN_INDIVIDUALS_PER_ISLAND = 5
        island_population_size = max(
            MIN_INDIVIDUALS_PER_ISLAND, total_population_size // num_islands
        )

        if island_population_size * num_islands > total_population_size:
            total_population_size = island_population_size * num_islands
            print(
                f"Adjusted total population size to {total_population_size} to ensure at least {island_population_size} individuals per island.",
                file=sys.stderr,
            )

        print(
            f"\nInitializing {num_islands} islands, each with {island_population_size} individuals (total: {total_population_size})...",
            file=sys.stderr,
        )

        base_individual_config = copy.deepcopy(base_options_info)
        for (
            forced_path,
            forced_value,
        ) in lookups.forced_options_lookup.items():  # pragma: no cover
            if forced_path in base_individual_config:
                base_individual_config[forced_path]["value"] = forced_value

        print("Calculating initial base configuration fitness...", file=sys.stderr)
        initial_repo_path = repo_paths[0] if repo_paths else None
        if initial_repo_path is None:
            print(
                "Error: No repository paths provided for initialization.",
                file=sys.stderr,
            )
            return [], {}, [], 0, num_islands

        initial_worker_context = WorkerContext(
            repo_path=initial_repo_path,
            process_id=0,
        )

        base_fitness = run_clang_format_and_count_changes(
            generate_clang_format_config(base_individual_config),
            repo_path=initial_worker_context.repo_path,
            process_id=initial_worker_context.process_id,
            debug=debug,
            file_sample_percentage=file_sample_percentage,
            random_seed=random_seed,
        )
        print(f"Initial base configuration fitness: {base_fitness}", file=sys.stderr)

        populations: list[list[dict[str, Any]]] = []
        for _ in range(num_islands):
            island_pop = []
            for _ in range(island_population_size):
                individual_config = copy.deepcopy(base_individual_config)
                island_pop.append(
                    {"config": individual_config, "fitness": base_fitness}
                )
            populations.append(island_pop)

        best_overall_individual: dict[str, Any] = {
            "config": base_individual_config,
            "fitness": base_fitness,
        }
        print(
            f"\nInitial overall best fitness: {best_overall_individual['fitness']}",
            file=sys.stderr,
        )

        fitness_history_per_island: list[list[float]] = [[] for _ in range(num_islands)]
        for i in range(num_islands):
            fitness_history_per_island[i].append(base_fitness)

        return (
            populations,
            best_overall_individual,
            fitness_history_per_island,
            island_population_size,
            num_islands,
        )

    @override
    def optimize(
        self,
        base_options_info: dict[str, Any],
        repo_paths: list[str],
        lookups: GeneticAlgorithmLookups,
        file_sample_percentage: float,
        random_seed: int,
        checkpoint_resume_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Optimizes clang-format configuration using a genetic algorithm with an island model.

        Args:
            base_options_info (dict): The initial flat dictionary from clang-format --dump-config.
            repo_paths (list): A list of paths to the temporary git repositories for parallel processing.
            lookups (GeneticAlgorithmLookups): A dataclass containing lookup dictionaries for
                                               option values and forced options.
            file_sample_percentage (float): Percentage of files to sample for fitness calculation.
            random_seed (int): Seed for random file sampling.

        Returns:
            dict: The flat dictionary of the best clang-format configuration found.
        """
        config = self.config
        num_iterations = config.num_iterations
        total_population_size = config.total_population_size
        num_islands = config.num_islands
        debug = config.debug
        plot_fitness = config.plot_fitness
        polish_passes = config.polish_passes
        checkpoint_interval = config.checkpoint_interval
        checkpoint_path = checkpoint_resume_path

        if num_islands < 1:
            print(
                "Error: Number of islands must be at least 1. Setting to 1.",
                file=sys.stderr,
            )
            num_islands = 1

        # Checkpoint resume: load state and skip to saved iteration
        start_iteration = 0
        populations: list[list[dict[str, Any]]] = []
        best_overall_individual: dict[str, Any] = {}
        fitness_history_per_island: list[list[float]] = []
        island_population_size = 0

        if checkpoint_path:
            checkpoint_data = load_checkpoint(checkpoint_path)
            if checkpoint_data:
                print(
                    f"Resuming from checkpoint: {checkpoint_path}",
                    file=sys.stderr,
                )
                populations = checkpoint_data["populations"]
                best_overall_individual = {
                    "config": checkpoint_data["best_config"],
                    "fitness": checkpoint_data["best_fitness"],
                }
                fitness_history_per_island = checkpoint_data[
                    "fitness_history_per_island"
                ]
                start_iteration = checkpoint_data["iteration"] + 1
                num_islands = len(populations)
                island_population_size = len(populations[0]) if populations else 0
                print(
                    f"  Resumed at iteration {start_iteration}/{num_iterations}",
                    f" best fitness: {best_overall_individual['fitness']}",
                    file=sys.stderr,
                )
            else:
                print(
                    f"Checkpoint file not found or invalid: {checkpoint_path}. Starting fresh.",
                    file=sys.stderr,
                )

        if start_iteration == 0 and not populations:
            (
                populations,
                best_overall_individual,
                fitness_history_per_island,
                island_population_size,
                num_islands,
            ) = self._initialize_populations(
                base_options_info,
                repo_paths,
                lookups,
                num_islands,
                total_population_size,
                debug,
                file_sample_percentage,
                random_seed,
            )

        if not populations:
            return {}

        pool = multiprocessing.Pool(processes=len(repo_paths))
        interrupted = False
        try:
            (
                populations,
                best_overall_individual,
                fitness_history_per_island,
                interrupted,
            ) = self._run_evolution_loop(
                populations,
                best_overall_individual,
                fitness_history_per_island,
                island_population_size,
                num_islands,
                num_iterations,
                start_iteration,
                repo_paths,
                lookups,
                debug,
                file_sample_percentage,
                random_seed,
                plot_fitness,
                self.config.migration_interval,
                pool,
                checkpoint_path,
                checkpoint_interval,
            )

            print(
                f"\nGenetic algorithm finished. Best overall fitness: {best_overall_individual['fitness']}",
                file=sys.stderr,
            )

            # Coordinate descent polish phase
            if polish_passes > 0 and not interrupted:
                print(
                    f"\n--- Coordinate Descent Polish (max {polish_passes} passes) ---",
                    file=sys.stderr,
                )
                polish_repo_path = repo_paths[0] if repo_paths else None
                if polish_repo_path:
                    best_config: dict[str, Any] = cast(
                        dict[str, Any], copy.deepcopy(best_overall_individual["config"])
                    )
                    best_fitness: float = cast(
                        float, best_overall_individual["fitness"]
                    )
                    polished_config, polished_fitness = _polish_coordinate_descent(
                        best_config,
                        polish_repo_path,
                        lookups,
                        debug,
                        file_sample_percentage,
                        random_seed,
                        polish_passes,
                        pool,
                    )
                    if polished_fitness < best_fitness:
                        best_overall_individual = {
                            "config": polished_config,
                            "fitness": polished_fitness,
                        }
                        print(
                            f"  Polish improved fitness: {best_overall_individual['fitness']}",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"  Polish did not improve fitness. Best remains: {best_overall_individual['fitness']}",
                            file=sys.stderr,
                        )
                else:
                    print(
                        "  Skipping polish: no repo paths available.", file=sys.stderr
                    )  # pragma: no cover
        finally:
            if interrupted:  # pragma: no cover
                print("Forcing termination of worker pool...", file=sys.stderr)
                pool.terminate()
            else:
                pool.close()
            pool.join()
            print("Worker pool shut down.", file=sys.stderr)

        return cast(dict[str, Any], best_overall_individual["config"])

    def _run_evolution_loop(
        self,
        populations: list[list[dict[str, Any]]],
        best_overall_individual: dict[str, Any],
        fitness_history_per_island: list[list[float]],
        island_population_size: int,
        num_islands: int,
        num_iterations: int,
        start_iteration: int,
        repo_paths: list[str],
        lookups: GeneticAlgorithmLookups,
        debug: bool,
        file_sample_percentage: float,
        random_seed: int,
        plot_fitness: bool,
        migration_interval: int,
        pool: Pool,
        checkpoint_path: str | None,
        checkpoint_interval: int,
    ) -> tuple[
        list[list[dict[str, Any]]],
        dict[str, Any],
        list[list[float]],
        bool,
    ]:
        """Run the island evolution loop with migration and early termination.

        Returns:
            Tuple of (populations, best_overall_individual, fitness_history_per_island, interrupted).
        """
        global plt
        fig = None
        ax = None
        lines: list[Any] = []

        if plot_fitness and matplotlib_available:  # pragma: no cover
            assert plt is not None
            plt.ion()
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title("Best Fitness Over Generations for Each Island")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Best Fitness (Number of Changes)")
            ax.grid(True)
            lines = [
                ax.plot([], [], label=f"Island {i + 1}")[0] for i in range(num_islands)
            ]
            ax.legend()
            fig.show()
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)
        elif plot_fitness and not matplotlib_available:  # pragma: no cover
            print(
                "Plotting requested but matplotlib is not available. Skipping plot.",
                file=sys.stderr,
            )
            plot_fitness = False

        interrupted = False

        try:
            for iteration in range(start_iteration, num_iterations):
                print(
                    f"\n--- Iteration {iteration + 1}/{num_iterations} ---",
                    file=sys.stderr,
                )

                tasks_args = []
                for i, island_pop in enumerate(populations):
                    island_args_for_this_island = IslandEvolutionArgs(
                        population=island_pop,
                        island_population_size=island_population_size,
                        island_index=i,
                        lookups=lookups,
                        debug=debug,
                        file_sample_percentage=file_sample_percentage,
                        random_seed=random_seed,
                    )
                    tasks_args.append(
                        (
                            island_args_for_this_island,
                            repo_paths,
                            lookups,
                            debug,
                            file_sample_percentage,
                            random_seed,
                        )
                    )

                results = pool.imap_unordered(
                    _island_evolution_task_wrapper, tasks_args
                )

                for (
                    island_index,
                    new_island_pop,
                    best_in_island_for_this_gen,
                ) in results:
                    populations[island_index] = new_island_pop
                    print(
                        f"    Island {island_index + 1} best fitness: {best_in_island_for_this_gen['fitness']}",
                        file=sys.stderr,
                    )

                    fitness_history_per_island[island_index].append(
                        best_in_island_for_this_gen["fitness"]
                    )

                    if (
                        best_in_island_for_this_gen["fitness"]
                        < best_overall_individual["fitness"]
                    ):
                        best_overall_individual = best_in_island_for_this_gen
                        print(
                            f"    New overall best fitness found: {best_overall_individual['fitness']}",
                            file=sys.stderr,
                        )

                if best_overall_individual["fitness"] == 0:
                    print(
                        f"\nPerfect configuration found (fitness=0) at iteration {iteration + 1}. Stopping early.",
                        file=sys.stderr,
                    )
                    break

                if plot_fitness and not interrupted:  # pragma: no cover
                    assert ax is not None
                    assert fig is not None
                    assert plt is not None
                    for i, history in enumerate(fitness_history_per_island):
                        lines[i].set_data(range(len(history)), history)
                    ax.relim()
                    ax.autoscale_view()
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    plt.pause(0.01)

                if (
                    num_islands > 1
                    and (iteration + 1) % migration_interval == 0
                    and not interrupted
                ):
                    print(
                        f"\n--- Performing migration at iteration {iteration + 1} ---",
                        file=sys.stderr,
                    )
                    _perform_migration(populations, debug)

                    all_individuals_after_migration: list[dict[str, Any]] = [
                        ind for island_pop in populations for ind in island_pop
                    ]
                    if all_individuals_after_migration:
                        current_overall_best_after_migration = min(
                            all_individuals_after_migration, key=lambda x: x["fitness"]
                        )
                        if (
                            current_overall_best_after_migration["fitness"]
                            < best_overall_individual["fitness"]
                        ):  # pragma: no cover
                            best_overall_individual = (
                                current_overall_best_after_migration
                            )
                            print(
                                f"  New overall best fitness found after migration: {current_overall_best_after_migration['fitness']}",
                                file=sys.stderr,
                            )
                        else:
                            print(
                                f"  Overall best fitness remains: {best_overall_individual['fitness']} after migration.",
                                file=sys.stderr,
                            )
                    else:
                        print(
                            "Warning: All populations are empty after migration. This should not happen.",
                            file=sys.stderr,
                        )  # pragma: no cover

                # Save checkpoint at configured intervals (skip last iteration)
                if (
                    checkpoint_path
                    and checkpoint_interval > 0
                    and iteration < num_iterations - 1
                    and (iteration + 1) % checkpoint_interval == 0
                ):
                    save_checkpoint(
                        checkpoint_path,
                        cast(dict[str, Any], best_overall_individual["config"]),
                        cast(float, best_overall_individual["fitness"]),
                        iteration,
                        populations,
                        fitness_history_per_island,
                    )
                    print(
                        f"  Checkpoint saved at iteration {iteration + 1}",
                        file=sys.stderr,
                    )

        except KeyboardInterrupt:  # pragma: no cover
            print(
                "\nCtrl-C detected. Terminating optimization immediately...",
                file=sys.stderr,
            )
            interrupted = True
            if matplotlib_available and plt:
                try:
                    plt.close("all")
                    print("Matplotlib plots closed.", file=sys.stderr)
                except Exception as e:
                    print(
                        f"Warning: Could not close matplotlib plots: {e}",
                        file=sys.stderr,
                    )

        # Keep plot open at end if not interrupted
        if (
            plot_fitness and matplotlib_available and not interrupted
        ):  # pragma: no cover
            assert plt is not None
            plt.ioff()
            plt.show()

        return (
            populations,
            best_overall_individual,
            fitness_history_per_island,
            interrupted,
        )
