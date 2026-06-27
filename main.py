import argparse
import json
import os
import subprocess
import sys
import tempfile  # New import for temporary directories
import shutil  # New import for copying/removing directories
import yaml


# Import functions from the new modules (relative imports within the src package)
from src.clang_format_parser import (
    get_clang_format_options,
    parse_clang_format_options,
    generate_clang_format_config,
)
from src.config_loader import load_json_option_values, load_forced_options
from src.data_classes import GeneticAlgorithmLookups
from src.optimization_engine import (
    polish_coordinate_descent,
    run_island_ga,
    run_nevergrad_optimization,
)
from src.clang_format_adapter import (
    build_search_space,
    config_to_flat_options,
    make_fitness_function,
)
from src.analyze_conventions import analyze as analyze_conventions
from src.utils import run_command

# Global debug flag (will be set from args)
debug_mode = False

# Fixed random seed for reproducibility of file sampling
# This ensures that if file sampling is used, the same subset of files is chosen
# for fitness evaluation across different runs with the same parameters.
RANDOM_SEED = 42

OptionInfo = dict[str, str | list[str] | None]


def find_options_without_json_values(
    flat_options_info: dict[str, OptionInfo],
    json_options_lookup: dict[str, OptionInfo],
    forced_options_lookup: dict[str, OptionInfo],
    missing_list: list[str],
) -> None:
    """
    Finds options in the flat dump-config structure that are not in the
    JSON lookup or have no possible values listed (excluding booleans which are auto-tested).
    Also excludes options that are present in the forced_options_lookup.

    Args:
        flat_options_info (dict): The flat dictionary from parsed dump-config.
        json_options_lookup (dict): The flat dictionary from the JSON file.
        forced_options_lookup (dict): The flat dictionary from the forced options YAML file.
        missing_list (list): The list to append missing option names to.
    """
    for full_option_path, option_info in flat_options_info.items():
        # Check if the option is in the JSON lookup and has possible values
        # OR if it's a boolean (which is handled automatically)
        # AND if it's NOT in the forced options lookup
        if (
            full_option_path not in json_options_lookup
            or not json_options_lookup[full_option_path]["possible_values"]
        ) and (full_option_path not in forced_options_lookup):
            # If not in JSON or no values in JSON, check if it's a boolean
            if option_info["type"] != "bool":
                # If it's not a boolean and not in JSON/no values, and not forced, add its full path to missing list
                missing_list.append(full_option_path)
            # If it *is* a boolean and not in JSON/no values, it will be auto-tested, so don't add to missing list


def get_repo_disk_usage(repo_path: str) -> int:
    """Return disk usage of repo in bytes, or 0 on failure.

    Uses ``du -sb`` which is fast even for repos with many files since it only
    reads directory metadata, not file contents. Note that ``-b`` is a GNU
    extension; on non-GNU systems this returns 0, which is safe because
    tmpfs detection is Linux-only anyway.
    """
    try:
        result = run_command(
            ["du", "-sb", repo_path],
            capture_output=True,
            text=True,
            check=False,
        )
        size_str = result.stdout.split()[0] if result.stdout else "0"
        return int(size_str)
    except (subprocess.CalledProcessError, ValueError):
        return 0


def find_tmpfs_mounts() -> list[str]:
    """Return mount points of all tmpfs filesystems, ordered by mount path.

    Reads ``/proc/mounts`` on Linux; returns an empty list on other platforms
    or when the file cannot be read.
    """
    mounts = []
    try:
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3 and parts[2] == "tmpfs":
                    mounts.append(parts[1])
    except OSError:
        pass
    return mounts


def get_best_temp_location(
    repo_path: str,
    jobs: int,
    debug: bool,
) -> tuple[str, bool]:
    """Return a (path, was_auto_detected) pair for temp repo storage.

    Follows the Strangler Fig pattern — new detection code grows alongside the
    existing ``tempfile.mkdtemp()`` default; the old path is preserved as
    fallback and is never removed.

    Priority:
        1. User-specified via ``--temp-dir`` (highest priority, no auto-detect).
        2. Auto-detect any tmpfs mount with enough free space.
        3. Fall back to the original behaviour – use ``tempfile.gettempdir()``
           so users who don't opt in see exactly the same behavior as before
           this feature was added.
    """
    repo_size = get_repo_disk_usage(repo_path)
    needed_per_job = max(1, jobs)  # at least one copy per job
    needed_total = repo_size * 2 * needed_per_job

    # Check all tmpfs mounts — RAM-backed, 10-50x faster than disk /tmp
    for mount in find_tmpfs_mounts():
        try:
            if not os.access(mount, os.W_OK):
                continue
            stat = shutil.disk_usage(mount)
            if stat.free > needed_total:
                if debug:
                    print(
                        f"tmpfs available at {mount} ({stat.free} bytes free)",
                        file=sys.stderr,
                    )
                return (mount, True)
        except (OSError, AttributeError):
            continue

    # Fallback to original behavior — preserved for backward compatibility
    if debug:
        print(
            f"Falling back to default temp dir: {tempfile.gettempdir()}",
            file=sys.stderr,
        )
    return (tempfile.gettempdir(), False)


def cmd_optimize(args: argparse.Namespace) -> None:
    """Execute the optimize subcommand."""
    # Set global debug flag
    global debug_mode
    debug_mode = args.debug

    # Basic validation
    if not os.path.isdir(args.repo_path):
        print(
            f"Error: Repository path '{args.repo_path}' is not a valid directory.",
            file=sys.stderr,
        )
        exit(1)

    # Ensure the path is absolute for reliable chdir/restore
    repo_path_abs = os.path.abspath(args.repo_path)
    if not os.path.isdir(repo_path_abs):
        print(
            f"Error: Absolute repository path '{repo_path_abs}' is not a valid directory.",
            file=sys.stderr,
        )
        exit(1)

    print(f"Analyzing repository: {repo_path_abs}", file=sys.stderr)

    options_info = None
    if args.start_config_file:
        print(
            f"\nLoading initial configuration from: {args.start_config_file}",
            file=sys.stderr,
        )
        if not os.path.exists(args.start_config_file):
            print(
                f"Error: Start config file '{args.start_config_file}' not found.",
                file=sys.stderr,
            )
            exit(1)
        try:
            with open(args.start_config_file) as f:
                start_config_content = f.read()
            options_info = parse_clang_format_options(start_config_content)
            if not options_info:
                print(
                    f"Error: Failed to parse YAML from '{args.start_config_file}'. Please ensure it's a valid .clang-format file.",
                    file=sys.stderr,
                )
                exit(1)
            print(
                f"Successfully loaded and parsed initial configuration from '{args.start_config_file}'.",
                file=sys.stderr,
            )
        except OSError as e:
            print(
                f"Error reading start config file '{args.start_config_file}': {e}",
                file=sys.stderr,
            )
            exit(1)
        except yaml.YAMLError as e:
            print(
                f"Error parsing YAML from '{args.start_config_file}': {e}",
                file=sys.stderr,
            )
            exit(1)
    else:
        # Get initial clang-format options structure from dump-config
        print(
            "\nGetting initial configuration from 'clang-format --dump-config'...",
            file=sys.stderr,
        )
        options_output = get_clang_format_options(debug=debug_mode)

        if not options_output:
            print(
                "\nFailed to retrieve clang-format options from --dump-config.",
                file=sys.stderr,
            )
            exit(1)

        # Parse and flatten the options
        options_info = parse_clang_format_options(options_output)

        if not options_info:
            print(
                "\nFailed to parse clang-format options from --dump-config.",
                file=sys.stderr,
            )
            exit(1)

        print(
            "\nSuccessfully parsed and flattened clang-format options structure from --dump-config.",
            file=sys.stderr,
        )

    # Run convention analysis (default behavior, opt-out with --no-analyze)
    analysis_results = None
    if not args.no_analyze:
        print("\nAnalyzing code conventions...", file=sys.stderr)
        analysis_results = analyze_conventions(repo_path_abs)
        if analysis_results:
            print(
                f"Detected {len(analysis_results)} convention(s) from source code.",
                file=sys.stderr,
            )
        else:
            print(
                "Convention analysis completed. No strong conventions detected.",
                file=sys.stderr,
            )
    else:
        print("Convention analysis skipped (--no-analyze).", file=sys.stderr)

    # --dry-run: print detected conventions and exit
    if args.dry_run:
        if analysis_results:
            print(yaml.dump(analysis_results, default_flow_style=False))
        else:
            print("# No conventions detected.", file=sys.stderr)
        return

    # Load external configurations (json_options_lookup is already flat)
    json_options_lookup = load_json_option_values(args.option_values_json_file)
    # Load and flatten forced options
    forced_options_lookup = load_forced_options(args.forced_options_yaml_file)

    # Create GeneticAlgorithmLookups object (used by both optimizers for option info)
    lookups = GeneticAlgorithmLookups(
        json_options_lookup=json_options_lookup,
        forced_options_lookup=forced_options_lookup,
    )

    # Identify options missing from JSON or without possible values (excluding booleans)
    # and not present in forced options
    missing_options: list[str] = []
    # Pass the flat options_info directly
    find_options_without_json_values(
        options_info,
        lookups.json_options_lookup,
        lookups.forced_options_lookup,
        missing_options,
    )

    if missing_options:
        print(
            "\nThe following options were found in the base config but were not present in the provided JSON file or had no possible values listed (and are not booleans or forced options):",
            file=sys.stderr,
        )
        for opt_path in missing_options:  # opt_path now contains the full path
            print(f"- {opt_path}", file=sys.stderr)
        print(
            "These options will retain their values from the base config unless specified in the forced options YAML file.",
            file=sys.stderr,
        )
    elif args.option_values_json_file:
        print(
            "\nAll non-boolean options found in the base config were present in the provided JSON file with possible values, or were explicitly forced.",
            file=sys.stderr,
        )
    else:
        print(
            "\nNo JSON file with option values was provided. All non-boolean options will retain their values from the base config unless specified in the forced options YAML file. Boolean options will be tested automatically.",
            file=sys.stderr,
        )

    num_jobs = args.jobs
    if num_jobs < 1:
        print(
            "Error: Number of jobs must be at least 1. Setting to 1.", file=sys.stderr
        )
        num_jobs = 1

    temp_repo_paths: list[str] = []
    print(
        f"\nPreparing {num_jobs} temporary copies of the repository for parallel processing...",
        file=sys.stderr,
    )

    # Use user-specified temp dir if provided, otherwise auto-detect or fall back
    base_temp_dir = (
        args.temp_dir if hasattr(args, "temp_dir") and args.temp_dir else None
    )
    if not base_temp_dir:
        base_temp_dir, _ = get_best_temp_location(repo_path_abs, num_jobs, debug_mode)

    try:
        for i in range(num_jobs):
            # Atomic unique directory within the chosen base location
            temp_dir = tempfile.mkdtemp(
                prefix=f"clang_opt_repo_{i}_",
                dir=base_temp_dir,
            )
            print(f"  Copying '{repo_path_abs}' to '{temp_dir}'...", file=sys.stderr)
            # Copy contents of the original repo to the temporary directory
            # dirs_exist_ok=True is for Python 3.8+
            _ = shutil.copytree(repo_path_abs, temp_dir, dirs_exist_ok=True)
            # Initialize the temp copy as a git repo so that git ls-files, git diff,
            # and git restore work even when the source was a subdirectory of a repo.
            try:
                _ = run_command(
                    ["git", "init"],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                _ = run_command(
                    ["git", "add", "."],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                _ = run_command(
                    [
                        "git",
                        "-c",
                        "user.name=clang-format-auto-infer",
                        "-c",
                        "user.email=test@localhost",
                        "commit",
                        "-m",
                        "initial",
                    ],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                print(
                    f"  Error initializing git repo in '{temp_dir}': {e}",
                    file=sys.stderr,
                )
                exit(1)
            temp_repo_paths.append(temp_dir)
        print("Temporary repositories prepared.", file=sys.stderr)

        # No need for multiprocessing.Manager or shared counter for Nevergrad anymore
        # as the executor handles process management and repo path assignment is now
        # based on the worker's process ID directly.

        print("\nStarting optimization...", file=sys.stderr)

        optimized_options_info = None

        # Build search space and fitness function
        search_space = build_search_space(options_info, lookups, analysis_results)
        initial_config = {k: v.get("value") for k, v in options_info.items()}
        fitness_fn = make_fitness_function(
            repo_path=temp_repo_paths[0],
            process_id=0,
            lookups=lookups,
            base_options=options_info,
            debug=debug_mode,
            file_sample_percentage=args.file_sample_percentage,
            random_seed=RANDOM_SEED,
        )

        if args.optimizer == "genetic":
            best = run_island_ga(
                initial_config=initial_config,
                search_space=search_space,
                fitness_fn=fitness_fn,
                num_islands=args.islands,
                population_size=args.population_size,
                num_iterations=args.iterations,
                migration_interval=args.migration_interval,
                debug=debug_mode,
                random_seed=RANDOM_SEED,
            )
            # Polish with coordinate descent
            if args.polish_passes > 0:
                polished = polish_coordinate_descent(
                    config=best.config,
                    initial_fitness=best.fitness,
                    search_space=search_space,
                    fitness_fn=fitness_fn,
                    max_passes=args.polish_passes,
                    debug=debug_mode,
                )
                best = polished
            optimized_options_info = config_to_flat_options(best.config, options_info)
        elif args.optimizer == "nevergrad":
            result = run_nevergrad_optimization(
                search_space=search_space,
                objective=fitness_fn,
                budget=args.ng_budget,
                num_workers=num_jobs,
                optimizer_name=args.ng_optimizer,
                debug=debug_mode,
                initial_config=initial_config,
            )
            optimized_options_info = config_to_flat_options(
                result.best_config, options_info
            )
        else:
            print(f"Error: Unknown optimizer '{args.optimizer}'.", file=sys.stderr)
            exit(1)

        print("\nOptimization complete.", file=sys.stderr)

        # Generate the final optimized configuration from the modified flat structure
        optimized_config = generate_clang_format_config(optimized_options_info)

        # Output the final configuration
        if args.output_file:
            print(
                f"\nWriting optimized configuration to: {args.output_file}",
                file=sys.stderr,
            )
            try:
                with open(args.output_file, "w") as f:
                    _ = f.write(optimized_config)
                print("Optimized configuration written successfully.", file=sys.stderr)
            except OSError as e:
                print(f"Error writing to file {args.output_file}: {e}", file=sys.stderr)
                exit(1)
        else:
            print("\nOptimized configuration:", file=sys.stderr)
            print(optimized_config)

    finally:
        print("\nCleaning up temporary repositories...", file=sys.stderr)
        for temp_dir in temp_repo_paths:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    print(f"  Removed '{temp_dir}'.", file=sys.stderr)
            except OSError as e:
                print(
                    f"Error removing temporary directory '{temp_dir}': {e}",
                    file=sys.stderr,
                )
        print("Cleanup complete.", file=sys.stderr)

        # No multiprocessing manager to shut down anymore


def cmd_fetch_options(args: argparse.Namespace) -> None:
    """Execute the fetch-options subcommand."""
    from src.option_fetcher import (
        build_urls,
        fetch_html_content,
        parse_options,
        resolve_version,
    )

    version: str = resolve_version(args.version)
    urls = build_urls(version)

    html_content = None
    fetched_url: str | None = None
    for url in urls:
        print(f"Fetching from: {url}", file=sys.stderr)
        html_content = fetch_html_content(url)
        if html_content:
            fetched_url = url
            break

    if html_content:
        options_dict = parse_options(html_content)
        if options_dict:
            options_list: list[dict[str, object]] = []
            for name in sorted(options_dict.keys()):
                info = options_dict[name]
                options_list.append(
                    {
                        "name": name,
                        "type": info["type"],
                        "possible_values": info["possible_values"],
                    }
                )

            print(json.dumps(options_list, indent=2))
            print(
                f"Fetched {len(options_list)} options from: {fetched_url}",
                file=sys.stderr,
            )
        else:
            print("[]")
            sys.exit(0)

    else:
        sys.exit(1)


def main() -> None:
    """Parse command-line arguments and dispatch to the appropriate subcommand."""
    parser = argparse.ArgumentParser(
        description="Optimize clang-format configuration for a git repository."
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- optimize subcommand (default) ---
    opt_parser = subparsers.add_parser(
        "optimize",
        help="Optimize clang-format configuration for a git repository.",
    )
    _ = opt_parser.add_argument(
        "repo_path",
        help="Path to the git repository or subdirectory within a git repository to analyze.",
    )
    _ = opt_parser.add_argument(
        "--output",
        dest="output_file",
        help="Path to the file where the optimized configuration will be written (optional). If not provided, output is written to stdout.",
    )
    _ = opt_parser.add_argument(
        "--option-values-json",
        dest="option_values_json_file",
        help="Path to a JSON file containing clang-format options and their possible values (generated by 'main.py fetch-options').",
    )
    _ = opt_parser.add_argument(
        "--forced-options-yaml",
        dest="forced_options_yaml_file",
        help="Path to a YAML file containing options that should be forced to a specific value.",
    )
    _ = opt_parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug output (print commands being executed).",
    )
    _ = opt_parser.add_argument(
        "--optimizer",
        choices=["genetic", "nevergrad"],
        default="genetic",
        help="Choose the optimization algorithm (genetic or nevergrad). Default: genetic.",
    )
    _ = opt_parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="[Genetic Algorithm] Number of iterations (generations) for the genetic algorithm.",
    )
    _ = opt_parser.add_argument(
        "--population-size",
        type=int,
        default=4,
        help="[Genetic Algorithm] Total number of individuals across all islands in the genetic algorithm population.",
    )
    _ = opt_parser.add_argument(
        "--islands",
        type=int,
        default=1,
        help="[Genetic Algorithm] Number of independent populations (islands) for the genetic algorithm. Set to 1 for a single population.",
    )
    _ = opt_parser.add_argument(
        "--plot-fitness",
        action="store_true",
        help="[Genetic Algorithm & Nevergrad] Visualize the best fitness score over time for each island/evaluation.",
    )
    _ = opt_parser.add_argument(
        "--polish-passes",
        type=int,
        default=3,
        help="[Genetic Algorithm] Number of coordinate descent polish passes after GA convergence. Set to 0 to disable.",
    )
    _ = opt_parser.add_argument(
        "--migration-interval",
        type=int,
        default=15,
        help="[Genetic Algorithm] Number of generations between island migrations.",
    )
    _ = opt_parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help="[Genetic Algorithm] Save checkpoint every N iterations (0 = disabled).",
    )
    _ = opt_parser.add_argument(
        "--checkpoint-resume",
        type=str,
        default=None,
        help="[Genetic Algorithm] Resume optimization from a checkpoint file.",
    )
    _ = opt_parser.add_argument(
        "--ng-budget",
        type=int,
        default=1000,
        help="[Nevergrad] Total number of evaluations (budget) for the Nevergrad optimizer.",
    )
    _ = opt_parser.add_argument(
        "--ng-optimizer",
        type=str,
        default="OnePlusOne",
        help="[Nevergrad] Name of the Nevergrad optimizer to use (e.g., OnePlusOne, CMA, DE, PSO). See Nevergrad documentation for options.",
    )
    _ = opt_parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel jobs to run for fitness calculation. Each job uses a copy of the repository.",
    )
    _ = opt_parser.add_argument(
        "--start-config-file",
        dest="start_config_file",
        help="Path to an existing .clang-format file to use as the starting configuration for optimization. If not provided, defaults from 'clang-format --dump-config' are used.",
    )
    _ = opt_parser.add_argument(
        "--file-sample-percentage",
        type=float,
        default=100.0,
        help="Percentage of files to randomly sample for fitness calculation (0.0-100.0). Use a lower value for faster but less accurate optimization.",
    )
    _ = opt_parser.add_argument(
        "--no-analyze",
        action="store_true",
        help="Skip automatic convention analysis of source files. By default, conventions are detected and used to constrain the search space.",
    )
    _ = opt_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run convention analysis and print detected conventions as YAML, then exit without optimizing.",
    )
    _ = opt_parser.add_argument(
        "--temp-dir",
        default=None,
        help=(
            "Use a specific directory for temporary repository copies. When omitted, "
            "auto-detects fast backends like /dev/shm (tmpfs) if available."
        ),
    )

    # --- fetch-options subcommand ---
    fetch_parser = subparsers.add_parser(
        "fetch-options",
        help="Fetch clang-format style options from LLVM documentation.",
    )
    _ = fetch_parser.add_argument(
        "--version",
        default=None,
        help=(
            "Clang version to fetch docs for (e.g. '18.1.8', '22.1.7'). "
            "Use 'latest' for the current trunk docs. "
            "Defaults to auto-detecting from the installed clang-format."
        ),
    )

    args = parser.parse_args()

    if args.command == "fetch-options":
        cmd_fetch_options(args)
    else:
        # Default to optimize when no subcommand is given
        if args.command is None and len(sys.argv) > 1:
            # User provided positional args but no subcommand — treat as optimize
            args.command = "optimize"
            args = parser.parse_args(["optimize"] + sys.argv[1:])
        elif args.command is None:
            parser.print_help()
            sys.exit(1)
        cmd_optimize(args)


if __name__ == "__main__":  # pragma: no cover — entry point, tested via imports
    # Note: To run this main script after moving, you should typically run it
    # as a module from the directory *above* src, like:
    # python -m src.main /path/to/repo
    main()
