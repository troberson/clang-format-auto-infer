import yaml
import sys
import subprocess
import json
from typing import Any
from .utils import run_command

# Info dict stored per option: {'type': str, 'value': <yaml value>}
OptionInfo = dict[str, str | Any]

# Recursive nested config: dict[str, Any]
NestedConfig = dict[str, Any]


def get_clang_format_options(debug: bool = False) -> str | None:
    """
    Runs 'clang-format --dump-config' to get a list of all possible options.

    Args:
        debug (bool): Enable debug output for run_command.

    Returns:
        str: The output from 'clang-format --dump-config'.
        None: If clang-format is not found or an error occurs.
    """
    cmd = ["clang-format", "--dump-config"]
    try:
        result = run_command(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            debug=debug,  # Pass debug flag
        )
        return result.stdout
    except FileNotFoundError:
        print(
            "Error: clang-format command not found. Please ensure it is installed and in your PATH.",
            file=sys.stderr,
        )
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error running clang-format --dump-config: {e}", file=sys.stderr)
        print(f"Stderr: {e.stderr}", file=sys.stderr)
        return None


def parse_clang_format_options(yaml_string: str) -> dict[str, OptionInfo] | None:
    """
    Parses the YAML output from 'clang-format --dump-config' and flattens it.
    Nested dictionaries are represented with dot-separated keys (e.g., "Parent.SubOption").

    Args:
        yaml_string (str): The YAML output string.

    Returns:
        dict: A flat dictionary mapping dot-separated option names to a dictionary containing
              'type' (str: 'bool', 'int', 'str', 'list', 'dict', etc.) and
              'value' (any: the parsed value).
              Returns None if parsing fails.
    """
    try:
        config = yaml.safe_load(yaml_string)
        if not isinstance(config, dict):
            print(
                "Error: Parsed clang-format config is not a dictionary.",
                file=sys.stderr,
            )
            return None

        flat_options: dict[str, OptionInfo] = {}

        def flatten_dict(data: dict[str, Any], current_path: str = "") -> None:
            for key, value in data.items():
                full_path = f"{current_path}.{key}" if current_path else key
                value_type = type(value).__name__
                if value_type == "dict":
                    flatten_dict(value, full_path)  # Recurse for nested dicts
                else:
                    flat_options[full_path] = {"type": value_type, "value": value}

        flatten_dict(config)
        return flat_options

    except yaml.YAMLError as e:
        print(f"Error parsing YAML output from clang-format: {e}", file=sys.stderr)
        return None


def generate_clang_format_config(flat_options_info: dict[str, OptionInfo]) -> str:
    """
    Generates a YAML string for a .clang-format file from a flat dictionary of options.
    Reconstructs nested dictionaries from dot-separated names.

    Args:
        flat_options_info (dict): A flat dictionary mapping dot-separated option names
                                  to their info dicts (containing 'type' and 'value').

    Returns:
        str: A YAML formatted string.
    """
    nested_config: NestedConfig = {}

    # Conditionally include QualifierOrder only when QualifierAlignment is Custom.
    # clang-format rejects QualifierOrder otherwise with "unknown key" error.
    include_qualifier_order = (
        flat_options_info.get("QualifierAlignment", {}).get("value") == "Custom"
    )

    # Sort keys to ensure consistent output order, especially for nested structures
    # This helps with reproducibility of the generated YAML.
    for full_path in sorted(flat_options_info.keys()):
        info = flat_options_info[full_path]
        # Skip QualifierOrder when QualifierAlignment is not Custom
        if full_path == "QualifierOrder" and not include_qualifier_order:
            continue
        parts = full_path.split(".")
        current_level = nested_config
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                # This is the final part, assign the value
                current_level[part] = info["value"]
            else:
                # This is an intermediate part, ensure it's a dictionary
                if part not in current_level:
                    current_level[part] = {}
                current_level = current_level[part]

    return yaml.dump(nested_config, default_flow_style=False, sort_keys=False)


class IncrementalConfigBuilder:
    """
    Builds a nested config dict once from a flat options dict, then patches
    individual leaf values and serializes to YAML without reconstructing the
    full tree. This avoids O(N) dict reconstruction per value test.

    Thread-safe for single-threaded use (the typical worker case).
    """

    def __init__(self, flat_options_info: dict[str, OptionInfo]) -> None:
        self._nested: NestedConfig = {}
        # Cache the path parts for each option to avoid repeated splits
        self._path_cache: dict[str, list[str]] = {}
        self._build_nested(flat_options_info)

    def _build_nested(self, flat_options_info: dict[str, OptionInfo]) -> None:
        for full_path in sorted(flat_options_info.keys()):
            info = flat_options_info[full_path]
            parts = full_path.split(".")
            self._path_cache[full_path] = parts
            current_level: NestedConfig = self._nested
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    current_level[part] = info["value"]
                else:
                    if part not in current_level:
                        current_level[part] = {}
                    current_level = current_level[part]

    def set_value(self, full_option_path: str, value: Any) -> str:
        """
        Set a single option value in the cached nested dict and return YAML.

        Args:
            full_option_path: Dot-separated option path (e.g. "IndentWidth").
            value: The new value to set.

        Returns:
            YAML string with the updated value.
        """
        parts = self._path_cache.get(full_option_path)
        if parts is None:
            # Fallback: path not in cache, compute on the fly
            parts = full_option_path.split(".")
            self._path_cache[full_option_path] = parts

        current_level: NestedConfig = self._nested
        for part in parts[:-1]:
            current_level = current_level[part]
        current_level[parts[-1]] = value

        return yaml.dump(self._nested, default_flow_style=False, sort_keys=False)

    def build(self) -> str:
        """
        Serialize the current nested dict to YAML without modifying values.
        """
        return yaml.dump(self._nested, default_flow_style=False, sort_keys=False)

    def get_value(self, full_option_path: str) -> Any:
        """Get the current value of an option from the cached nested dict."""
        parts = self._path_cache.get(full_option_path)
        if parts is None:
            parts = full_option_path.split(".")
            self._path_cache[full_option_path] = parts

        current_level: NestedConfig = self._nested
        for part in parts[:-1]:
            current_level = current_level[part]
        return current_level[parts[-1]]


def save_checkpoint(
    checkpoint_path: str,
    best_config: dict[str, Any],
    best_fitness: float,
    iteration: int,
    populations: list[list[dict[str, Any]]],
    fitness_history_per_island: list[list[float]],
) -> None:
    """
    Save optimization state to disk so it can be resumed after interruption.

    Args:
        checkpoint_path: File path to write checkpoint data.
        best_config: Best flat config dict found so far.
        best_fitness: Fitness score of best_config.
        iteration: 0-based iteration index when checkpoint was saved.
        populations: Current island populations.
        fitness_history_per_island: Fitness history for each island.
    """
    data = {
        "best_config": best_config,
        "best_fitness": best_fitness,
        "iteration": iteration,
        "populations": populations,
        "fitness_history_per_island": fitness_history_per_island,
    }
    with open(checkpoint_path, "w") as f:
        json.dump(data, f)


def load_checkpoint(checkpoint_path: str) -> dict[str, Any] | None:
    """
    Load optimization state from a checkpoint file.

    Args:
        checkpoint_path: File path to read checkpoint data from.

    Returns:
        dict with keys: best_config, best_fitness, iteration, populations,
        fitness_history_per_island. Returns None if file not found or invalid.
    """
    try:
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
