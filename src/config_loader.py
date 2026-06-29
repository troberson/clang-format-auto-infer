import json
import yaml
import os
import subprocess
import sys
from typing import Any


def load_json_option_values(file_path):
    """
    Loads clang-format option values and possible values from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: A dictionary mapping option names to their info (including possible_values).
              Returns an empty dict if file_path is None. Exits on error.
    """
    if not file_path:
        return {}

    if not os.path.exists(file_path):
        print(
            f"Error: Option values JSON file not found at '{file_path}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        with open(file_path) as f:
            json_list = json.load(f)
            if not isinstance(json_list, list):
                print(
                    f"Error: JSON file '{file_path}' does not contain a list.",
                    file=sys.stderr,
                )
                sys.exit(1)
            # Create a lookup dictionary by option name
            json_options_lookup = {
                item["name"]: item for item in json_list if "name" in item
            }

            # Language: "None" is a valid clang-format enum value but means
            # "do not use" — it disables language detection and should never
            # be selected by the optimizer.
            if "Language" in json_options_lookup:
                values = json_options_lookup["Language"].get("possible_values")
                if values and "None" in values:
                    json_options_lookup["Language"]["possible_values"] = [
                        v for v in values if v != "None"
                    ]

            # InsertTrailingCommas is JavaScript-only and conflicts with
            # BinPackArguments. This tool only formats C/C++/ObjC files,
            # so exclude it entirely.
            # JavaScriptQuotes and JavaScriptWrapImports are also JS-only.
            for js_option in (
                "InsertTrailingCommas",
                "JavaScriptQuotes",
                "JavaScriptWrapImports",
            ):
                json_options_lookup.pop(js_option, None)

            # Post-process: fill in missing sub-options by cross-referencing
            # with clang-format --dump-config. The HTML fetcher may miss some
            # nested options (e.g., AlignConsecutiveAssignments.AcrossComments)
            # that clang-format exposes as flat dot-notation keys.
            _fill_missing_sub_options(json_options_lookup)

        print(f"Successfully loaded option values from '{file_path}'.", file=sys.stderr)
        return json_options_lookup
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{file_path}': {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:  # pragma: no cover
        print(
            f"Error reading option values JSON file '{file_path}': {e}", file=sys.stderr
        )
        sys.exit(1)


def _fill_missing_sub_options(json_options_lookup: dict[str, Any]) -> None:
    """Add missing sub-options by querying clang-format --dump-config.

    The HTML fetcher may miss nested options that clang-format exposes as
    flat dot-notation keys (e.g., AlignConsecutiveAssignments.AcrossComments).
    This function discovers those gaps and fills them with inferred types.

    Args:
        json_options_lookup: Dict to mutate in-place.
    """
    try:
        result = subprocess.run(
            ["clang-format", "--dump-config"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return  # pragma: no cover

        import yaml as _yaml

        cfg = _yaml.safe_load(result.stdout)
        if not cfg:
            return  # pragma: no cover

        def _flatten(d, prefix=""):
            keys = []
            for k, v in d.items():
                full = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
                if isinstance(v, dict):
                    keys.extend(_flatten(v, full))
                else:
                    keys.append((full, type(v).__name__, v))
            return keys

        flat = _flatten(cfg)
        # Options that are explicitly excluded and should not be re-added.
        _excluded = {
            "InsertTrailingCommas",
            "JavaScriptQuotes",
            "JavaScriptWrapImports",
        }
        # Only fill in sub-options whose parent exists in the JSON lookup.
        # This avoids adding entries for options the fetcher never saw.
        _parent_names = {k.rsplit(".", 1)[0] for k in json_options_lookup if "." in k}
        _parent_names.update(k for k in json_options_lookup if "." not in k)

        for name, py_type, _value in flat:
            if name in _excluded:
                continue

            existing = json_options_lookup.get(name)
            # Fill possible_values if missing or empty.
            if existing is not None:
                if existing.get("possible_values"):
                    continue
                # Option exists in JSON but has no possible_values.
                # Only fill booleans — integers get a wide range of valid
                # values depending on the option, so a generic range is unsafe.
                if py_type == "bool":  # pragma: no cover
                    existing["possible_values"] = ["true", "false"]
                continue

            # Only add new entries if the parent option exists in the JSON lookup.
            parent = name.rsplit(".", 1)[0] if "." in name else name
            if parent not in json_options_lookup:
                continue
            # Infer clang-format type from Python type.
            # Only fill booleans — integers are left without possible_values
            # because a generic range is unsafe for options like ColumnLimit.
            if py_type == "bool":
                json_options_lookup[name] = {
                    "type": "bool",
                    "possible_values": ["true", "false"],
                }
            # Strings/lists are left without possible_values; the optimizer
            # will treat them as fixed at their dump-config value.
    except (
        subprocess.TimeoutExpired,
        FileNotFoundError,
        Exception,
    ):  # pragma: no cover
        # If clang-format is unavailable or fails, just skip this step.
        pass


def load_forced_options(file_path):
    """
    Loads forced clang-format options from a YAML file and flattens the structure.
    Nested dictionaries are represented with dot-separated keys (e.g., "Parent.SubOption").

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: A flat dictionary mapping dot-separated option names to their forced values.
              Returns an empty dict if file_path is None. Exits on error.
    """
    if not file_path:
        return {}

    if not os.path.exists(file_path):
        print(
            f"Error: Forced options YAML file not found at '{file_path}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        with open(file_path) as f:
            raw_forced_options = yaml.safe_load(f)
            if not isinstance(raw_forced_options, dict):
                print(
                    f"Error: YAML file '{file_path}' does not contain a dictionary.",
                    file=sys.stderr,
                )
                sys.exit(1)

        flat_forced_options = {}

        def flatten_dict(data, current_path=""):
            for key, value in data.items():
                full_path = f"{current_path}.{key}" if current_path else key
                if isinstance(value, dict):
                    flatten_dict(value, full_path)  # Recurse for nested dicts
                else:
                    flat_forced_options[full_path] = value

        flatten_dict(raw_forced_options)

        print(
            f"Successfully loaded and flattened forced options from '{file_path}'.",
            file=sys.stderr,
        )
        return flat_forced_options
    except yaml.YAMLError as e:
        print(f"Error parsing YAML from '{file_path}': {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:  # pragma: no cover
        print(
            f"Error reading forced options YAML file '{file_path}': {e}",
            file=sys.stderr,
        )
        sys.exit(1)
