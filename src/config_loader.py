import json
import yaml
import os
import sys


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
