from bs4 import BeautifulSoup, Tag
import argparse
import requests
import subprocess
import sys
import re
import json


OptionDict = dict[str, str | list[str] | None]


def get_clang_format_version():
    """Detect the installed clang-format version string (e.g. '22.1.7')."""
    try:
        result = subprocess.run(
            ["clang-format", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        # Output looks like "Debian clang-format version 22.1.7 (1)"
        match = re.search(r"(\d+\.\d+\.\d+)", result.stdout)
        if match:
            return match.group(1)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return None


def build_urls(version_hint: str) -> list[str]:
    """Build a list of candidate URLs for the given version hint, ordered by preference.

    - 'latest'  -> [trunk docs]
    - 'X.Y.Z'   -> [X.Y.Z docs, X.Y.0 docs, trunk docs]

    Note: Release docs live under tools/clang/docs/, while trunk uses docs/.
    """
    if version_hint == "latest":
        return ["https://clang.llvm.org/docs/ClangFormatStyleOptions.html"]

    parts = version_hint.split(".")
    candidates: list[str] = []
    # Try exact version first
    candidates.append(
        f"https://releases.llvm.org/{version_hint}/tools/clang/docs/ClangFormatStyleOptions.html"
    )
    # Try major.minor.0 release
    if len(parts) >= 2:
        candidates.append(
            f"https://releases.llvm.org/{parts[0]}.{parts[1]}.0/tools/clang/docs/ClangFormatStyleOptions.html"
        )
    # Fallback to trunk
    candidates.append("https://clang.llvm.org/docs/ClangFormatStyleOptions.html")
    return candidates


def resolve_version(arg_version: str | None) -> str:
    """Resolve the effective version string.

    If arg_version is None, auto-detect from the installed clang-format.
    Falls back to 'latest' if detection fails.
    """
    if arg_version is not None:
        return arg_version
    detected = get_clang_format_version()
    if detected:
        print(f"Auto-detected clang-format version: {detected}", file=sys.stderr)
        return detected
    print(
        "Warning: Could not auto-detect clang-format version. Using latest docs.",
        file=sys.stderr,
    )
    return "latest"


def fetch_html_content(url: str) -> str | None:
    """Fetches HTML content from a given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}", file=sys.stderr)
        return None


def _extract_option_details(
    container_tag: Tag, parent_name: str | None = None
) -> list[OptionDict]:
    """
    Extracts option details (name, type, possible values) from a given BeautifulSoup tag.
    Handles both top-level options (from dd tags) and nested options (from li tags).
    Recursively finds and extracts nested options.

    Args:
        container_tag (Tag): The BeautifulSoup tag (dd or li) containing the option's details.
        parent_name (str, optional): The name of the parent option, if this is a nested option.

    Returns:
        list: A list of dictionaries, where each dictionary represents an option
              (including itself and any nested options).
    """
    options_found: list[OptionDict] = []
    option_name: str | None = None
    option_type: str | None = None
    values: list[str] = []

    # Determine where to find the option name and type
    if container_tag.name == "dd":
        # For top-level options, name and type are in the sibling dt tag
        dt_tag = container_tag.find_previous_sibling("dt")
        if dt_tag and isinstance(dt_tag, Tag):
            strong_tag = dt_tag.find("strong")
            if strong_tag and isinstance(strong_tag, Tag):
                option_name = strong_tag.get_text().strip()

            if strong_tag:
                next_sibling = strong_tag.next_sibling
                while next_sibling:
                    if isinstance(next_sibling, Tag) and next_sibling.name == "code":
                        option_type = next_sibling.get_text().strip()
                        break
                    if isinstance(next_sibling, Tag):
                        break
                    next_sibling = next_sibling.next_sibling

    elif container_tag.name == "li":
        # For nested options, name and type are in the first code tag within the li
        code_tag_declaration = container_tag.find("code")
        if code_tag_declaration and isinstance(code_tag_declaration, Tag):
            declaration_text = code_tag_declaration.get_text().strip()
            parts = declaration_text.split(" ", 1)
            if len(parts) == 2:
                option_type = parts[0].strip()
                option_name = parts[1].strip()
            else:
                # If it doesn't fit 'type name' pattern, it's likely not a valid nested option declaration.
                # Skip processing this li as an option, as per user request for examples/descriptions.
                return []  # This li is not a valid option, so return empty list

    if not option_name or not option_type:
        # If we couldn't get a name/type, we can't process this option
        # This warning is for cases where the dt/dd or li structure is fundamentally broken for an option.
        if container_tag.name == "dd":
            print(
                "Warning: Could not parse top-level option name/type from dt/dd pair.",
                file=sys.stderr,
            )
        # No warning for li here, as handled above by returning []
        return []  # Return empty list if parsing failed for this item

    # Construct the full option name (e.g., "BraceWrapping.AfterCaseLabel")
    full_option_name: str = (
        f"{parent_name}.{option_name}" if parent_name else option_name
    )

    # --- Extract possible values for the current option ---
    if option_type == "Boolean":
        values = ["true", "false"]
    else:
        # Find the ul that contains nested options, if any (only relevant for dd tags)
        nested_options_ul: Tag | None = None
        if container_tag.name == "dd":
            nested_heading = container_tag.find(
                "p", string=re.compile(r"Nested configuration flags:")
            )
            if nested_heading and isinstance(nested_heading, Tag):
                found_ul = nested_heading.find_next_sibling("ul")
                if found_ul and isinstance(found_ul, Tag):
                    nested_options_ul = found_ul

        # Look for lists of values. These are typically ul tags.
        # We need to distinguish them from the ul containing nested options.
        for ul_tag in container_tag.find_all(
            "ul", recursive=False
        ):  # Only direct ul children
            if ul_tag == nested_options_ul:
                continue  # This ul contains nested options, not values for the current option

            # Heuristic to identify a list of values:
            # Check if the ul contains li elements, and those li elements contain a code tag
            # whose text is a single word or matches the "(in configuration: ...)" pattern.
            potential_values_found = False
            current_ul_values: list[str] = []
            for li in ul_tag.find_all("li", recursive=False):
                li_text: str = li.get_text().strip()
                config_value_match = re.search(
                    r"\(in configuration:\s*(.*?)\)", li_text
                )
                if config_value_match:
                    current_ul_values.append(config_value_match.group(1).strip())
                    potential_values_found = True
                else:
                    code_tag_value = li.find("code", recursive=False)
                    if code_tag_value and isinstance(code_tag_value, Tag):
                        code_text = code_tag_value.get_text().strip()
                        if " " not in code_text or not re.match(
                            r"^\w+\s+\w+$", code_text
                        ):
                            current_ul_values.append(code_text)
                            potential_values_found = True
                    # else: if no code tag, it's probably not a value list

            if potential_values_found:
                values.extend(current_ul_values)
                # Assuming there's only one primary list of values for an option
                break

    # Add sane values for Integer/Unsigned options related to Offset or Width
    if option_type in ["Integer", "Unsigned"]:
        if "offset" in full_option_name.lower() or "width" in full_option_name.lower():
            sane_int_values: list[int] = []
            if option_type == "Integer":
                sane_int_values = [-4, -2, 0, 1, 2, 3, 4, 8]
            elif option_type == "Unsigned":
                sane_int_values = [0, 1, 2, 3, 4, 8]

            all_values_set = set(values)
            for val in sane_int_values:
                all_values_set.add(str(val))
            values = sorted(
                list(all_values_set),
                key=lambda x: int(x) if x.lstrip("-").isdigit() else x,
            )

    # Add the current option to the list of found options
    options_found.append(
        {
            "name": full_option_name,
            "type": option_type,
            "possible_values": values if values else None,
        }
    )

    # --- Check for nested options within the current container_tag (only for dd tags) ---
    if container_tag.name == "dd":
        nested_heading = container_tag.find(
            "p", string=re.compile(r"Nested configuration flags:")
        )
        if nested_heading and isinstance(nested_heading, Tag):
            nested_ul = nested_heading.find_next_sibling("ul")
            if nested_ul and isinstance(nested_ul, Tag):
                for li_tag in nested_ul.find_all("li", recursive=False):
                    if isinstance(li_tag, Tag):
                        nested_options = _extract_option_details(
                            li_tag, full_option_name
                        )
                        options_found.extend(nested_options)

    return options_found


def parse_options(html_content: str) -> dict[str, OptionDict]:
    """Parses HTML content to extract clang-format options and their values."""
    soup = BeautifulSoup(html_content, "lxml")  # Use lxml parser

    all_options_data: dict[str, OptionDict] = {}

    options_section_heading = None
    for h2_tag in soup.find_all("h2"):
        if h2_tag.get_text(strip=True).startswith("Configurable Format Style Options"):
            options_section_heading = h2_tag
            break

    if not options_section_heading:
        print(
            "Could not find the 'Configurable Format Style Options' section.",
            file=sys.stderr,
        )
        return all_options_data

    current_element = options_section_heading.find_next_sibling()
    first_dl = None
    while current_element:
        if current_element.name == "dl":  # type: ignore
            first_dl = current_element
            break
        if current_element.name == "h2":  # type: ignore
            break
        current_element = current_element.find_next_sibling()

    if not first_dl:
        print(
            "Could not find the start of the options list (<dl> tag) after the heading.",
            file=sys.stderr,
        )
        return all_options_data

    current_dl = first_dl
    while current_dl and isinstance(current_dl, Tag):
        if current_dl.name == "dl":
            dd_tag = current_dl.find("dd")
            if dd_tag and isinstance(dd_tag, Tag):
                extracted_options = _extract_option_details(dd_tag)
                for option_info in extracted_options:
                    name = option_info["name"]
                    assert isinstance(name, str)
                    all_options_data[name] = {
                        "type": option_info["type"],
                        "possible_values": option_info["possible_values"],
                    }

        current_dl = current_dl.find_next_sibling()
        if current_dl and isinstance(current_dl, Tag) and current_dl.name == "h2":
            break

    return all_options_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch clang-format style options from LLVM documentation."
    )
    _ = parser.add_argument(
        "--version",
        default=None,
        help=(
            "Clang version to fetch docs for (e.g. '18.1.8', '22.1.7'). "
            "Use 'latest' for the current trunk docs. "
            "Defaults to auto-detecting from the installed clang-format."
        ),
    )
    args = parser.parse_args()

    version: str = resolve_version(args.version)  # type: ignore[arg-type]
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
            # Sort options by name for consistent output
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
