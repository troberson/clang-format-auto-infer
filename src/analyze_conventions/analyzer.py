"""Main analyzer that wires all detectors together."""

from typing import Any

from .common import _source_files  # pyright: ignore[reportPrivateUsage]
from .indent import detect_indent_width, detect_use_tab
from .layout import (
    detect_access_modifier_offset,
    detect_column_limit,
    detect_max_empty_lines,
)
from .braces import (
    detect_brace_style_control,
    detect_brace_style_function,
)
from .comments import detect_reflow_comments
from .literals import detect_numeric_literal_case
from .includes import detect_sort_includes
from .language import detect_language
from .qualifiers import detect_qualifier_alignment
from .style import (
    detect_allow_short_functions,
    detect_allow_short_blocks,
    detect_align_consecutive_declarations,
    detect_indent_goto_labels,
    detect_align_case_labels,
    detect_trailing_comment_style,
)


def analyze(
    path: str,
    percentile: float = 0.95,
) -> dict[str, Any]:
    """Analyze a codebase and return detected conventions as a structured dict.

    Returns a dict mapping clang-format option names to detected values.
    Options with value 'Leave' or None are excluded.

    Args:
        path: Path to the source directory to analyze.
        percentile: Percentile for column limit detection.

    Returns:
        Dict of option_name -> detected_value.
    """
    files = _source_files(path)
    if not files:
        return {}

    indent_width = detect_indent_width(files)
    column_limit = detect_column_limit(files, percentile, indent_width or 4)
    language = detect_language(files)
    qualifier_align = detect_qualifier_alignment(files)
    access_offset = detect_access_modifier_offset(files)
    max_empty = detect_max_empty_lines(files)
    hex_digit_case, prefix_case, exponent_case, suffix_case = (
        detect_numeric_literal_case(files)
    )
    brace_control = detect_brace_style_control(files)
    brace_function = detect_brace_style_function(files)
    use_tab = detect_use_tab(files)
    reflow_comments = detect_reflow_comments(files, column_limit)
    sort_includes = detect_sort_includes(files)
    allow_short_functions = detect_allow_short_functions(files)
    allow_short_blocks = detect_allow_short_blocks(files)
    align_consecutive_decls = detect_align_consecutive_declarations(files)
    indent_goto_labels = detect_indent_goto_labels(files)
    align_case_labels = detect_align_case_labels(files)
    trailing_comment_style = detect_trailing_comment_style(files)

    result: dict[str, Any] = {}
    result["Language"] = language
    result["QualifierAlignment"] = qualifier_align
    result["IndentWidth"] = indent_width
    result["ColumnLimit"] = column_limit
    if access_offset is not None:
        result["AccessModifierOffset"] = access_offset
    result["MaxEmptyLinesToKeep"] = max_empty

    # Always force Custom for options that have sub-option families.
    # Without Custom, all sub-options are silently ignored by clang-format.
    result["BreakBeforeBraces"] = "Custom"
    result["SpaceBeforeParens"] = "Custom"
    result["SpacesInParens"] = "Custom"

    # Brace wrapping sub-options — detected from source.
    if brace_control != "Leave":
        result["BraceWrapping.AfterControlStatement"] = (
            "Never" if brace_control == "Attach" else True
        )
    if brace_function != "Leave":
        result["BraceWrapping.AfterFunction"] = (
            False if brace_function == "Attach" else True
        )

    # Reflow comments
    if reflow_comments != "Leave":
        result["ReflowComments"] = reflow_comments

    # Tab vs spaces
    if use_tab != "Leave":
        result["UseTab"] = use_tab
        # When using tabs, TabWidth must match IndentWidth for clang-format
        # to actually use tabs for indentation. Otherwise it falls back to
        # spaces, causing massive whitespace churn.
        if use_tab == "Always":
            result["TabWidth"] = indent_width or 4

    # Include sorting
    result["SortIncludes.Enabled"] = sort_includes

    # NumericLiteralCase sub-options
    if hex_digit_case != "Leave":
        result["NumericLiteralCase.HexDigit"] = hex_digit_case
    if prefix_case != "Leave":
        result["NumericLiteralCase.Prefix"] = prefix_case
    if exponent_case != "Leave":
        result["NumericLiteralCase.ExponentLetter"] = exponent_case
    if suffix_case != "Leave":
        result["NumericLiteralCase.Suffix"] = suffix_case

    # BSD-style churn options
    if allow_short_functions != "Leave":
        result["AllowShortFunctionsOnASingleLine"] = allow_short_functions
    if allow_short_blocks != "Leave":
        result["AllowShortBlocksOnASingleLine"] = allow_short_blocks
    if align_consecutive_decls is not None:
        result["AlignConsecutiveDeclarations.Enabled"] = align_consecutive_decls
    if indent_goto_labels is not None:
        result["IndentGotoLabels"] = indent_goto_labels
    if align_case_labels is not None:
        result["AlignConsecutiveShortCaseStatements.Enabled"] = align_case_labels
    if trailing_comment_style != "Leave":
        result["AlignTrailingComments.Kind"] = trailing_comment_style

    return result
