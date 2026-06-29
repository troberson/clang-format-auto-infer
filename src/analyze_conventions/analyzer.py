"""Main analyzer that wires all detectors together."""

from dataclasses import dataclass
from typing import Any, override

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


@dataclass
class DetectedOption:
    """A detected clang-format option with metadata.

    Attributes:
        value: The detected value for this option.
        confidence: How the value was determined.
            - 'detected': Strong signal from source analysis.
            - 'guessed': No observable signal; default assumed.
            - 'forced': Always emitted regardless of source (e.g. Custom for
              dependency-gated options).
        tier: Optimization phase this option belongs to.
            - 'resolve': Guessed values that need empirical disambiguation.
            - 'structure': High-impact options that shape the fitness landscape.
            - 'polish': Lower-impact options for final tuning.
    """

    value: Any
    confidence: str  # "detected" | "guessed" | "forced"
    tier: str = "polish"  # "resolve" | "structure" | "polish"

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, DetectedOption):
            return self.value == other.value
        return self.value == other

    @override
    def __hash__(self) -> int:
        return hash(self.value)


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
    result["QualifierAlignment"] = "Custom"
    result["QualifierOrder"] = ["inline", "static", "type", "const"]

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


def _confidence_for_indent_width(files: list[str]) -> str:
    """Return confidence level for indent width detection.

    Returns 'guessed' when tabs dominate (pure-tab code has no observable
    indent unit), 'detected' when space-based indentation provides a signal.
    """
    tab_lines = 0
    space_lines = 0
    for fpath in files:
        try:
            with open(fpath, errors="replace") as f:
                for line in f:
                    stripped = line.lstrip(" \t")
                    leading = line[: len(line) - len(stripped)]
                    if not leading:
                        continue
                    if "\t" in leading:
                        tab_lines += 1
                    elif leading:
                        space_lines += 1
        except OSError:  # pragma: no cover
            continue
    total = tab_lines + space_lines
    if total == 0:
        return "guessed"
    if tab_lines > space_lines:
        return "guessed"  # pure-tab, unknowable
    return "detected"


def analyze_with_metadata(
    path: str,
    percentile: float = 0.95,
) -> dict[str, DetectedOption]:
    """Analyze a codebase and return detected conventions with metadata.

    Strangler fig: this is the new API alongside the existing analyze().
    Returns DetectedOption objects with confidence and tier metadata.

    Args:
        path: Path to the source directory to analyze.
        percentile: Percentile for column limit detection.

    Returns:
        Dict of option_name -> DetectedOption.
    """
    files = _source_files(path)
    if not files:
        return {}

    indent_width = detect_indent_width(files)
    indent_confidence = _confidence_for_indent_width(files)
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

    result: dict[str, DetectedOption] = {}

    # Always-detected options
    result["Language"] = DetectedOption(language, "forced")
    result["QualifierAlignment"] = DetectedOption(qualifier_align, "detected")

    # IndentWidth — guessed when pure-tab
    result["IndentWidth"] = DetectedOption(
        indent_width,
        indent_confidence,
        "resolve" if indent_confidence == "guessed" else "structure",
    )
    result["ColumnLimit"] = DetectedOption(column_limit, "detected", "structure")

    if access_offset is not None:
        result["AccessModifierOffset"] = DetectedOption(access_offset, "detected")
    result["MaxEmptyLinesToKeep"] = DetectedOption(max_empty, "detected")

    # Forced invariants — Custom for dependency-gated options
    result["BreakBeforeBraces"] = DetectedOption("Custom", "forced")
    result["SpaceBeforeParens"] = DetectedOption("Custom", "forced")
    result["SpacesInParens"] = DetectedOption("Custom", "forced")
    result["QualifierAlignment"] = DetectedOption("Custom", "forced")
    result["QualifierOrder"] = DetectedOption(
        ["inline", "static", "type", "const"], "forced"
    )

    # Brace wrapping sub-options — detected from source
    if brace_control != "Leave":
        result["BraceWrapping.AfterControlStatement"] = DetectedOption(
            "Never" if brace_control == "Attach" else True,
            "detected",
            "structure",
        )
    if brace_function != "Leave":
        result["BraceWrapping.AfterFunction"] = DetectedOption(
            False if brace_function == "Attach" else True,
            "detected",
            "structure",
        )

    # Reflow comments
    if reflow_comments != "Leave":
        result["ReflowComments"] = DetectedOption(reflow_comments, "detected")

    # Tab vs spaces
    if use_tab != "Leave":
        result["UseTab"] = DetectedOption(use_tab, "detected", "structure")
        if use_tab == "Always":
            result["TabWidth"] = DetectedOption(
                indent_width or 4,
                indent_confidence,
                "resolve" if indent_confidence == "guessed" else "structure",
            )

    # Include sorting
    result["SortIncludes.Enabled"] = DetectedOption(sort_includes, "detected")

    # NumericLiteralCase sub-options
    if hex_digit_case != "Leave":
        result["NumericLiteralCase.HexDigit"] = DetectedOption(
            hex_digit_case, "detected"
        )
    if prefix_case != "Leave":
        result["NumericLiteralCase.Prefix"] = DetectedOption(prefix_case, "detected")
    if exponent_case != "Leave":
        result["NumericLiteralCase.ExponentLetter"] = DetectedOption(
            exponent_case, "detected"
        )
    if suffix_case != "Leave":
        result["NumericLiteralCase.Suffix"] = DetectedOption(suffix_case, "detected")

    # BSD-style churn options
    if allow_short_functions != "Leave":
        result["AllowShortFunctionsOnASingleLine"] = DetectedOption(
            allow_short_functions, "detected", "structure"
        )
    if allow_short_blocks != "Leave":
        result["AllowShortBlocksOnASingleLine"] = DetectedOption(
            allow_short_blocks, "detected", "structure"
        )
    if align_consecutive_decls is not None:
        result["AlignConsecutiveDeclarations.Enabled"] = DetectedOption(
            align_consecutive_decls, "detected"
        )
    if indent_goto_labels is not None:
        result["IndentGotoLabels"] = DetectedOption(indent_goto_labels, "detected")
    if align_case_labels is not None:
        result["AlignConsecutiveShortCaseStatements.Enabled"] = DetectedOption(
            align_case_labels, "detected"
        )
    if trailing_comment_style != "Leave":
        result["AlignTrailingComments.Kind"] = DetectedOption(
            trailing_comment_style, "detected", "structure"
        )

    return result
