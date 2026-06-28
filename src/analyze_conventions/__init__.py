"""Analyze a codebase to derive sensible clang-format settings.

Package structure (strangler fig pattern):
- Each detector lives in its own module.
- This __init__ re-exports the full public API so existing imports work unchanged.
"""

from .common import _source_files  # pyright: ignore[reportPrivateUsage]
from .indent import detect_indent_width, detect_use_tab
from .layout import (
    detect_access_modifier_offset,
    detect_column_limit,
    detect_max_empty_lines,
)
from .braces import (
    detect_brace_style,
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
from .analyzer import DetectedOption, analyze, analyze_with_metadata
from .impact import measure_impact

__all__ = [
    "_source_files",
    "analyze",
    "analyze_with_metadata",
    "DetectedOption",
    "detect_access_modifier_offset",
    "detect_allow_short_blocks",
    "detect_allow_short_functions",
    "detect_align_case_labels",
    "detect_align_consecutive_declarations",
    "detect_brace_style",
    "detect_brace_style_control",
    "detect_brace_style_function",
    "detect_column_limit",
    "detect_indent_goto_labels",
    "detect_indent_width",
    "detect_language",
    "detect_max_empty_lines",
    "detect_numeric_literal_case",
    "detect_qualifier_alignment",
    "detect_reflow_comments",
    "detect_sort_includes",
    "detect_trailing_comment_style",
    "detect_use_tab",
    "measure_impact",
]
