"""Tests for src/option_fetcher — option fetching and parsing."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.option_fetcher import (  # noqa: E402
    build_urls,
    fetch_html_content,
    get_clang_format_version,
    parse_options,
    resolve_version,
)

# ---------------------------------------------------------------------------
# build_urls
# ---------------------------------------------------------------------------


class TestBuildUrls:
    def test_latest_returns_trunk(self):
        urls = build_urls("latest")
        assert urls == ["https://clang.llvm.org/docs/ClangFormatStyleOptions.html"]

    def test_exact_version_returns_candidates(self):
        urls = build_urls("18.1.8")
        assert urls == [
            "https://releases.llvm.org/18.1.8/tools/clang/docs/ClangFormatStyleOptions.html",
            "https://releases.llvm.org/18.1.0/tools/clang/docs/ClangFormatStyleOptions.html",
            "https://clang.llvm.org/docs/ClangFormatStyleOptions.html",
        ]

    def test_single_part_version_falls_back(self):
        urls = build_urls("18")
        assert urls == [
            "https://releases.llvm.org/18/tools/clang/docs/ClangFormatStyleOptions.html",
            "https://clang.llvm.org/docs/ClangFormatStyleOptions.html",
        ]

    def test_two_part_version(self):
        urls = build_urls("18.1")
        assert urls == [
            "https://releases.llvm.org/18.1/tools/clang/docs/ClangFormatStyleOptions.html",
            "https://releases.llvm.org/18.1.0/tools/clang/docs/ClangFormatStyleOptions.html",
            "https://clang.llvm.org/docs/ClangFormatStyleOptions.html",
        ]

    def test_trunk_always_last_fallback(self):
        urls = build_urls("99.99.99")
        assert urls[-1] == "https://clang.llvm.org/docs/ClangFormatStyleOptions.html"


# ---------------------------------------------------------------------------
# resolve_version
# ---------------------------------------------------------------------------


class TestResolveVersion:
    def test_explicit_version_passed_through(self):
        assert resolve_version("18.1.8") == "18.1.8"

    def test_explicit_latest_passed_through(self):
        assert resolve_version("latest") == "latest"

    @patch("src.option_fetcher.get_clang_format_version")
    def test_none_auto_detects(self, mock_get):
        mock_get.return_value = "22.1.7"
        assert resolve_version(None) == "22.1.7"

    @patch("src.option_fetcher.get_clang_format_version")
    def test_none_falls_back_to_latest(self, mock_get, capsys):
        mock_get.return_value = None
        result = resolve_version(None)
        assert result == "latest"
        captured = capsys.readouterr()
        assert "Could not auto-detect" in captured.err

    @patch("src.option_fetcher.get_clang_format_version")
    def test_auto_detect_prints_version(self, mock_get, capsys):
        mock_get.return_value = "18.1.8"
        _ = resolve_version(None)
        captured = capsys.readouterr()
        assert "Auto-detected clang-format version: 18.1.8" in captured.err


# ---------------------------------------------------------------------------
# get_clang_format_version
# ---------------------------------------------------------------------------


class TestGetClangFormatVersion:
    @patch("src.option_fetcher.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="Debian clang-format version 22.1.7 (1)"
        )
        assert get_clang_format_version() == "22.1.7"

    @patch("src.option_fetcher.subprocess.run")
    def test_file_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        assert get_clang_format_version() is None

    @patch("src.option_fetcher.subprocess.run")
    def test_called_process_error(self, mock_run):
        import subprocess

        mock_run.side_effect = subprocess.CalledProcessError(1, "clang-format")
        assert get_clang_format_version() is None

    @patch("src.option_fetcher.subprocess.run")
    def test_no_version_match(self, mock_run):
        mock_run.return_value = MagicMock(stdout="no version here")
        assert get_clang_format_version() is None


# ---------------------------------------------------------------------------
# fetch_html_content
# ---------------------------------------------------------------------------


class TestFetchHtmlContent:
    @patch("src.option_fetcher.requests.get")
    def test_success(self, mock_get):
        mock_get.return_value = MagicMock(text="<html>ok</html>", status_code=200)
        assert fetch_html_content("http://example.com") == "<html>ok</html>"

    @patch("src.option_fetcher.requests.get")
    def test_http_error(self, mock_get, capsys):
        import requests

        mock_get.side_effect = requests.exceptions.HTTPError("404")
        assert fetch_html_content("http://example.com") is None
        captured = capsys.readouterr()
        assert "Error fetching URL" in captured.err

    @patch("src.option_fetcher.requests.get")
    def test_connection_error(self, mock_get, capsys):
        import requests

        mock_get.side_effect = requests.exceptions.ConnectionError("refused")
        assert fetch_html_content("http://example.com") is None
        captured = capsys.readouterr()
        assert "Error fetching URL" in captured.err


# ---------------------------------------------------------------------------
# parse_options
# ---------------------------------------------------------------------------

SAMPLE_HTML = """
<html>
<body>
<h2>Configurable Format Style Options</h2>
<dl>
<dt><strong>IndentWidth</strong> <code>unsigned</code></dt>
<dd>The column number to indent blocks.</dd>
<dt><strong>UseTab</strong> <code>Boolean</code></dt>
<dd>
<p>How to handle spaces that form part of indentation.</p>
<ul>
<li><code>Never</code></li>
<li><code>ForIndentation</code></li>
<li><code>ForIndentationAndAlignment</code></li>
</ul>
</dd>
<dt><strong>BreakBeforeBraces</strong> <code>Boolean</code></dt>
<dd>
<p>Control where the opening brace of a namespace, class or function body is placed.</p>
<p>Nested configuration flags:</p>
<ul>
<li><code>Boolean NestedNamespace</code></li>
</ul>
</dd>
</dl>
<h2>Another Section</h2>
</body>
</html>
"""


class TestParseOptions:
    def test_parses_basic_options(self):
        result = parse_options(SAMPLE_HTML)
        assert "IndentWidth" in result
        assert result["IndentWidth"]["type"] == "unsigned"

    def test_parses_boolean_with_values(self):
        result = parse_options(SAMPLE_HTML)
        assert "UseTab" in result
        assert result["UseTab"]["type"] == "Boolean"
        assert result["UseTab"]["possible_values"] == ["true", "false"]

    def test_parses_nested_options(self):
        result = parse_options(SAMPLE_HTML)
        assert "BreakBeforeBraces" in result
        assert "BreakBeforeBraces.NestedNamespace" in result
        assert result["BreakBeforeBraces.NestedNamespace"]["type"] == "Boolean"

    def test_stops_at_next_section(self):
        """Options in sections after the heading are not parsed."""
        result = parse_options(SAMPLE_HTML)
        # 3 top-level + 1 nested = 4 options
        assert len(result) == 4

    def test_missing_heading_returns_empty(self, capsys):
        result = parse_options("<html><body><h2>Something Else</h2></body></html>")
        assert result == {}
        captured = capsys.readouterr()
        assert "Could not find" in captured.err

    def test_missing_dl_returns_empty(self, capsys):
        html = "<html><body><h2>Configurable Format Style Options</h2><p>no dl</p></body></html>"
        result = parse_options(html)
        assert result == {}
        captured = capsys.readouterr()
        assert "Could not find" in captured.err

    def test_empty_html_returns_empty(self):
        result = parse_options("")
        assert result == {}

    def test_option_with_no_values(self):
        result = parse_options(SAMPLE_HTML)
        assert result["IndentWidth"]["possible_values"] is None

    def test_dd_with_no_dt_sibling(self, capsys):
        """A dd tag with no preceding dt returns empty and prints warning."""
        html = """
        <html><body>
        <h2>Configurable Format Style Options</h2>
        <dl><dd>orphan dd</dd></dl>
        </body></html>
        """
        result = parse_options(html)
        assert result == {}
        captured = capsys.readouterr()
        assert "Could not parse top-level option" in captured.err

    def test_enum_option_with_values(self):
        """Non-Boolean option with enum values extracted from ul."""
        html = """
        <html><body>
        <h2>Configurable Format Style Options</h2>
        <dl>
        <dt><strong>BreakBeforeBraces</strong> <code>enum</code></dt>
        <dd>
        <ul>
        <li><code>Attach</code></li>
        <li><code>Allman</code></li>
        <li><code>K&R</code></li>
        </ul>
        </dd>
        </dl>
        </body></html>
        """
        result = parse_options(html)
        assert "BreakBeforeBraces" in result
        assert result["BreakBeforeBraces"]["possible_values"] == [
            "Attach",
            "Allman",
            "K&R",
        ]

    def test_enum_option_with_in_configuration_syntax(self):
        """Enum values using '(in configuration: ...)' syntax."""
        html = """
        <html><body>
        <h2>Configurable Format Style Options</h2>
        <dl>
        <dt><strong>AlignConsecutiveAssignments</strong> <code>Boolean</code></dt>
        <dd>
        <ul>
        <li>Align consecutive elements (in configuration: <code>Always</code>)</li>
        <li>Do not align (in configuration: <code>Never</code>)</li>
        </ul>
        </dd>
        </dl>
        </body></html>
        """
        result = parse_options(html)
        assert "AlignConsecutiveAssignments" in result
        values = result["AlignConsecutiveAssignments"]["possible_values"]
        assert values == ["true", "false"]

    def test_integer_offset_gets_sane_values(self):
        """Integer options with 'offset' in name get sane default values."""
        html = """
        <html><body>
        <h2>Configurable Format Style Options</h2>
        <dl>
        <dt><strong>NamespaceIndentation</strong> <code>Integer</code></dt>
        <dd>Control namespace indentation.</dd>
        </dl>
        </body></html>
        """
        result = parse_options(html)
        # No "offset" or "width" in name, so no sane values added
        assert result["NamespaceIndentation"]["possible_values"] is None

    def test_integer_offset_name_gets_sane_values(self):
        """Integer options with 'offset' in name get sane default values."""
        html = """
        <html><body>
        <h2>Configurable Format Style Options</h2>
        <dl>
        <dt><strong>IndentOuterRenamespace</strong> <code>Integer</code></dt>
        <dd>Offset for something.</dd>
        </dl>
        </body></html>
        """
        result = parse_options(html)
        # No "offset" or "width" in name
        assert result["IndentOuterRenamespace"]["possible_values"] is None

    def test_unsigned_width_gets_sane_values(self):
        """Unsigned options with 'width' in name get sane default values."""
        html = """
        <html><body>
        <h2>Configurable Format Style Options</h2>
        <dl>
        <dt><strong>ColumnLimit</strong> <code>Unsigned</code></dt>
        <dd>Max column width.</dd>
        </dl>
        </body></html>
        """
        result = parse_options(html)
        # "width" not in name, but "ColumnLimit" doesn't contain "width"
        assert result["ColumnLimit"]["possible_values"] is None

    def test_unsigned_width_option_gets_sane_values(self):
        """Unsigned options with 'width' in name get sane default values."""
        html = """
        <html><body>
        <h2>Configurable Format Style Options</h2>
        <dl>
        <dt><strong>PenaltyExcessCharacter</strong> <code>Unsigned</code></dt>
        <dd>Width penalty.</dd>
        </dl>
        </body></html>
        """
        result = parse_options(html)
        # No "width" in name
        assert result["PenaltyExcessCharacter"]["possible_values"] is None

    def test_integer_offset_option_gets_sane_values(self):
        """Integer options with 'offset' in name get sane default values."""
        html = """
        <html><body>
        <h2>Configurable Format Style Options</h2>
        <dl>
        <dt><strong>FixMacrosWithConditions</strong> <code>Integer</code></dt>
        <dd>Some offset thing.</dd>
        </dl>
        </body></html>
        """
        result = parse_options(html)
        # No "offset" in name
        assert result["FixMacrosWithConditions"]["possible_values"] is None

    def test_unsigned_width_in_name_gets_sane_values(self):
        """Unsigned options with 'width' in name get sane default values."""
        html = """
        <html><body>
        <h2>Configurable Format Style Options</h2>
        <dl>
        <dt><strong>SomeWidth</strong> <code>Unsigned</code></dt>
        <dd>A width option.</dd>
        </dl>
        </body></html>
        """
        result = parse_options(html)
        values = result["SomeWidth"]["possible_values"]
        assert values is not None
        assert "0" in values
        assert "8" in values

    def test_integer_offset_in_name_gets_sane_values(self):
        """Integer options with 'offset' in name get sane default values."""
        html = """
        <html><body>
        <h2>Configurable Format Style Options</h2>
        <dl>
        <dt><strong>SomeOffset</strong> <code>Integer</code></dt>
        <dd>An offset option.</dd>
        </dl>
        </body></html>
        """
        result = parse_options(html)
        values = result["SomeOffset"]["possible_values"]
        assert values is not None
        assert "-4" in values
        assert "8" in values

    def test_invalid_nested_li_returns_empty(self):
        """A nested li that doesn't match 'type name' pattern returns empty."""
        html = """
        <html><body>
        <h2>Configurable Format Style Options</h2>
        <dl>
        <dt><strong>BraceWrapping</strong> <code>Boolean</code></dt>
        <dd>
        <p>Nested configuration flags:</p>
        <ul>
        <li><code>justaname</code></li>
        </ul>
        </dd>
        </dl>
        </body></html>
        """
        result = parse_options(html)
        assert "BraceWrapping" in result
        # The invalid nested li should not produce a nested option
        assert len(result) == 1

    def test_multiple_dl_blocks(self):
        """Options spread across multiple dl blocks are all parsed."""
        html = """
        <html><body>
        <h2>Configurable Format Style Options</h2>
        <dl>
        <dt><strong>IndentWidth</strong> <code>unsigned</code></dt>
        <dd>Indent width.</dd>
        </dl>
        <dl>
        <dt><strong>UseTab</strong> <code>Boolean</code></dt>
        <dd>Use tabs.</dd>
        </dl>
        <h2>Next Section</h2>
        </body></html>
        """
        result = parse_options(html)
        assert "IndentWidth" in result
        assert "UseTab" in result
        assert len(result) == 2

    def test_enum_values_with_in_configuration_syntax(self):
        """Non-Boolean enum using '(in configuration: ...)' syntax."""
        html = """
        <html><body>
        <h2>Configurable Format Style Options</h2>
        <dl>
        <dt><strong>AccessModifierOffset</strong> <code>Integer</code></dt>
        <dd>
        <ul>
        <li>Align consecutive elements (in configuration: <code>Always</code>)</li>
        </ul>
        </dd>
        </dl>
        </body></html>
        """
        result = parse_options(html)
        values = result["AccessModifierOffset"]["possible_values"]
        assert values is not None
        assert "Always" in values
        # Also has sane offset values
        assert "-4" in values

    def test_dd_with_both_values_ul_and_nested_ul(self):
        """A dd with both a values list and nested options ul."""
        html = """
        <html><body>
        <h2>Configurable Format Style Options</h2>
        <dl>
        <dt><strong>SortIncludes</strong> <code>enum</code></dt>
        <dd>
        <ul>
        <li><code>RemoveRedundant</code></li>
        <li><code>Keep</code></li>
        </ul>
        <p>Nested configuration flags:</p>
        <ul>
        <li><code>Boolean Fallback</code></li>
        </ul>
        </dd>
        </dl>
        </body></html>
        """
        result = parse_options(html)
        assert "SortIncludes" in result
        assert result["SortIncludes"]["possible_values"] == ["RemoveRedundant", "Keep"]
        assert "SortIncludes.Fallback" in result

    def test_dt_with_text_node_before_code(self):
        """Type extraction skips text nodes before finding code tag."""
        html = """
        <html><body>
        <h2>Configurable Format Style Options</h2>
        <dl>
        <dt><strong>IndentWidth</strong> (in bytes) <code>unsigned</code></dt>
        <dd>Indent width.</dd>
        </dl>
        </body></html>
        """
        result = parse_options(html)
        assert "IndentWidth" in result
        assert result["IndentWidth"]["type"] == "unsigned"

    def test_h2_immediately_after_heading_breaks_search(self, capsys):
        """An h2 immediately after the options heading stops the dl search (line 280)."""
        html = """
        <html><body>
        <h2>Configurable Format Style Options</h2>
        <h2>Another Section</h2>
        </body></html>
        """
        result = parse_options(html)
        assert result == {}
        captured = capsys.readouterr()
        assert "Could not find" in captured.err

    def test_non_code_tag_before_code_stops_type_search(self, capsys):
        """A non-code Tag between strong and code stops type extraction (line 125)."""
        html = """
        <html><body>
        <h2>Configurable Format Style Options</h2>
        <dl>
        <dt><strong>IndentWidth</strong> <em>deprecated</em> <code>unsigned</code></dt>
        <dd>Indent width.</dd>
        </dl>
        </body></html>
        """
        result = parse_options(html)
        # The <em> tag is a Tag, so the loop breaks before finding <code>
        # No type is extracted, so the option is skipped
        assert result == {}
        captured = capsys.readouterr()
        assert "Could not parse top-level option" in captured.err

    def test_nested_ul_before_values_ul_is_skipped(self):
        """When nested options ul appears before values ul, it's skipped (line 179)."""
        html = """
        <html><body>
        <h2>Configurable Format Style Options</h2>
        <dl>
        <dt><strong>BraceWrapping</strong> <code>enum</code></dt>
        <dd>
        <p>Nested configuration flags:</p>
        <ul>
        <li><code>Boolean AfterCaseLabel</code></li>
        </ul>
        <ul>
        <li><code>Attach</code></li>
        <li><code>Allman</code></li>
        </ul>
        </dd>
        </dl>
        </body></html>
        """
        result = parse_options(html)
        assert "BraceWrapping" in result
        # Values ul is processed, nested ul is skipped
        assert result["BraceWrapping"]["possible_values"] == ["Attach", "Allman"]
        assert "BraceWrapping.AfterCaseLabel" in result
