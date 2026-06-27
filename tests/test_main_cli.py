"""Tests for main.py CLI subcommand routing."""

import sys
from pathlib import Path
from unittest.mock import patch

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


class TestSubcommandRouting:
    """Test that subcommands are correctly routed."""

    @patch("main.cmd_optimize")
    @patch("main.argparse.ArgumentParser.parse_args")
    def test_optimize_subcommand_routes(self, mock_parse, mock_cmd):
        import argparse

        mock_parse.return_value = argparse.Namespace(
            command="optimize", repo_path="/tmp"
        )
        from main import main

        main()
        mock_cmd.assert_called_once()

    @patch("main.cmd_fetch_options")
    @patch("main.argparse.ArgumentParser.parse_args")
    def test_fetch_options_subcommand_routes(self, mock_parse, mock_cmd):
        import argparse

        mock_parse.return_value = argparse.Namespace(
            command="fetch-options", version=None
        )
        from main import main

        main()
        mock_cmd.assert_called_once()

    @patch("main.cmd_optimize")
    @patch("main.argparse.ArgumentParser.parse_args")
    def test_default_routes_to_optimize(self, mock_parse, mock_cmd):
        import argparse

        # Simulate user providing positional args but no subcommand
        mock_parse.side_effect = [
            argparse.Namespace(command=None),
            argparse.Namespace(command="optimize", repo_path="/tmp"),
        ]
        from main import main

        main()
        mock_cmd.assert_called_once()

    @patch("main.argparse.ArgumentParser.parse_args")
    @patch("main.argparse.ArgumentParser.print_help")
    def test_no_args_prints_help(self, mock_help, mock_parse):
        import argparse

        mock_parse.return_value = argparse.Namespace(command=None)
        with patch("sys.argv", ["main.py"]):
            from main import main

            try:
                main()
            except SystemExit:
                pass
            mock_help.assert_called_once()


class TestFetchOptionsSubcommand:
    """Test the fetch-options subcommand logic."""

    @patch("src.option_fetcher.resolve_version")
    @patch("src.option_fetcher.build_urls")
    @patch("src.option_fetcher.fetch_html_content")
    @patch("src.option_fetcher.parse_options")
    def test_fetch_options_success(
        self, mock_parse, mock_fetch, mock_urls, mock_resolve, capsys
    ):
        mock_resolve.return_value = "18.1.8"
        mock_urls.return_value = ["http://example.com"]
        mock_fetch.return_value = "<html></html>"
        mock_parse.return_value = {
            "IndentWidth": {"type": "unsigned", "possible_values": None}
        }

        import argparse

        from main import cmd_fetch_options

        args = argparse.Namespace(version=None)
        cmd_fetch_options(args)

        captured = capsys.readouterr()
        assert "IndentWidth" in captured.out

    @patch("src.option_fetcher.resolve_version")
    @patch("src.option_fetcher.build_urls")
    @patch("src.option_fetcher.fetch_html_content")
    def test_fetch_options_no_html(self, mock_fetch, mock_urls, mock_resolve):
        mock_resolve.return_value = "latest"
        mock_urls.return_value = ["http://example.com"]
        mock_fetch.return_value = None

        import argparse

        from main import cmd_fetch_options

        args = argparse.Namespace(version=None)
        try:
            cmd_fetch_options(args)
        except SystemExit as e:
            assert e.code == 1


class TestAnalyzeFlags:
    """Test --no-analyze and --dry-run flags in cmd_optimize."""

    @patch("main.analyze_conventions")
    def test_dry_run_prints_yaml_and_returns(self, mock_analyze, capsys):
        import argparse

        mock_analyze.return_value = {"IndentWidth": "4"}
        from main import cmd_optimize

        args = argparse.Namespace(
            repo_path="/tmp",
            debug=False,
            start_config_file=None,
            dry_run=True,
            no_analyze=False,
        )
        with patch("main.os.path.isdir", return_value=True):
            cmd_optimize(args)

        captured = capsys.readouterr()
        assert "IndentWidth" in captured.out

    @patch("main.analyze_conventions")
    def test_dry_run_no_conventions(self, mock_analyze, capsys):
        import argparse

        mock_analyze.return_value = {}
        from main import cmd_optimize

        args = argparse.Namespace(
            repo_path="/tmp",
            debug=False,
            start_config_file=None,
            dry_run=True,
            no_analyze=False,
        )
        with patch("main.os.path.isdir", return_value=True):
            cmd_optimize(args)

        captured = capsys.readouterr()
        assert "No conventions detected" in captured.err

    def test_no_analyze_skips_analysis(self, capsys):
        import argparse

        from main import cmd_optimize

        args = argparse.Namespace(
            repo_path="/tmp",
            debug=False,
            start_config_file=None,
            dry_run=False,
            no_analyze=True,
            option_values_json_file=None,
            forced_options_yaml_file=None,
            output_file=None,
            optimizer="genetic",
            islands=1,
            population_size=4,
            iterations=1,
            migration_interval=15,
            polish_passes=0,
            file_sample_percentage=100.0,
            jobs=1,
        )
        with patch("main.os.path.isdir", return_value=True):
            with patch("main.analyze_conventions") as mock_analyze:
                with patch("main.run_island_ga") as mock_ga:
                    from src.optimization_engine.types import Individual

                    mock_ga.return_value = Individual(config={}, fitness=0)
                    with patch("main.tempfile.mkdtemp", return_value="/tmp/test"):
                        with patch("main.shutil.copytree"):
                            with patch("main.run_command"):
                                with patch(
                                    "main.generate_clang_format_config",
                                    return_value="",
                                ):
                                    with patch("main.shutil.rmtree"):
                                        cmd_optimize(args)
                                        mock_analyze.assert_not_called()

        captured = capsys.readouterr()
        assert "skipped" in captured.err

    @patch("main.build_search_space")
    @patch("main.load_forced_options")
    @patch("main.load_json_option_values")
    @patch("main.analyze_conventions")
    def test_analysis_results_passed_to_build_search_space(
        self, mock_analyze, mock_load_json, mock_load_forced, mock_bss
    ):
        import argparse

        mock_analyze.return_value = {"IndentWidth": "4"}
        mock_load_json.return_value = {}
        mock_load_forced.return_value = {}
        from main import cmd_optimize

        args = argparse.Namespace(
            repo_path="/tmp",
            debug=False,
            start_config_file=None,
            dry_run=False,
            no_analyze=False,
            option_values_json_file=None,
            forced_options_yaml_file=None,
            output_file=None,
            optimizer="genetic",
            islands=1,
            population_size=4,
            iterations=1,
            migration_interval=15,
            polish_passes=0,
            file_sample_percentage=100.0,
            jobs=1,
        )
        with patch("main.os.path.isdir", return_value=True):
            with patch("main.run_island_ga") as mock_ga:
                from src.optimization_engine.types import Individual

                mock_ga.return_value = Individual(config={}, fitness=0)
                with patch("main.tempfile.mkdtemp", return_value="/tmp/test"):
                    with patch("main.shutil.copytree"):
                        with patch("main.run_command"):
                            with patch(
                                "main.generate_clang_format_config",
                                return_value="",
                            ):
                                with patch("main.shutil.rmtree"):
                                    cmd_optimize(args)

        # Verify analyze was called with the repo path
        mock_analyze.assert_called_once_with("/tmp")
        # Verify build_search_space was called with analysis_results
        mock_bss.assert_called_once()
        call_kwargs = mock_bss.call_args
        assert call_kwargs[0][2] == {"IndentWidth": "4"}
