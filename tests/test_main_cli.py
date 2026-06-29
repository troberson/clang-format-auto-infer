"""Tests for main.py CLI subcommand routing."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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
        # Non-iterative optimizer should not pass polish_undetect
        assert call_kwargs[1].get("polish_undetect") is False

    @patch("main.build_search_space")
    @patch("main.load_forced_options")
    @patch("main.load_json_option_values")
    @patch("main.analyze_conventions")
    def test_iterative_optimizer_passes_polish_undetect(
        self, mock_analyze, mock_load_json, mock_load_forced, mock_bss
    ):
        import argparse

        from src.analyze_conventions import DetectedOption

        mock_analyze.return_value = {
            "IndentWidth": DetectedOption(4, "detected", "polish")
        }
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
            optimizer="iterative",
            islands=1,
            population_size=4,
            iterations=1,
            migration_interval=15,
            polish_passes=0,
            file_sample_percentage=100.0,
            jobs=1,
            ng_optimizer="TwoPointsDE",
            convergence_threshold=20,
        )
        with patch("main.os.path.isdir", return_value=True):
            with patch("main.run_iterative_optimization") as mock_iter:
                from unittest.mock import MagicMock

                mock_result = MagicMock()
                mock_result.best_config = {}
                mock_iter.return_value = mock_result
                with patch("main.tempfile.mkdtemp", return_value="/tmp/test"):
                    with patch("main.shutil.copytree"):
                        with patch("main.run_command"):
                            with patch(
                                "main.generate_clang_format_config",
                                return_value="",
                            ):
                                with patch("main.shutil.rmtree"):
                                    cmd_optimize(args)

        # Verify polish_undetect=True for iterative optimizer
        mock_bss.assert_called_once()
        call_kwargs = mock_bss.call_args
        assert call_kwargs[1].get("polish_undetect") is True

    @patch("main.build_search_space")
    @patch("main.load_forced_options")
    @patch("main.load_json_option_values")
    @patch("main.analyze_conventions")
    def test_analyzer_option_not_in_options_info_merged_into_initial_config(
        self, mock_analyze, mock_load_json, mock_load_forced, _mock_bss
    ):
        """When the analyzer detects an option not present in dump-config
        (e.g., QualifierOrder when QualifierAlignment is not Custom), it must
        be merged into both options_info and initial_config so the optimizer
        can use it."""
        import argparse

        from src.analyze_conventions import DetectedOption

        # Analyzer returns QualifierOrder, which is NOT in our minimal options_info
        mock_analyze.return_value = {
            "QualifierOrder": DetectedOption(
                ["inline", "static", "type", "const"], "forced", "resolve"
            )
        }
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

        # Verify run_island_ga was called with initial_config containing QualifierOrder
        mock_ga.assert_called_once()
        call_kwargs = mock_ga.call_args[1]
        initial_config = call_kwargs["initial_config"]
        assert "QualifierOrder" in initial_config
        assert initial_config["QualifierOrder"] == [
            "inline",
            "static",
            "type",
            "const",
        ]


class TestTempDirFeature:
    """Test --temp-dir CLI argument and tmpfs auto-detection."""

    @patch("main.run_command")
    def test_get_repo_disk_usage_returns_bytes(self, mock_run):
        from main import get_repo_disk_usage

        mock_run.return_value.stdout = "1048576\t/path/to/repo\n"
        mock_run.return_value.stderr = ""
        size = get_repo_disk_usage("/path/to/repo")
        assert size == 1048576
        mock_run.assert_called_once()

    @patch("main.run_command")
    def test_get_repo_disk_usage_returns_zero_on_exception(self, mock_run):
        from main import get_repo_disk_usage

        mock_run.side_effect = ValueError("bad output")
        size = get_repo_disk_usage("/nonexistent")
        assert size == 0

    def test_find_tmpfs_mounts_returns_matching_mounts(self):
        from main import find_tmpfs_mounts

        lines = [
            "none /dev/shm tmpfs rw,seclabel,size=8192000k 0 0\n",
            "/dev/sda1 /tmp ext4 rw,relatime 0 0\n",
            "none /run/user/1000 tmpfs rw,nosuid 0 0\n",
        ]
        mock_file = MagicMock()
        mock_file.__iter__ = MagicMock(return_value=iter(lines))
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        with patch("builtins.open", MagicMock(return_value=mock_file)):
            mounts = find_tmpfs_mounts()
        assert mounts == ["/dev/shm", "/run/user/1000"]

    def test_find_tmpfs_mounts_empty_file(self):
        from main import find_tmpfs_mounts

        mock_file = MagicMock()
        mock_file.__iter__ = MagicMock(return_value=iter([]))
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        with patch("builtins.open", MagicMock(return_value=mock_file)):
            mounts = find_tmpfs_mounts()
        assert mounts == []

    def test_find_tmpfs_mounts_os_error_returns_empty(self):
        from main import find_tmpfs_mounts

        with patch("builtins.open", side_effect=OSError("no /proc/mounts")):
            mounts = find_tmpfs_mounts()
        assert mounts == []

    @patch("main.find_tmpfs_mounts")
    @patch("main.shutil.disk_usage")
    @patch("main.get_repo_disk_usage")
    def test_get_best_temp_location_selects_tmpfs_when_available(
        self, mock_usage, mock_disk, mock_mounts
    ):
        from main import get_best_temp_location

        mock_usage.return_value = 10_000_000  # 10MB repo
        mock_disk.return_value.free = 1_000_000_000  # 1GB free
        mock_mounts.return_value = ["/dev/shm"]

        path, detected = get_best_temp_location("/repo", jobs=2, debug=True)
        assert path == "/dev/shm"
        assert detected is True

    @patch("main.find_tmpfs_mounts")
    @patch("main.shutil.disk_usage")
    @patch("main.get_repo_disk_usage")
    def test_get_best_temp_location_fallback_when_tmpfs_insufficient(
        self, mock_usage, mock_disk, mock_mounts
    ):
        from main import get_best_temp_location

        mock_usage.return_value = 500_000_000  # 500MB repo
        mock_disk.return_value.free = 1_000_000  # only 1MB free, need 500MB*2*2=2GB
        mock_mounts.return_value = ["/dev/shm"]

        path, detected = get_best_temp_location("/repo", jobs=2, debug=False)
        assert path == "/tmp"  # default tempdir
        assert detected is False

    @patch("main.find_tmpfs_mounts")
    @patch("main.shutil.disk_usage")
    @patch("main.get_repo_disk_usage")
    def test_get_best_temp_location_fallback_when_disk_usage_raises(
        self, mock_usage, mock_disk, mock_mounts
    ):
        """OSError from disk_usage falls back to default temp dir."""
        from main import get_best_temp_location

        mock_usage.return_value = 10_000_000
        mock_mounts.return_value = ["/dev/shm"]
        mock_disk.side_effect = OSError("permission denied")

        _, detected = get_best_temp_location("/repo", jobs=1, debug=True)
        assert detected is False

    @patch("main.find_tmpfs_mounts")
    @patch("main.get_repo_disk_usage")
    def test_get_best_temp_location_fallback_when_no_tmpfs(
        self, mock_usage, mock_mounts
    ):
        from main import get_best_temp_location

        mock_usage.return_value = 10_000_000
        mock_mounts.return_value = []  # no tmpfs mounts found

        path, detected = get_best_temp_location("/repo", jobs=1, debug=True)
        assert path == "/tmp"  # default tempdir
        assert detected is False

    @patch("main.find_tmpfs_mounts")
    @patch("main.shutil.disk_usage")
    @patch("main.os.access")
    @patch("main.get_repo_disk_usage")
    def test_get_best_temp_location_skips_non_writable_tmpfs(
        self, mock_usage, mock_access, mock_disk, mock_mounts
    ):
        """Non-writable tmpfs mounts (e.g. /run) are skipped."""
        from main import get_best_temp_location

        mock_usage.return_value = 10_000_000
        mock_mounts.return_value = ["/run", "/dev/shm"]
        mock_access.side_effect = [False, True]  # /run not writable, /dev/shm is
        mock_disk.return_value.free = 1_000_000_000

        path, detected = get_best_temp_location("/repo", jobs=1, debug=False)
        assert path == "/dev/shm"
        assert detected is True

    def test_cmd_optimize_uses_explicit_temp_dir(self):
        """When --temp-dir is provided, it is used as the base for temp dirs."""
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
            temp_dir="/custom/tmp",
        )
        with patch("main.os.path.isdir", return_value=True):
            with patch("main.run_island_ga") as mock_ga:
                from src.optimization_engine.types import Individual

                mock_ga.return_value = Individual(config={}, fitness=0)
                with patch("main.tempfile.mkdtemp") as mock_mkdir:
                    with patch("main.shutil.copytree"):
                        with patch("main.run_command"):
                            with patch(
                                "main.generate_clang_format_config",
                                return_value="",
                            ):
                                with patch("main.shutil.rmtree"):
                                    cmd_optimize(args)
                                    # Verify mkdtemp was called with dir=/custom/tmp
                                    call_kwargs = mock_mkdir.call_args[1]
                                    assert call_kwargs["dir"].startswith("/custom/tmp")

    @patch("main.get_best_temp_location")
    def test_cmd_optimize_auto_detects_when_no_temp_dir(self, mock_detect):
        """When --temp-dir is omitted, auto-detection is triggered."""
        import argparse

        from main import cmd_optimize

        mock_detect.return_value = ("/dev/shm", True)

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
            temp_dir=None,
        )
        with patch("main.os.path.isdir", return_value=True):
            with patch("main.run_island_ga") as mock_ga:
                from src.optimization_engine.types import Individual

                mock_ga.return_value = Individual(config={}, fitness=0)
                with patch("main.tempfile.mkdtemp") as mock_mkdir:
                    with patch("main.shutil.copytree"):
                        with patch("main.run_command"):
                            with patch(
                                "main.generate_clang_format_config",
                                return_value="",
                            ):
                                with patch("main.shutil.rmtree"):
                                    cmd_optimize(args)
                                    mock_detect.assert_called_once()
                                    # Verify mkdtemp was called with dir=/dev/shm
                                    call_kwargs = mock_mkdir.call_args[1]
                                    assert call_kwargs["dir"].startswith("/dev/shm")


class TestCmdOptimizeErrorPaths:
    """Test error handling and alternate branches in cmd_optimize."""

    def test_invalid_repo_path_exits(self):
        import argparse

        from main import cmd_optimize

        args = argparse.Namespace(repo_path="/nonexistent", debug=False)
        with patch("main.os.path.isdir", return_value=False):
            with pytest.raises(SystemExit) as exc:
                cmd_optimize(args)
            assert exc.value.code == 1

    def test_invalid_abs_repo_path_exits(self):
        import argparse

        from main import cmd_optimize

        args = argparse.Namespace(repo_path="/tmp", debug=False)
        with patch("main.os.path.isdir", side_effect=[True, False]):
            with patch("main.os.path.abspath", return_value="/absolute/path"):
                with pytest.raises(SystemExit) as exc:
                    cmd_optimize(args)
                assert exc.value.code == 1

    def test_start_config_file_not_found_exits(self):
        import argparse

        from main import cmd_optimize

        args = argparse.Namespace(
            repo_path="/tmp",
            debug=False,
            start_config_file="/nonexistent.yaml",
        )
        with patch("main.os.path.isdir", return_value=True):
            with patch("main.os.path.exists", return_value=False):
                with pytest.raises(SystemExit) as exc:
                    cmd_optimize(args)
                assert exc.value.code == 1

    def test_start_config_file_parse_failure_exits(self):
        import argparse

        from main import cmd_optimize

        args = argparse.Namespace(
            repo_path="/tmp",
            debug=False,
            start_config_file="/tmp/config.yaml",
        )
        with patch("main.os.path.isdir", return_value=True):
            with patch("main.os.path.exists", return_value=True):
                with patch("builtins.open", mock_open_data("invalid: [yaml")):
                    with patch("main.parse_clang_format_options", return_value=None):
                        with pytest.raises(SystemExit) as exc:
                            cmd_optimize(args)
                        assert exc.value.code == 1

    def test_start_config_file_os_error(self):
        import argparse

        from main import cmd_optimize

        args = argparse.Namespace(
            repo_path="/tmp",
            debug=False,
            start_config_file="/tmp/config.yaml",
        )
        with patch("main.os.path.isdir", return_value=True):
            with patch("main.os.path.exists", return_value=True):
                with patch("builtins.open", side_effect=OSError("permission denied")):
                    with pytest.raises(SystemExit) as exc:
                        cmd_optimize(args)
                    assert exc.value.code == 1

    def test_start_config_file_yaml_error(self):
        import argparse

        from main import cmd_optimize
        import yaml

        args = argparse.Namespace(
            repo_path="/tmp",
            debug=False,
            start_config_file="/tmp/config.yaml",
        )
        with patch("main.os.path.isdir", return_value=True):
            with patch("main.os.path.exists", return_value=True):
                with patch("builtins.open", mock_open_data("valid: yaml")):
                    with patch(
                        "main.parse_clang_format_options",
                        side_effect=yaml.YAMLError("bad"),
                    ):
                        with pytest.raises(SystemExit) as exc:
                            cmd_optimize(args)
                        assert exc.value.code == 1

    def test_start_config_file_success(self, capsys):
        import argparse

        from main import cmd_optimize

        args = argparse.Namespace(
            repo_path="/tmp",
            debug=False,
            start_config_file="/tmp/config.yaml",
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
            with patch("main.os.path.exists", return_value=True):
                with patch("builtins.open", mock_open_data("IndentWidth: 4")):
                    with patch(
                        "main.parse_clang_format_options",
                        return_value=mock_options_info(),
                    ):
                        with patch("main.build_search_space", return_value=[]):
                            with patch("main.run_island_ga") as mock_ga:
                                from src.optimization_engine.types import Individual

                                mock_ga.return_value = Individual(config={}, fitness=0)
                                with patch("main.tempfile.mkdtemp"):
                                    with patch("main.shutil.copytree"):
                                        with patch("main.run_command"):
                                            with patch(
                                                "main.generate_clang_format_config",
                                                return_value="",
                                            ):
                                                with patch("main.shutil.rmtree"):
                                                    cmd_optimize(args)

        captured = capsys.readouterr()
        assert "Successfully loaded and parsed initial configuration" in captured.err

    def test_option_values_json_file_printed(self, capsys):
        import argparse

        from main import cmd_optimize

        args = argparse.Namespace(
            repo_path="/tmp",
            debug=False,
            start_config_file=None,
            dry_run=False,
            no_analyze=True,
            option_values_json_file="/tmp/values.json",
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
            with patch("main.get_clang_format_options", return_value="output"):
                with patch(
                    "main.parse_clang_format_options", return_value=mock_options_info()
                ):
                    with patch("main.load_json_option_values", return_value={}):
                        with patch(
                            "main.find_options_without_json_values"
                        ):  # no missing options
                            with patch("main.build_search_space", return_value=[]):
                                with patch("main.run_island_ga") as mock_ga:
                                    from src.optimization_engine.types import Individual

                                    mock_ga.return_value = Individual(
                                        config={}, fitness=0
                                    )
                                    with patch("main.tempfile.mkdtemp"):
                                        with patch("main.shutil.copytree"):
                                            with patch("main.run_command"):
                                                with patch(
                                                    "main.generate_clang_format_config",
                                                    return_value="",
                                                ):
                                                    with patch("main.shutil.rmtree"):
                                                        cmd_optimize(args)

        captured = capsys.readouterr()
        assert "provided JSON file" in captured.err

    def test_no_json_file_printed(self, capsys):
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
            with patch("main.get_clang_format_options", return_value="output"):
                with patch(
                    "main.parse_clang_format_options", return_value=mock_options_info()
                ):
                    with patch(
                        "main.find_options_without_json_values"
                    ):  # no missing options
                        with patch("main.build_search_space", return_value=[]):
                            with patch("main.run_island_ga") as mock_ga:
                                from src.optimization_engine.types import Individual

                                mock_ga.return_value = Individual(config={}, fitness=0)
                                with patch("main.tempfile.mkdtemp"):
                                    with patch("main.shutil.copytree"):
                                        with patch("main.run_command"):
                                            with patch(
                                                "main.generate_clang_format_config",
                                                return_value="",
                                            ):
                                                with patch("main.shutil.rmtree"):
                                                    cmd_optimize(args)

        captured = capsys.readouterr()
        assert "No JSON file with option values was provided" in captured.err
        import argparse

        from main import cmd_optimize

        args = argparse.Namespace(
            repo_path="/tmp",
            debug=False,
            start_config_file=None,
        )
        with patch("main.os.path.isdir", return_value=True):
            with patch("main.get_clang_format_options", return_value=None):
                with pytest.raises(SystemExit) as exc:
                    cmd_optimize(args)
                assert exc.value.code == 1

    def test_parse_clang_format_options_returns_none_exits(self):
        import argparse

        from main import cmd_optimize

        args = argparse.Namespace(
            repo_path="/tmp",
            debug=False,
            start_config_file=None,
        )
        with patch("main.os.path.isdir", return_value=True):
            with patch("main.get_clang_format_options", return_value="some output"):
                with patch("main.parse_clang_format_options", return_value=None):
                    with pytest.raises(SystemExit) as exc:
                        cmd_optimize(args)
                    assert exc.value.code == 1

    def test_jobs_less_than_one_clamped(self, capsys):
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
            jobs=0,
        )
        with patch("main.os.path.isdir", return_value=True):
            with patch("main.get_clang_format_options", return_value="output"):
                with patch(
                    "main.parse_clang_format_options", return_value=mock_options_info()
                ):
                    with patch("main.build_search_space", return_value=[]):
                        with patch("main.run_island_ga") as mock_ga:
                            from src.optimization_engine.types import Individual

                            mock_ga.return_value = Individual(config={}, fitness=0)
                            with patch("main.tempfile.mkdtemp"):
                                with patch("main.shutil.copytree"):
                                    with patch("main.run_command"):
                                        with patch(
                                            "main.generate_clang_format_config",
                                            return_value="",
                                        ):
                                            with patch("main.shutil.rmtree"):
                                                cmd_optimize(args)

        captured = capsys.readouterr()
        assert "at least 1" in captured.err

    def test_git_init_failure_exits(self):
        import argparse
        import subprocess

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
            with patch("main.get_clang_format_options", return_value="output"):
                with patch(
                    "main.parse_clang_format_options", return_value=mock_options_info()
                ):
                    with patch("main.build_search_space", return_value=[]):
                        with patch("main.run_island_ga") as mock_ga:
                            from src.optimization_engine.types import Individual

                            mock_ga.return_value = Individual(config={}, fitness=0)
                            with patch("main.tempfile.mkdtemp"):
                                with patch("main.shutil.copytree"):
                                    with patch(
                                        "main.run_command",
                                        side_effect=subprocess.CalledProcessError(
                                            1, "git"
                                        ),
                                    ):
                                        with pytest.raises(SystemExit) as exc:
                                            with patch("main.shutil.rmtree"):
                                                cmd_optimize(args)
                                        assert exc.value.code == 1

    def test_polish_passes_runs(self):
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
            polish_passes=2,
            file_sample_percentage=100.0,
            jobs=1,
        )
        with patch("main.os.path.isdir", return_value=True):
            with patch("main.get_clang_format_options", return_value="output"):
                with patch(
                    "main.parse_clang_format_options", return_value=mock_options_info()
                ):
                    with patch("main.build_search_space", return_value=[]):
                        with patch("main.run_island_ga") as mock_ga:
                            from src.optimization_engine.types import Individual

                            mock_ga.return_value = Individual(config={}, fitness=0)
                            with patch("main.polish_coordinate_descent") as mock_polish:
                                mock_polish.return_value = Individual(
                                    config={}, fitness=0
                                )
                                with patch("main.tempfile.mkdtemp"):
                                    with patch("main.shutil.copytree"):
                                        with patch("main.run_command"):
                                            with patch(
                                                "main.generate_clang_format_config",
                                                return_value="",
                                            ):
                                                with patch("main.shutil.rmtree"):
                                                    cmd_optimize(args)
                                                    mock_polish.assert_called_once()

    def test_nevergrad_optimizer_runs(self):
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
            optimizer="nevergrad",
            ng_optimizer="DiscreteOnePlusOne",
            convergence_threshold=20,
            file_sample_percentage=100.0,
            jobs=1,
        )
        with patch("main.os.path.isdir", return_value=True):
            with patch("main.get_clang_format_options", return_value="output"):
                with patch(
                    "main.parse_clang_format_options", return_value=mock_options_info()
                ):
                    with patch("main.build_search_space", return_value=[]):
                        with patch("main.run_nevergrad_optimization") as mock_ng:
                            from unittest.mock import MagicMock

                            mock_result = MagicMock()
                            mock_result.best_config = {}
                            mock_ng.return_value = mock_result
                            with patch("main.tempfile.mkdtemp"):
                                with patch("main.shutil.copytree"):
                                    with patch("main.run_command"):
                                        with patch(
                                            "main.generate_clang_format_config",
                                            return_value="",
                                        ):
                                            with patch("main.shutil.rmtree"):
                                                cmd_optimize(args)
                                                mock_ng.assert_called_once()

    def test_iterative_optimizer_runs(self):
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
            optimizer="iterative",
            convergence_threshold=20,
            islands=1,
            population_size=4,
            file_sample_percentage=100.0,
            jobs=1,
        )
        with patch("main.os.path.isdir", return_value=True):
            with patch("main.get_clang_format_options", return_value="output"):
                with patch(
                    "main.parse_clang_format_options", return_value=mock_options_info()
                ):
                    with patch("main.build_search_space", return_value=[]):
                        with patch("main.run_iterative_optimization") as mock_iter:
                            from unittest.mock import MagicMock

                            mock_result = MagicMock()
                            mock_result.best_config = {}
                            mock_iter.return_value = mock_result
                            with patch("main.tempfile.mkdtemp"):
                                with patch("main.shutil.copytree"):
                                    with patch("main.run_command"):
                                        with patch(
                                            "main.generate_clang_format_config",
                                            return_value="",
                                        ):
                                            with patch("main.shutil.rmtree"):
                                                cmd_optimize(args)
                                                mock_iter.assert_called_once()

    def test_unknown_optimizer_exits(self):
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
            optimizer="unknown",
            file_sample_percentage=100.0,
            jobs=1,
        )
        with patch("main.os.path.isdir", return_value=True):
            with patch("main.get_clang_format_options", return_value="output"):
                with patch(
                    "main.parse_clang_format_options", return_value=mock_options_info()
                ):
                    with patch("main.build_search_space", return_value=[]):
                        with patch("main.tempfile.mkdtemp"):
                            with patch("main.shutil.copytree"):
                                with patch("main.run_command"):
                                    with pytest.raises(SystemExit) as exc:
                                        with patch("main.shutil.rmtree"):
                                            cmd_optimize(args)
                                    assert exc.value.code == 1

    def test_output_file_written(self):
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
            output_file="/tmp/output.yaml",
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
            with patch("main.get_clang_format_options", return_value="output"):
                with patch(
                    "main.parse_clang_format_options", return_value=mock_options_info()
                ):
                    with patch("main.build_search_space", return_value=[]):
                        with patch("main.run_island_ga") as mock_ga:
                            from src.optimization_engine.types import Individual

                            mock_ga.return_value = Individual(config={}, fitness=0)
                            with patch("main.tempfile.mkdtemp"):
                                with patch("main.shutil.copytree"):
                                    with patch("main.run_command"):
                                        with patch(
                                            "main.generate_clang_format_config",
                                            return_value="test: config",
                                        ):
                                            with patch(
                                                "builtins.open", mock_open_data()
                                            ):
                                                with patch("main.shutil.rmtree"):
                                                    cmd_optimize(args)

    def test_output_file_os_error(self):
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
            output_file="/tmp/output.yaml",
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
            with patch("main.get_clang_format_options", return_value="output"):
                with patch(
                    "main.parse_clang_format_options", return_value=mock_options_info()
                ):
                    with patch("main.build_search_space", return_value=[]):
                        with patch("main.run_island_ga") as mock_ga:
                            from src.optimization_engine.types import Individual

                            mock_ga.return_value = Individual(config={}, fitness=0)
                            with patch("main.tempfile.mkdtemp"):
                                with patch("main.shutil.copytree"):
                                    with patch("main.run_command"):
                                        with patch(
                                            "main.generate_clang_format_config",
                                            return_value="test: config",
                                        ):
                                            with patch(
                                                "builtins.open",
                                                side_effect=OSError("disk full"),
                                            ):
                                                with pytest.raises(SystemExit) as exc:
                                                    with patch("main.shutil.rmtree"):
                                                        cmd_optimize(args)
                                                assert exc.value.code == 1

    def test_cleanup_os_error(self, capsys):
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
            with patch("main.get_clang_format_options", return_value="output"):
                with patch(
                    "main.parse_clang_format_options", return_value=mock_options_info()
                ):
                    with patch("main.build_search_space", return_value=[]):
                        with patch("main.run_island_ga") as mock_ga:
                            from src.optimization_engine.types import Individual

                            mock_ga.return_value = Individual(config={}, fitness=0)
                            with patch("main.tempfile.mkdtemp"):
                                with patch("main.shutil.copytree"):
                                    with patch("main.run_command"):
                                        with patch(
                                            "main.generate_clang_format_config",
                                            return_value="",
                                        ):
                                            with patch(
                                                "main.os.path.exists", return_value=True
                                            ):
                                                with patch(
                                                    "main.shutil.rmtree",
                                                    side_effect=OSError(
                                                        "permission denied"
                                                    ),
                                                ):
                                                    cmd_optimize(args)

        captured = capsys.readouterr()
        assert "Error removing temporary directory" in captured.err


class TestPipelineWiring:
    """Test that analyze_with_metadata and measure_impact are wired into main.py."""

    def test_dry_run_prints_detected_option_metadata(self, capsys):
        """Dry-run with DetectedOption objects prints confidence and tier."""
        import argparse

        from main import cmd_optimize
        from src.analyze_conventions import DetectedOption

        args = argparse.Namespace(
            repo_path="/tmp",
            debug=False,
            start_config_file=None,
            dry_run=True,
            no_analyze=False,
        )
        with patch("main.os.path.isdir", return_value=True):
            with patch(
                "main.analyze_conventions",
                return_value={
                    "IndentWidth": DetectedOption(4, "detected", "structure")
                },
            ):
                cmd_optimize(args)

        captured = capsys.readouterr()
        assert "IndentWidth: 4" in captured.out
        assert "detected" in captured.out
        assert "structure" in captured.out


class TestFetchOptionsEmptyResult:
    """Test fetch-options subcommand when no options are found."""

    @patch("src.option_fetcher.resolve_version")
    @patch("src.option_fetcher.build_urls")
    @patch("src.option_fetcher.fetch_html_content")
    @patch("src.option_fetcher.parse_options")
    def test_fetch_options_empty_list(
        self, mock_parse, mock_fetch, mock_urls, mock_resolve, capsys
    ):
        mock_resolve.return_value = "18.1.8"
        mock_urls.return_value = ["http://example.com"]
        mock_fetch.return_value = "<html></html>"
        mock_parse.return_value = []  # Empty list

        import argparse

        from main import cmd_fetch_options

        args = argparse.Namespace(version=None)
        with patch("main.sys.exit"):
            cmd_fetch_options(args)

        captured = capsys.readouterr()
        assert "[]" in captured.out


def mock_open_data(data=None):
    """Helper to create a mock for builtins.open that returns specific data."""
    from unittest.mock import MagicMock

    mock_file = MagicMock()
    if data is not None:
        mock_file.read.return_value = data
    mock_file.__enter__ = MagicMock(return_value=mock_file)
    mock_file.__exit__ = MagicMock(return_value=False)
    return MagicMock(return_value=mock_file)


def mock_options_info():
    """Return a minimal OptionInfo dict that parse_clang_format_options would produce."""
    return {"IndentWidth": {"type": "int", "value": 4}}


class TestWarnUnusedIterativeFlags:
    """Test _warn_unused_iterative_flags warns and overrides GA/nevergrad flags."""

    def test_warns_about_unused_flags(self, capsys):
        from main import _warn_unused_iterative_flags  # pyright: ignore[reportPrivateUsage]

        import argparse

        args = argparse.Namespace(
            optimizer="iterative",
            iterations=200,
            population_size=8,
            islands=3,
            polish_passes=5,
            migration_interval=10,
            ng_optimizer="CMA",
        )
        _warn_unused_iterative_flags(args)
        captured = capsys.readouterr()
        assert "ignores" in captured.err
        assert "--iterations" in captured.err
        assert "--ng-optimizer" in captured.err

    def test_overrides_flags_with_defaults(self):
        from main import _warn_unused_iterative_flags  # pyright: ignore[reportPrivateUsage]

        import argparse

        args = argparse.Namespace(
            optimizer="iterative",
            iterations=200,
            population_size=8,
            islands=3,
            polish_passes=5,
            migration_interval=10,
            ng_optimizer="CMA",
        )
        _warn_unused_iterative_flags(args)
        assert args.iterations == 100
        assert args.population_size == 4
        assert args.islands == 1
        assert args.polish_passes == 0
        assert args.migration_interval == 15
        assert args.ng_optimizer == "TwoPointsDE"

    def test_no_warning_when_no_unused_flags(self, capsys):
        from main import _warn_unused_iterative_flags  # pyright: ignore[reportPrivateUsage]

        import argparse

        args = argparse.Namespace(optimizer="iterative")
        _warn_unused_iterative_flags(args)
        captured = capsys.readouterr()
        assert "ignores" not in captured.err

    def test_main_path_calls_warn_for_iterative(self, capsys, monkeypatch):
        """Ensure main() invokes _warn_unused_iterative_flags when --optimizer iterative."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "main.py",
                "optimize",
                "--optimizer",
                "iterative",
                "--iterations",
                "200",
                "/tmp/fake",
            ],
        )
        from main import main

        with patch("main.cmd_optimize") as mock_optimize:
            mock_optimize.side_effect = SystemExit(0)
            try:
                main()
            except SystemExit:
                pass
        captured = capsys.readouterr()
        assert "ignores" in captured.err
        assert "iterative" in captured.err
        assert "--iterations" in captured.err
