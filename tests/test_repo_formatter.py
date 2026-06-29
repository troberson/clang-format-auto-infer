"""Tests for repo_formatter — mock run_command to avoid real clang-format/git."""

import os
import subprocess
import threading
from unittest.mock import patch, MagicMock
import pytest

from src.repo_formatter import (
    run_clang_format_and_count_changes,
    ClangFormatWorkerError,
    _is_build_dir,  # pyright: ignore[reportPrivateUsage]
)


def _make_mock_result(stdout="", stderr="", returncode=0):
    """Create a mock CompletedProcess."""
    mock = MagicMock(spec=subprocess.CompletedProcess)
    mock.stdout = stdout
    mock.stderr = stderr
    mock.returncode = returncode
    mock.cmd = ["mock-cmd"]
    return mock


class TestRunClangFormatAndCountChanges:
    @patch("src.repo_formatter.run_command")
    def test_zero_changes(self, mock_run, tmp_path):
        repo = str(tmp_path)
        os.makedirs(repo, exist_ok=True)
        # git ls-files returns one file, git diff shows no changes
        mock_run.side_effect = [
            _make_mock_result(stdout="main.cpp\n"),  # ls-files
            None,  # clang-format (no output)
            _make_mock_result(stdout=""),  # git diff (no changes)
            None,  # git restore
        ]
        result = run_clang_format_and_count_changes(
            "BasedOnStyle: LLVM\n", repo, 1, False, 100.0, 42
        )
        assert result == 0

    @patch("src.repo_formatter.run_command")
    def test_counts_insertions_and_deletions(self, mock_run, tmp_path):
        repo = str(tmp_path)
        os.makedirs(repo, exist_ok=True)
        mock_run.side_effect = [
            _make_mock_result(stdout="a.cpp\nb.cpp\n"),
            None,
            _make_mock_result(
                stdout=" 2 files changed, 5 insertions(+), 3 deletions(-)\n"
            ),
            None,
        ]
        result = run_clang_format_and_count_changes(
            "BasedOnStyle: LLVM\n", repo, 1, False, 100.0, 42
        )
        assert result == 8

    @patch("src.repo_formatter.run_command")
    def test_invalid_config_returns_inf(self, mock_run, tmp_path):
        repo = str(tmp_path)
        os.makedirs(repo, exist_ok=True)
        mock_run.side_effect = [
            _make_mock_result(stdout="a.cpp\n"),
            subprocess.CalledProcessError(
                1, ["clang-format"], output="", stderr="cannot be used with"
            ),
            None,  # git restore
        ]
        result = run_clang_format_and_count_changes(
            "BadOption: true\n", repo, 1, False, 100.0, 42
        )
        assert result == float("inf")

    @patch("src.repo_formatter.run_command")
    def test_crash_returns_inf(self, mock_run, tmp_path):
        repo = str(tmp_path)
        os.makedirs(repo, exist_ok=True)
        mock_run.side_effect = [
            _make_mock_result(stdout="a.cpp\n"),
            subprocess.CalledProcessError(
                1, ["clang-format"], output="", stderr="PLEASE submit a bug report"
            ),
            None,
        ]
        result = run_clang_format_and_count_changes(
            "BasedOnStyle: LLVM\n", repo, 1, False, 100.0, 42
        )
        assert result == float("inf")

    @patch("src.repo_formatter.run_command")
    def test_timeout_returns_inf(self, mock_run, tmp_path):
        repo = str(tmp_path)
        os.makedirs(repo, exist_ok=True)
        mock_run.side_effect = [
            _make_mock_result(stdout="a.cpp\n"),
            subprocess.TimeoutExpired(["clang-format"], 120),
            None,
        ]
        result = run_clang_format_and_count_changes(
            "BasedOnStyle: LLVM\n", repo, 1, False, 100.0, 42
        )
        assert result == float("inf")

    @patch("src.repo_formatter.run_command")
    def test_git_error_returns_minus_one(self, mock_run, tmp_path):
        repo = str(tmp_path)
        os.makedirs(repo, exist_ok=True)
        mock_run.side_effect = [
            _make_mock_result(stdout="a.cpp\n"),
            None,
            subprocess.CalledProcessError(
                1, ["git", "diff"], output="", stderr="error"
            ),
            None,
        ]
        result = run_clang_format_and_count_changes(
            "BasedOnStyle: LLVM\n", repo, 1, False, 100.0, 42
        )
        assert result == -1

    @patch("src.repo_formatter.run_command")
    def test_no_files_returns_zero(self, mock_run, tmp_path):
        repo = str(tmp_path)
        os.makedirs(repo, exist_ok=True)
        mock_run.side_effect = [
            _make_mock_result(stdout=""),  # no tracked C/C++ files
            None,
        ]
        result = run_clang_format_and_count_changes(
            "BasedOnStyle: LLVM\n", repo, 1, False, 100.0, 42
        )
        assert result == 0

    @patch("src.repo_formatter.run_command")
    def test_file_listing_error_returns_zero(self, mock_run, tmp_path):
        """When git ls-files fails, _get_cached_file_list caches [] and the function returns 0."""
        repo = str(tmp_path)
        os.makedirs(repo, exist_ok=True)
        mock_run.side_effect = [
            subprocess.CalledProcessError(1, ["git", "ls-files"]),
            None,
        ]
        result = run_clang_format_and_count_changes(
            "BasedOnStyle: LLVM\n", repo, 1, False, 100.0, 42
        )
        assert result == 0

    @patch("src.repo_formatter.run_command")
    def test_file_sampling(self, mock_run, tmp_path):
        repo = str(tmp_path)
        os.makedirs(repo, exist_ok=True)
        files = "\n".join([f"file{i}.cpp" for i in range(10)]) + "\n"
        mock_run.side_effect = [
            _make_mock_result(stdout=files),
            None,
            _make_mock_result(stdout=""),
            None,
        ]
        result = run_clang_format_and_count_changes(
            "BasedOnStyle: LLVM\n", repo, 1, False, 50.0, 42
        )
        assert result == 0
        # Verify clang-format was called with only 5 files (50% of 10)
        clang_call = mock_run.call_args_list[1]
        cmd = clang_call[1].get("cmd", clang_call[0][0])
        # cmd should have clang-format + style flag + 5 files
        file_args = [a for a in cmd if a.endswith(".cpp")]
        assert len(file_args) == 5

    @patch("src.repo_formatter.run_command")
    def test_temp_config_cleaned_up(self, mock_run, tmp_path):
        repo = str(tmp_path)
        os.makedirs(repo, exist_ok=True)
        mock_run.side_effect = [
            _make_mock_result(stdout="a.cpp\n"),
            None,
            _make_mock_result(stdout=""),
            None,
        ]
        _ = run_clang_format_and_count_changes(
            "BasedOnStyle: LLVM\n", repo, 1, False, 100.0, 42
        )
        thread_id = threading.get_ident()
        temp_name = f".clang-format.tmp.{thread_id}"
        assert not os.path.exists(os.path.join(repo, temp_name))

    @patch("src.repo_formatter.run_command")
    def test_git_restore_called(self, mock_run, tmp_path):
        repo = str(tmp_path)
        os.makedirs(repo, exist_ok=True)
        mock_run.side_effect = [
            _make_mock_result(stdout="a.cpp\n"),
            None,
            _make_mock_result(stdout=""),
            _make_mock_result(stdout=""),  # git restore
        ]
        _ = run_clang_format_and_count_changes(
            "BasedOnStyle: LLVM\n", repo, 1, False, 100.0, 42
        )
        restore_call = mock_run.call_args_list[3]
        assert "git" in restore_call[1].get("cmd", restore_call[0][0])
        assert "restore" in restore_call[1].get("cmd", restore_call[0][0])

    @patch("src.repo_formatter.run_command")
    def test_clang_format_not_found_raises(self, mock_run, tmp_path):
        repo = str(tmp_path)
        os.makedirs(repo, exist_ok=True)
        mock_run.side_effect = [
            _make_mock_result(stdout="a.cpp\n"),
            FileNotFoundError(),
            None,
        ]
        with pytest.raises(ClangFormatWorkerError, match="clang-format not found"):
            _ = run_clang_format_and_count_changes(
                "BasedOnStyle: LLVM\n", repo, 1, False, 100.0, 42
            )

    @patch("src.repo_formatter.run_command")
    def test_other_clang_format_error_raises(self, mock_run, tmp_path):
        """Non-invalid, non-crash errors raise ClangFormatWorkerError."""
        repo = str(tmp_path)
        os.makedirs(repo, exist_ok=True)
        mock_run.side_effect = [
            _make_mock_result(stdout="a.cpp\n"),
            subprocess.CalledProcessError(
                1, ["clang-format"], output="", stderr="some other error"
            ),
            None,
        ]
        with pytest.raises(ClangFormatWorkerError, match="clang-format critical error"):
            _ = run_clang_format_and_count_changes(
                "BasedOnStyle: LLVM\n", repo, 1, False, 100.0, 42
            )

    @patch("src.repo_formatter.run_command")
    def test_git_restore_error_logged(self, mock_run, tmp_path, capsys):
        """When git restore fails, an error is printed but execution continues."""
        repo = str(tmp_path)
        os.makedirs(repo, exist_ok=True)
        mock_run.side_effect = [
            _make_mock_result(stdout="a.cpp\n"),
            None,
            _make_mock_result(stdout=""),
            subprocess.CalledProcessError(1, ["git", "restore"]),
        ]
        result = run_clang_format_and_count_changes(
            "BasedOnStyle: LLVM\n", repo, 1, False, 100.0, 42
        )
        assert result == 0
        captured = capsys.readouterr()
        assert "Error resetting git repository" in captured.err

    @patch("src.repo_formatter.run_command")
    def test_debug_prints_cache_miss(self, mock_run, tmp_path, capsys):
        """Debug mode prints cache miss message."""
        repo = str(tmp_path)
        os.makedirs(repo, exist_ok=True)
        mock_run.side_effect = [
            _make_mock_result(stdout="a.cpp\n"),
            None,
            _make_mock_result(stdout=""),
            None,
        ]
        _ = run_clang_format_and_count_changes(
            "BasedOnStyle: LLVM\n",
            repo,
            1,
            debug=True,
            file_sample_percentage=100.0,
            random_seed=42,
        )
        captured = capsys.readouterr()
        assert "cache miss" in captured.err

    @patch("src.repo_formatter.run_command")
    def test_debug_prints_sampling(self, mock_run, tmp_path, capsys):
        """Debug mode prints sampling info."""
        repo = str(tmp_path)
        os.makedirs(repo, exist_ok=True)
        files = "\n".join([f"file{i}.cpp" for i in range(10)]) + "\n"
        mock_run.side_effect = [
            _make_mock_result(stdout=files),
            None,
            _make_mock_result(stdout=""),
            None,
        ]
        _ = run_clang_format_and_count_changes(
            "BasedOnStyle: LLVM\n",
            repo,
            1,
            debug=True,
            file_sample_percentage=50.0,
            random_seed=42,
        )
        captured = capsys.readouterr()
        assert "Sampling" in captured.err

    @patch("src.repo_formatter.run_command")
    def test_invalid_config_debug_prints(self, mock_run, tmp_path, capsys):
        """Debug mode prints details for invalid config."""
        repo = str(tmp_path)
        os.makedirs(repo, exist_ok=True)
        mock_run.side_effect = [
            _make_mock_result(stdout="a.cpp\n"),
            subprocess.CalledProcessError(
                1, ["clang-format"], output="", stderr="cannot be used with"
            ),
            None,
        ]
        _ = run_clang_format_and_count_changes(
            "BadOption: true\n",
            repo,
            1,
            debug=True,
            file_sample_percentage=100.0,
            random_seed=42,
        )
        captured = capsys.readouterr()
        assert "invalid configuration" in captured.err

    @patch("src.repo_formatter.run_command")
    def test_invalid_config_debug_prints_stdout(self, mock_run, tmp_path, capsys):
        """Debug mode prints stdout when present for invalid config."""
        repo = str(tmp_path)
        os.makedirs(repo, exist_ok=True)
        mock_run.side_effect = [
            _make_mock_result(stdout="a.cpp\n"),
            subprocess.CalledProcessError(
                1, ["clang-format"], output="some stdout", stderr="cannot be used with"
            ),
            None,
        ]
        _ = run_clang_format_and_count_changes(
            "BadOption: true\n",
            repo,
            1,
            debug=True,
            file_sample_percentage=100.0,
            random_seed=42,
        )
        captured = capsys.readouterr()
        assert "Stdout:" in captured.err
        assert "some stdout" in captured.err

    @patch("src.repo_formatter.run_command")
    def test_timeout_debug_prints(self, mock_run, tmp_path, capsys):
        """Debug mode prints timeout details."""
        repo = str(tmp_path)
        os.makedirs(repo, exist_ok=True)
        mock_run.side_effect = [
            _make_mock_result(stdout="a.cpp\n"),
            subprocess.TimeoutExpired(["clang-format"], 120),
            None,
        ]
        _ = run_clang_format_and_count_changes(
            "BasedOnStyle: LLVM\n",
            repo,
            1,
            debug=True,
            file_sample_percentage=100.0,
            random_seed=42,
        )
        captured = capsys.readouterr()
        assert "timed out" in captured.err

    @patch("src.repo_formatter.run_command")
    def test_diff_unparseable_returns_zero(self, mock_run, tmp_path, capsys):
        """When diff output doesn't match insertions/deletions patterns, return 0."""
        repo = str(tmp_path)
        os.makedirs(repo, exist_ok=True)
        mock_run.side_effect = [
            _make_mock_result(stdout="a.cpp\n"),
            None,
            _make_mock_result(stdout="some unparseable output"),
            None,
        ]
        result = run_clang_format_and_count_changes(
            "BasedOnStyle: LLVM\n",
            repo,
            1,
            debug=True,
            file_sample_percentage=100.0,
            random_seed=42,
        )
        assert result == 0
        captured = capsys.readouterr()
        assert "Could not parse insertions/deletions" in captured.err

    @patch("src.repo_formatter.run_command")
    def test_debug_prints_no_files(self, mock_run, tmp_path, capsys):
        """Debug mode prints message when no C/C++ files found."""
        repo = str(tmp_path)
        os.makedirs(repo, exist_ok=True)
        mock_run.side_effect = [
            _make_mock_result(stdout=""),
            None,
        ]
        _ = run_clang_format_and_count_changes(
            "BasedOnStyle: LLVM\n",
            repo,
            1,
            debug=True,
            file_sample_percentage=100.0,
            random_seed=42,
        )
        captured = capsys.readouterr()
        assert "No C/C++/Objective-C files found" in captured.err

    @patch("src.repo_formatter.run_command")
    def test_sampling_over_100_skips_sampling(self, mock_run, tmp_path):
        """When file_sample_percentage >= 100, no sampling is applied."""
        repo = str(tmp_path)
        os.makedirs(repo, exist_ok=True)
        files = "a.cpp\nb.cpp\nc.cpp\n"
        mock_run.side_effect = [
            _make_mock_result(stdout=files),
            None,
            _make_mock_result(stdout=""),
            None,
        ]
        _ = run_clang_format_and_count_changes(
            "BasedOnStyle: LLVM\n",
            repo,
            1,
            False,
            file_sample_percentage=200.0,
            random_seed=42,
        )
        clang_call = mock_run.call_args_list[1]
        cmd = clang_call[1].get("cmd", clang_call[0][0])
        file_args = [a for a in cmd if a.endswith(".cpp")]
        assert len(file_args) == 3  # all files, no sampling

    @patch("src.repo_formatter.run_command")
    def test_non_seeded_random_sampling(self, mock_run, tmp_path):
        """Sampling without random_seed uses global random.sample."""
        repo = str(tmp_path)
        os.makedirs(repo, exist_ok=True)
        files = "\n".join([f"file{i}.cpp" for i in range(10)]) + "\n"
        mock_run.side_effect = [
            _make_mock_result(stdout=files),
            None,
            _make_mock_result(stdout=""),
            None,
        ]
        result = run_clang_format_and_count_changes(
            "BasedOnStyle: LLVM\n", repo, 1, False, 50.0, random_seed=42
        )
        assert result == 0
        clang_call = mock_run.call_args_list[1]
        cmd = clang_call[1].get("cmd", clang_call[0][0])
        file_args = [a for a in cmd if a.endswith(".cpp")]
        assert len(file_args) == 5

    @patch("src.repo_formatter.run_command")
    def test_crash_debug_prints(self, mock_run, tmp_path, capsys):
        """Debug mode prints crash details for PLEASE submit a bug report."""
        repo = str(tmp_path)
        os.makedirs(repo, exist_ok=True)
        mock_run.side_effect = [
            _make_mock_result(stdout="a.cpp\n"),
            subprocess.CalledProcessError(
                1, ["clang-format"], output="", stderr="PLEASE submit a bug report"
            ),
            None,
        ]
        _ = run_clang_format_and_count_changes(
            "BasedOnStyle: LLVM\n",
            repo,
            1,
            debug=True,
            file_sample_percentage=100.0,
            random_seed=42,
        )
        captured = capsys.readouterr()
        assert "crashed" in captured.err

    @patch("src.repo_formatter.run_command")
    def test_crash_debug_prints_stdout(self, mock_run, tmp_path, capsys):
        """Debug mode prints stdout when present for crash."""
        repo = str(tmp_path)
        os.makedirs(repo, exist_ok=True)
        mock_run.side_effect = [
            _make_mock_result(stdout="a.cpp\n"),
            subprocess.CalledProcessError(
                1,
                ["clang-format"],
                output="crash stdout",
                stderr="PLEASE submit a bug report",
            ),
            None,
        ]
        _ = run_clang_format_and_count_changes(
            "BasedOnStyle: LLVM\n",
            repo,
            1,
            debug=True,
            file_sample_percentage=100.0,
            random_seed=42,
        )
        captured = capsys.readouterr()
        assert "Stdout:" in captured.err
        assert "crash stdout" in captured.err
        assert "PLEASE submit a bug report" in captured.err

    @patch("src.repo_formatter.run_command")
    def test_critical_error_prints_stdout(self, mock_run, tmp_path, capsys):
        """Critical error path prints stdout when present."""
        repo = str(tmp_path)
        os.makedirs(repo, exist_ok=True)
        mock_run.side_effect = [
            _make_mock_result(stdout="a.cpp\n"),
            subprocess.CalledProcessError(
                1, ["clang-format"], output="some stdout", stderr="some other error"
            ),
            None,
        ]
        with pytest.raises(ClangFormatWorkerError):
            _ = run_clang_format_and_count_changes(
                "BasedOnStyle: LLVM\n", repo, 1, False, 100.0, 42
            )
        captured = capsys.readouterr()
        assert "some stdout" in captured.err

    @patch("src.repo_formatter.run_command")
    @patch("src.repo_formatter.shutil.copyfile")
    def test_config_copy_ioerror_logged(self, mock_copy, mock_run, tmp_path, capsys):
        """When copying error config fails, IOError is logged."""
        repo = str(tmp_path)
        os.makedirs(repo, exist_ok=True)
        mock_copy.side_effect = OSError("permission denied")
        mock_run.side_effect = [
            _make_mock_result(stdout="a.cpp\n"),
            subprocess.CalledProcessError(
                1, ["clang-format"], output="", stderr="some other error"
            ),
            None,
        ]
        with pytest.raises(ClangFormatWorkerError):
            _ = run_clang_format_and_count_changes(
                "BasedOnStyle: LLVM\n", repo, 1, False, 100.0, 42
            )
        captured = capsys.readouterr()
        assert "Error copying temporary config file" in captured.err

    @patch("src.repo_formatter.run_command")
    @patch("os.remove")
    def test_temp_config_remove_oserror_logged(
        self, mock_remove, mock_run, tmp_path, capsys
    ):
        """When removing temp config fails, OSError is logged."""
        repo = str(tmp_path)
        os.makedirs(repo, exist_ok=True)
        mock_remove.side_effect = OSError("permission denied")
        mock_run.side_effect = [
            _make_mock_result(stdout="a.cpp\n"),
            None,
            _make_mock_result(stdout=""),
            None,
        ]
        _ = run_clang_format_and_count_changes(
            "BasedOnStyle: LLVM\n", repo, 1, False, 100.0, 42
        )
        captured = capsys.readouterr()
        assert "Error removing temporary config file" in captured.err


class TestIsBuildDir:
    """Test _is_build_dir filters common build/output directories."""

    def test_build_dirs_excluded(self):
        assert _is_build_dir("build/data.cpp") is True
        assert (
            _is_build_dir("build/CMakeFiles/4.3.3/CompilerIdCXX/CMakeCXXCompilerId.cpp")
            is True
        )
        assert _is_build_dir("out/src/main.cpp") is True
        assert _is_build_dir("target/debug/lib.cpp") is True
        assert _is_build_dir(".cmake/api/json/file.cpp") is True
        assert _is_build_dir("dist/output.h") is True
        assert _is_build_dir("cmake/generated/foo.c") is True
        assert _is_build_dir("_build/temp.o") is True

    def test_source_files_not_excluded(self):
        assert _is_build_dir("src/main.cpp") is False
        assert _is_build_dir("include/foo.h") is False
        assert _is_build_dir("lib/util.c") is False
        assert _is_build_dir("main.cpp") is False
