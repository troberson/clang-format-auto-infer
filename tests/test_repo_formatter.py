"""Tests for repo_formatter — mock run_command to avoid real clang-format/git."""

import os
import subprocess
from unittest.mock import patch, MagicMock
import pytest

from src.repo_formatter import run_clang_format_and_count_changes


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
            None,                                      # clang-format (no output)
            _make_mock_result(stdout=""),              # git diff (no changes)
            None,                                      # git restore
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
            _make_mock_result(stdout=" 2 files changed, 5 insertions(+), 3 deletions(-)\n"),
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
            subprocess.CalledProcessError(1, ["clang-format"], output="", stderr="cannot be used with"),
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
            subprocess.CalledProcessError(1, ["clang-format"], output="", stderr="PLEASE submit a bug report"),
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
            subprocess.CalledProcessError(1, ["git", "diff"], output="", stderr="error"),
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
    def test_file_listing_error_returns_minus_one(self, mock_run, tmp_path):
        repo = str(tmp_path)
        os.makedirs(repo, exist_ok=True)
        mock_run.side_effect = [
            subprocess.CalledProcessError(1, ["git", "ls-files"]),
            None,
        ]
        result = run_clang_format_and_count_changes(
            "BasedOnStyle: LLVM\n", repo, 1, False, 100.0, 42
        )
        assert result == -1

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
        run_clang_format_and_count_changes(
            "BasedOnStyle: LLVM\n", repo, 1, False, 100.0, 42
        )
        assert not os.path.exists(os.path.join(repo, ".clang-format.tmp"))

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
        run_clang_format_and_count_changes(
            "BasedOnStyle: LLVM\n", repo, 1, False, 100.0, 42
        )
        restore_call = mock_run.call_args_list[3]
        assert "git" in restore_call[1].get("cmd", restore_call[0][0])
        assert "restore" in restore_call[1].get("cmd", restore_call[0][0])
