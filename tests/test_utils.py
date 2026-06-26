"""Tests for src.utils — run_command wrapper around subprocess.run."""

import subprocess
import sys
from io import StringIO
from unittest.mock import patch, MagicMock

import pytest

from src.utils import run_command


class TestRunCommand:
    def test_success_returns_completed_process(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                spec=subprocess.CompletedProcess, returncode=0, stdout="", stderr=""
            )
            result = run_command(["echo", "hello"])
            mock_run.assert_called_once_with(
                ["echo", "hello"],
                capture_output=False,
                text=False,
                check=False,
                cwd=None,
                timeout=None,
            )
            assert result.returncode == 0

    def test_passes_all_kwargs_through(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                spec=subprocess.CompletedProcess, returncode=0, stdout="ok", stderr=""
            )
            _ = run_command(
                ["git", "ls-files"],
                capture_output=True,
                text=True,
                check=True,
                cwd="/tmp",
                debug=False,
                timeout=30,
            )
            mock_run.assert_called_once_with(
                ["git", "ls-files"],
                capture_output=True,
                text=True,
                check=True,
                cwd="/tmp",
                timeout=30,
            )

    def test_debug_prints_command(self):
        with patch(
            "subprocess.run",
            return_value=MagicMock(spec=subprocess.CompletedProcess, returncode=0),
        ):
            with patch.object(sys, "stderr", new_callable=StringIO) as stderr:
                _ = run_command(["ls", "-la"], cwd="/tmp", timeout=10, debug=True)
                output = stderr.getvalue()
                assert "Executing command: ls -la" in output
                assert "cwd: /tmp" in output
                assert "timeout: 10s" in output

    def test_debug_prints_output_when_captured(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                spec=subprocess.CompletedProcess,
                returncode=0,
                stdout="hello",
                stderr="warn",
            )
            with patch.object(sys, "stderr", new_callable=StringIO) as stderr:
                _ = run_command(["echo"], capture_output=True, debug=True)
                output = stderr.getvalue()
                assert "Command stdout:" in output
                assert "hello" in output
                assert "Command stderr:" in output
                assert "warn" in output

    def test_file_not_found_raises(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("nope")):
            with patch.object(sys, "stderr", new_callable=StringIO) as stderr:
                with pytest.raises(FileNotFoundError):
                    _ = run_command(["nonexistent-command"])
                assert "Command not found: nonexistent-command" in stderr.getvalue()

    def test_timeout_expired_raises(self):
        exc = subprocess.TimeoutExpired(["slow-cmd"], 5)
        with patch("subprocess.run", side_effect=exc):
            with patch.object(sys, "stderr", new_callable=StringIO) as stderr:
                with pytest.raises(subprocess.TimeoutExpired):
                    _ = run_command(["slow-cmd"], timeout=5)
                assert "timed out after 5 seconds" in stderr.getvalue()

    def test_called_process_error_raises(self):
        exc = subprocess.CalledProcessError(1, ["bad-cmd"], output="", stderr="err")
        with patch("subprocess.run", side_effect=exc):
            with pytest.raises(subprocess.CalledProcessError):
                _ = run_command(["bad-cmd"], check=True)

    def test_called_process_error_debug_prints_details(self):
        exc = subprocess.CalledProcessError(2, ["bad-cmd"], output="", stderr="err")
        with patch("subprocess.run", side_effect=exc):
            with patch.object(sys, "stderr", new_callable=StringIO) as stderr:
                with pytest.raises(subprocess.CalledProcessError):
                    _ = run_command(["bad-cmd"], check=True, debug=True)
                output = stderr.getvalue()
                assert "exit code 2" in output
                assert "err" in output
