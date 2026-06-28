import os
import subprocess
import sys
import re
import shutil
import random
import threading
from .utils import run_command

# Define timeouts for external commands
CLANG_FORMAT_TIMEOUT = 120  # seconds
GIT_COMMAND_TIMEOUT = 60  # seconds


class ClangFormatWorkerError(Exception):
    """Raised when a worker encounters a fatal error (e.g., clang-format not found)."""

    pass


# Module-level cache for git ls-files results, keyed by repo_path.
# Workers are long-lived processes, so this cache persists across calls.
_file_list_cache: dict[str, list[str]] = {}


def _get_cached_file_list(repo_path: str, debug: bool) -> list[str]:
    """Return the list of tracked C/C++ files for repo_path, caching the result."""
    if repo_path not in _file_list_cache:
        if debug:
            print(
                f"  (cache miss) Computing file list for {repo_path}", file=sys.stderr
            )
        git_ls_files_cmd = [
            "git",
            "ls-files",
            "--",
            "*.c",
            "*.cc",
            "*.cpp",
            "*.cxx",
            "*.h",
            "*.hh",
            "*.hpp",
            "*.hxx",
            "*.m",
            "*.mm",
        ]
        try:
            result = run_command(
                git_ls_files_cmd,
                capture_output=True,
                text=True,
                check=True,
                debug=debug,
                timeout=GIT_COMMAND_TIMEOUT,
                cwd=repo_path,
            )
            _file_list_cache[repo_path] = result.stdout.splitlines()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"Error listing files in repo {repo_path}: {e}", file=sys.stderr)
            _file_list_cache[
                repo_path
            ] = []  # Cache empty list to avoid repeated failures on the same repo
    return _file_list_cache[repo_path]


def run_clang_format_and_count_changes(
    config_string: str,
    repo_path: str,
    process_id: int,
    debug: bool,
    file_sample_percentage: float,
    random_seed: int,
) -> float:
    """
    Runs a clang-format configuration on a repository, counts changes, and resets.

    Args:
        config_string (str): The clang-format configuration as a YAML string.
        repo_path (str): The path to the specific temporary git repository for this worker.
        process_id (int): The unique identifier for the worker process.
        debug (bool): Enable debug output.
        file_sample_percentage (float): Percentage of files to randomly sample for fitness calculation.
        random_seed (int): Seed for random file sampling.

    Returns:
        int or float('inf'): The total number of lines added or deleted by clang-format (>= 0).
                             Returns float('inf') if clang-format reports an invalid configuration
                             (e.g., "cannot be used with") or if it times out.
                             Returns -1 if a non-clang-format error occurs (like git diff or file listing).
                             Raises ClangFormatWorkerError if a fatal error occurs (e.g., clang-format not found).
    """
    # Use a unique temp config filename per invocation to avoid thread races.
    # ThreadPoolExecutor shares the same repo_path across threads, so a shared
    # filename would cause one thread to overwrite or delete another's config.
    thread_id = threading.get_ident()
    temp_config_file = os.path.join(repo_path, f".clang-format.tmp.{thread_id}")
    error_config_dest = "/tmp/clang-format.yml"

    # Write the configuration to the temporary file path inside the repo
    with open(temp_config_file, "w") as tmp_file:
        _ = tmp_file.write(config_string)

    try:
        # Get cached file list (avoids repeated git ls-files calls)
        files_to_format = _get_cached_file_list(repo_path, debug)

        if not files_to_format:
            if debug:
                print(
                    f"Worker {process_id}: No C/C++/Objective-C files found in the repository to format.",
                    file=sys.stderr,
                )
            return 0

        # Apply file sampling if percentage is less than 100%
        if file_sample_percentage < 100.0:
            num_files_to_sample = max(
                1, int(len(files_to_format) * (file_sample_percentage / 100.0))
            )

            if debug:
                print(
                    f"Worker {process_id}:   Sampling {num_files_to_sample} files ({file_sample_percentage:.1f}%) from {len(files_to_format)} available files.",
                    file=sys.stderr,
                )

            rng = random.Random(random_seed)
            sampled_files = rng.sample(files_to_format, num_files_to_sample)
            files_to_format = sampled_files

        # Run clang-format on the files, explicitly using the temporary config file
        clang_format_cmd = [
            "clang-format",
            f"-style=file:{temp_config_file}",
            "-i",
        ] + files_to_format
        try:
            _ = run_command(
                clang_format_cmd,
                check=True,
                capture_output=True,
                text=True,
                debug=debug,
                timeout=CLANG_FORMAT_TIMEOUT,
                cwd=repo_path,
            )
        except FileNotFoundError:
            print(
                f"Worker {process_id}: Error: clang-format command not found. Please ensure it is installed and in your PATH.",
                file=sys.stderr,
            )
            raise ClangFormatWorkerError("clang-format not found")
        except subprocess.TimeoutExpired as e:
            print(
                f"Worker {process_id}: Warning: clang-format command timed out. Treating as high cost.",
                file=sys.stderr,
            )
            if debug:
                print(
                    f"Worker {process_id}: Command: {' '.join(e.cmd)}", file=sys.stderr
                )
                print(f"Worker {process_id}: Timeout: {e.timeout}s", file=sys.stderr)
            return float("inf")
        except subprocess.CalledProcessError as e:
            error_output = (e.stdout or "") + (e.stderr or "")

            invalid_config_patterns = [
                "cannot be used with",
                "Unsuitable",
                "unknown enumerated scalar",
                "Error reading .clang-format",
            ]
            is_invalid_config = any(p in error_output for p in invalid_config_patterns)

            if is_invalid_config:
                if debug:
                    print(
                        f"Worker {process_id}: Warning: clang-format reported an invalid configuration. Treating as high cost.",
                        file=sys.stderr,
                    )
                    print(
                        f"Worker {process_id}: Command: {' '.join(e.cmd)}",
                        file=sys.stderr,
                    )
                    print(
                        f"Worker {process_id}: Exit code: {e.returncode}",
                        file=sys.stderr,
                    )
                    if e.stdout:
                        print(
                            f"Worker {process_id}: Stdout:\n{e.stdout}", file=sys.stderr
                        )
                    if e.stderr:
                        print(
                            f"Worker {process_id}: Stderr:\n{e.stderr}", file=sys.stderr
                        )
                return float("inf")
            elif "PLEASE submit a bug report" in error_output:
                if debug:
                    print(
                        f"Worker {process_id}: Warning: clang-format crashed with the current configuration. Treating as high cost.",
                        file=sys.stderr,
                    )
                    print(
                        f"Worker {process_id}: Command: {' '.join(e.cmd)}",
                        file=sys.stderr,
                    )
                    print(
                        f"Worker {process_id}: Exit code: {e.returncode}",
                        file=sys.stderr,
                    )
                    if e.stdout:
                        print(
                            f"Worker {process_id}: Stdout:\n{e.stdout}", file=sys.stderr
                        )
                    if e.stderr:
                        print(
                            f"Worker {process_id}: Stderr:\n{e.stderr}", file=sys.stderr
                        )
                return float("inf")
            else:
                print(
                    f"Worker {process_id}: Error running clang-format with the current configuration:",
                    file=sys.stderr,
                )
                print(
                    f"Worker {process_id}: Command: {' '.join(e.cmd)}", file=sys.stderr
                )
                print(
                    f"Worker {process_id}: Exit code: {e.returncode}", file=sys.stderr
                )
                if e.stdout:
                    print(f"Worker {process_id}: Stdout:\n{e.stdout}", file=sys.stderr)
                if e.stderr:
                    print(f"Worker {process_id}: Stderr:\n{e.stderr}", file=sys.stderr)

                try:
                    _ = shutil.copyfile(temp_config_file, error_config_dest)
                    print(
                        f"Worker {process_id}: Configuration causing the error copied to {error_config_dest} for inspection.",
                        file=sys.stderr,
                    )
                except OSError as copy_error:
                    print(
                        f"Worker {process_id}: Error copying temporary config file to {error_config_dest}: {copy_error}",
                        file=sys.stderr,
                    )

                raise ClangFormatWorkerError("clang-format critical error")

        # Count changes using git diff --shortstat
        git_diff_cmd = ["git", "diff", "--shortstat"]
        try:
            result = run_command(
                git_diff_cmd,
                capture_output=True,
                text=True,
                check=True,
                debug=debug,
                timeout=GIT_COMMAND_TIMEOUT,
                cwd=repo_path,
            )
            diff_output = result.stdout.strip()

            total_changes = 0
            insertions_match = re.search(r"(\d+) insertions?\(\+\)", diff_output)
            deletions_match = re.search(r"(\d+) deletions?\(-\)", diff_output)

            if insertions_match:
                total_changes += int(insertions_match.group(1))
            if deletions_match:
                total_changes += int(deletions_match.group(1))

            if not insertions_match and not deletions_match and diff_output:
                if debug:
                    print(
                        f"Worker {process_id}: Warning: Could not parse insertions/deletions from diff output: '{diff_output}'",
                        file=sys.stderr,
                    )
                total_changes = 0
            elif not diff_output:
                total_changes = 0

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"Worker {process_id}: Error running git diff: {e}", file=sys.stderr)
            return -1

        return total_changes

    finally:
        # Reset the repository changes
        git_restore_cmd = ["git", "restore", "."]
        try:
            _ = run_command(
                git_restore_cmd,
                check=True,
                capture_output=True,
                text=True,
                debug=debug,
                timeout=GIT_COMMAND_TIMEOUT,
                cwd=repo_path,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            stderr_msg = ""
            if (
                isinstance(e, subprocess.CalledProcessError) and e.stderr
            ):  # pragma: no cover
                stderr_msg = f" (stderr: {e.stderr.strip()})"
            print(
                f"Worker {process_id}: Error resetting git repository: {e}{stderr_msg}",
                file=sys.stderr,
            )

        # Clean up the temporary config file
        if os.path.exists(temp_config_file):
            try:
                os.remove(temp_config_file)
            except OSError as e:
                print(
                    f"Worker {process_id}: Error removing temporary config file {temp_config_file}: {e}",
                    file=sys.stderr,
                )
