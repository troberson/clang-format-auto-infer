---
name: commit-checks
description: >-
  Pre-commit coverage policy and triage workflow for clang-format-auto-infer.
  Triggers on git commits, coverage failures, or when asked about missing
  test coverage. Enforces 100% coverage before any commit lands.
---

# Commit Coverage Checks

Every commit to `clang-format-auto-infer` must pass the pre-commit hook, which
includes a **100% coverage gate**. If coverage drops below 100%, the hook
fails and prints the missing lines.

## Coverage Triage Workflow

When the pre-commit hook reports missing coverage, classify each missing line
into one of three categories and act accordingly:

### 1. Needs to be tested -> add a test

The line is reachable and testable. Write a unit test that exercises it.

**When:** Normal code paths, error handling that can be mocked, branch
conditions that can be triggered with specific inputs.

**How:**
- Use `@patch` to mock external dependencies (subprocess, file I/O, etc.)
- For error paths, use `side_effect` to inject exceptions
- Verify the line is covered by re-running `pytest --cov=src`

### 2. Rare and not testable -> add `# pragma: no cover`

The line is reachable in theory but impractical to test (requires specific
system state, external tool behavior, or race conditions).

**When:**
- Defensive fallbacks that should never trigger (e.g., cache miss on a
  pre-populated cache)
- Platform-specific branches that don't apply to the test environment
- Error paths requiring real tool crashes with specific output

**How:** Add `# pragma: no cover` on the specific line or block. Include a
brief comment explaining why.

```python
if parts is None:  # pragma: no cover — cache is always populated in __init__
    parts = full_option_path.split(".")
```

### 3. Totally dead code -> prune

The line is unreachable (logic error, leftover from refactoring, impossible
condition).

**When:**
- Code after an unconditional return/raise
- Branches guarded by conditions that are always false
- Leftover stubs from incomplete refactors

**How:** Delete the dead code. If it might be needed later, close the related
bead with a note explaining why it was removed.

## Running Coverage Manually

```bash
.venv/bin/python -m pytest tests/ --cov=src --cov-report=term-missing -q
```

## Pre-commit Hook

Located at `.git/hooks/pre-commit`. Runs:
1. `ruff check` — linting
2. `ruff format --check` — formatting
3. `mypy --check-untyped-defs` — type checking
4. `pytest tests/ -q` — unit tests
5. `pytest --cov=src` — coverage gate (must be 100%)

If coverage < 100%, the hook prints missing lines and the triage categories.
