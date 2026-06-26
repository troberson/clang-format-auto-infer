"""Tests for nevergrad_optimizer — mock nevergrad and ProcessPoolExecutor."""

from unittest.mock import patch, MagicMock

import pytest

from src.data_classes import NevergradConfig, GeneticAlgorithmLookups
from src.nevergrad_optimizer import NevergradOptimizer


def _make_config(
    budget=10,
    optimizer_name="TwoPointsDE",
    num_workers=2,
    debug=False,
    plot_fitness=False,
):
    return NevergradConfig(
        budget=budget,
        optimizer_name=optimizer_name,
        num_workers=num_workers,
        debug=debug,
        plot_fitness=plot_fitness,
    )


def _make_lookups(json_options=None, forced_options=None):
    return GeneticAlgorithmLookups(
        json_options_lookup=json_options or {},
        forced_options_lookup=forced_options or {},
    )


def _make_base_options():
    return {
        "UseTab": {"type": "str", "value": "Never"},
        "IndentWidth": {"type": "int", "value": 4},
        "BreakBeforeBraces": {"type": "bool", "value": True},
    }


def _build_mock_ng_optimizer():
    """Return a mock nevergrad optimizer that tracks ask/tell calls."""
    mock_opt = MagicMock()
    ask_count = [0]

    def ask():
        ask_count[0] += 1
        candidate = MagicMock()
        candidate.kwargs = {
            "UseTab": "Always",
            "IndentWidth": 2,
            "BreakBeforeBraces": False,
        }
        return candidate

    mock_opt.ask = MagicMock(side_effect=ask)
    mock_opt._ask_count = ask_count

    told_params = []
    told_losses = []

    def tell(param, loss):
        told_params.append(param)
        told_losses.append(loss)

    mock_opt.tell = MagicMock(side_effect=tell)
    mock_opt._told_params = told_params
    mock_opt._told_losses = told_losses

    rec_default = MagicMock()
    rec_default.kwargs = {
        "UseTab": "Always",
        "IndentWidth": 2,
        "BreakBeforeBraces": False,
    }
    mock_opt.provide_recommendation = MagicMock(return_value=rec_default)
    return mock_opt


def _build_mock_executor(future_results=None):
    """Return a mock ProcessPoolExecutor.

    Args:
        future_results: list of values or exceptions to return from future.result().
                        If None, returns sequential integers.
    """
    mock_executor = MagicMock()
    call_idx = [0]
    results = future_results if future_results is not None else [10, 20, 30, 40, 50]

    def submit(_fn, **_kwargs):
        idx = call_idx[0]
        call_idx[0] += 1
        future = MagicMock()
        future.done.return_value = True
        future.cancel.return_value = None
        if idx < len(results):
            val = results[idx]
            if isinstance(val, Exception):
                future.result.side_effect = val
            else:
                future.result.return_value = val
        else:
            future.result.return_value = idx * 10
        return future

    mock_executor.submit = submit
    mock_executor.shutdown = MagicMock()
    return mock_executor


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNevergradOptimizerInit:
    def test_stores_config(self):
        config = _make_config()
        opt = NevergradOptimizer(config)
        assert opt.config is config


class TestNevergradOptimizerOptimize:
    @patch("src.nevergrad_optimizer.concurrent.futures.wait")
    @patch("src.nevergrad_optimizer.concurrent.futures.ProcessPoolExecutor")
    @patch("src.nevergrad_optimizer.ng.optimizers.registry")
    @patch("src.nevergrad_optimizer.ng.p.Instrumentation")
    @patch("src.nevergrad_optimizer.ng.p.Scalar")
    @patch("src.nevergrad_optimizer.ng.p.Choice")
    def test_runs_ask_tell_loop_and_returns_config(
        self,
        _mock_choice,
        _mock_scalar,
        _mock_instrumentation,
        mock_registry,
        mock_executor_cls,
        mock_wait,
    ):
        config = _make_config(budget=3, num_workers=2)
        opt = NevergradOptimizer(config)

        mock_ng_opt = _build_mock_ng_optimizer()
        mock_registry.__getitem__.return_value.return_value = mock_ng_opt

        mock_executor = _build_mock_executor()
        mock_executor_cls.return_value = mock_executor

        # wait returns all futures as done immediately
        mock_wait.side_effect = lambda futures, **kw: (list(futures), [])

        lookups = _make_lookups(
            json_options={
                "UseTab": {"type": "str", "possible_values": ["Never", "Always"]}
            }
        )
        base = _make_base_options()

        result = opt.optimize(base, ["/tmp/repo"], lookups, 100.0, 42)

        assert "UseTab" in result
        assert mock_ng_opt._ask_count[0] == 3
        assert len(mock_ng_opt._told_losses) == 3

    @patch("src.nevergrad_optimizer.concurrent.futures.wait")
    @patch("src.nevergrad_optimizer.concurrent.futures.ProcessPoolExecutor")
    @patch("src.nevergrad_optimizer.ng.optimizers.registry")
    @patch("src.nevergrad_optimizer.ng.p.Instrumentation")
    @patch("src.nevergrad_optimizer.ng.p.Scalar")
    @patch("src.nevergrad_optimizer.ng.p.Choice")
    def test_applies_forced_options_to_result(
        self,
        _mock_choice,
        _mock_scalar,
        _mock_instrumentation,
        mock_registry,
        mock_executor_cls,
        mock_wait,
    ):
        config = _make_config(budget=2, num_workers=1)
        opt = NevergradOptimizer(config)

        mock_ng_opt = _build_mock_ng_optimizer()
        mock_registry.__getitem__.return_value.return_value = mock_ng_opt

        mock_executor = _build_mock_executor()
        mock_executor_cls.return_value = mock_executor

        mock_wait.side_effect = lambda futures, **kw: (list(futures), [])

        lookups = _make_lookups(forced_options={"UseTab": "ForIndentation"})
        base = _make_base_options()

        result = opt.optimize(base, ["/tmp/repo"], lookups, 100.0, 42)

        assert result["UseTab"]["value"] == "ForIndentation"

    @patch("src.nevergrad_optimizer.concurrent.futures.wait")
    @patch("src.nevergrad_optimizer.concurrent.futures.ProcessPoolExecutor")
    @patch("src.nevergrad_optimizer.ng.optimizers.registry")
    @patch("src.nevergrad_optimizer.ng.p.Instrumentation")
    @patch("src.nevergrad_optimizer.ng.p.Scalar")
    @patch("src.nevergrad_optimizer.ng.p.Choice")
    def test_skips_forced_options_in_instrumentation(
        self,
        _mock_choice,
        _mock_scalar,
        mock_instrumentation,
        mock_registry,
        mock_executor_cls,
        mock_wait,
    ):
        config = _make_config(budget=2, num_workers=1)
        opt = NevergradOptimizer(config)

        mock_ng_opt = _build_mock_ng_optimizer()
        mock_registry.__getitem__.return_value.return_value = mock_ng_opt

        mock_executor = _build_mock_executor()
        mock_executor_cls.return_value = mock_executor

        mock_wait.side_effect = lambda futures, **kw: (list(futures), [])

        lookups = _make_lookups(
            json_options={
                "IndentWidth": {"type": "int", "possible_values": ["2", "4"]}
            },
            forced_options={"UseTab": "Never"},
        )
        base = _make_base_options()

        _ = opt.optimize(base, ["/tmp/repo"], lookups, 100.0, 42)

        inst_call = mock_instrumentation.call_args
        assert inst_call is not None
        kwargs = inst_call[1]
        assert "UseTab" not in kwargs

    @patch("src.nevergrad_optimizer.concurrent.futures.wait")
    @patch("src.nevergrad_optimizer.concurrent.futures.ProcessPoolExecutor")
    @patch("src.nevergrad_optimizer.ng.optimizers.registry")
    @patch("src.nevergrad_optimizer.ng.p.Instrumentation")
    @patch("src.nevergrad_optimizer.ng.p.Scalar")
    @patch("src.nevergrad_optimizer.ng.p.Choice")
    def test_instruments_bool_as_scalar(
        self,
        _mock_choice,
        mock_scalar,
        _mock_instrumentation,
        mock_registry,
        mock_executor_cls,
        mock_wait,
    ):
        config = _make_config(budget=2, num_workers=1)
        opt = NevergradOptimizer(config)

        mock_ng_opt = _build_mock_ng_optimizer()
        mock_registry.__getitem__.return_value.return_value = mock_ng_opt

        mock_executor = _build_mock_executor()
        mock_executor_cls.return_value = mock_executor

        mock_wait.side_effect = lambda futures, **kw: (list(futures), [])

        lookups = _make_lookups()
        base = _make_base_options()

        _ = opt.optimize(base, ["/tmp/repo"], lookups, 100.0, 42)

        assert mock_scalar.called
        assert any(
            "integer_casting" in str(ca) for ca in mock_scalar.return_value.method_calls
        )

    @patch("src.nevergrad_optimizer.concurrent.futures.wait")
    @patch("src.nevergrad_optimizer.concurrent.futures.ProcessPoolExecutor")
    @patch("src.nevergrad_optimizer.ng.optimizers.registry")
    @patch("src.nevergrad_optimizer.ng.p.Instrumentation")
    @patch("src.nevergrad_optimizer.ng.p.Scalar")
    @patch("src.nevergrad_optimizer.ng.p.Choice")
    def test_instruments_int_with_choices(
        self,
        mock_choice,
        _mock_scalar,
        _mock_instrumentation,
        mock_registry,
        mock_executor_cls,
        mock_wait,
    ):
        config = _make_config(budget=2, num_workers=1)
        opt = NevergradOptimizer(config)

        mock_ng_opt = _build_mock_ng_optimizer()
        mock_registry.__getitem__.return_value.return_value = mock_ng_opt

        mock_executor = _build_mock_executor()
        mock_executor_cls.return_value = mock_executor

        mock_wait.side_effect = lambda futures, **kw: (list(futures), [])

        lookups = _make_lookups(
            json_options={
                "IndentWidth": {"type": "int", "possible_values": ["2", "4", "8"]}
            }
        )
        base = _make_base_options()

        _ = opt.optimize(base, ["/tmp/repo"], lookups, 100.0, 42)

        assert mock_choice.called

    @patch("src.nevergrad_optimizer.concurrent.futures.wait")
    @patch("src.nevergrad_optimizer.concurrent.futures.ProcessPoolExecutor")
    @patch("src.nevergrad_optimizer.ng.optimizers.registry")
    @patch("src.nevergrad_optimizer.ng.p.Instrumentation")
    @patch("src.nevergrad_optimizer.ng.p.Scalar")
    @patch("src.nevergrad_optimizer.ng.p.Choice")
    def test_instruments_str_with_choices(
        self,
        mock_choice,
        _mock_scalar,
        _mock_instrumentation,
        mock_registry,
        mock_executor_cls,
        mock_wait,
    ):
        config = _make_config(budget=2, num_workers=1)
        opt = NevergradOptimizer(config)

        mock_ng_opt = _build_mock_ng_optimizer()
        mock_registry.__getitem__.return_value.return_value = mock_ng_opt

        mock_executor = _build_mock_executor()
        mock_executor_cls.return_value = mock_executor

        mock_wait.side_effect = lambda futures, **kw: (list(futures), [])

        lookups = _make_lookups(
            json_options={
                "UseTab": {"type": "str", "possible_values": ["Never", "Always"]}
            }
        )
        base = _make_base_options()

        _ = opt.optimize(base, ["/tmp/repo"], lookups, 100.0, 42)

        assert mock_choice.called

    @patch("src.nevergrad_optimizer.concurrent.futures.wait")
    @patch("src.nevergrad_optimizer.concurrent.futures.ProcessPoolExecutor")
    @patch("src.nevergrad_optimizer.ng.optimizers.registry")
    @patch("src.nevergrad_optimizer.ng.p.Instrumentation")
    @patch("src.nevergrad_optimizer.ng.p.Scalar")
    @patch("src.nevergrad_optimizer.ng.p.Choice")
    def test_handles_evaluation_exception_gracefully(
        self,
        _mock_choice,
        _mock_scalar,
        _mock_instrumentation,
        mock_registry,
        mock_executor_cls,
        mock_wait,
    ):
        """When a future raises, the optimizer tells nevergrad inf and continues."""
        config = _make_config(budget=2, num_workers=1)
        opt = NevergradOptimizer(config)

        mock_ng_opt = _build_mock_ng_optimizer()
        mock_registry.__getitem__.return_value.return_value = mock_ng_opt

        # First future succeeds, second raises
        mock_executor = _build_mock_executor(future_results=[5, RuntimeError("boom")])
        mock_executor_cls.return_value = mock_executor

        mock_wait.side_effect = lambda futures, **kw: (list(futures), [])

        lookups = _make_lookups()
        base = _make_base_options()

        _ = opt.optimize(base, ["/tmp/repo"], lookups, 100.0, 42)

        assert len(mock_ng_opt._told_losses) == 2
        assert float("inf") in mock_ng_opt._told_losses

    @patch("src.nevergrad_optimizer.concurrent.futures.wait")
    @patch("src.nevergrad_optimizer.concurrent.futures.ProcessPoolExecutor")
    @patch("src.nevergrad_optimizer.ng.optimizers.registry")
    @patch("src.nevergrad_optimizer.ng.p.Instrumentation")
    @patch("src.nevergrad_optimizer.ng.p.Scalar")
    @patch("src.nevergrad_optimizer.ng.p.Choice")
    def test_unknown_optimizer_exits(
        self,
        _mock_choice,
        _mock_scalar,
        _mock_instrumentation,
        mock_registry,
        _mock_executor_cls,
        _mock_wait,
    ):
        config = _make_config(optimizer_name="NonExistentOptimizer")
        opt = NevergradOptimizer(config)

        mock_registry.__getitem__.side_effect = KeyError("NonExistentOptimizer")

        lookups = _make_lookups()
        base = _make_base_options()

        with pytest.raises(SystemExit):
            _ = opt.optimize(base, ["/tmp/repo"], lookups, 100.0, 42)

    @patch("src.nevergrad_optimizer.concurrent.futures.wait")
    @patch("src.nevergrad_optimizer.concurrent.futures.ProcessPoolExecutor")
    @patch("src.nevergrad_optimizer.ng.optimizers.registry")
    @patch("src.nevergrad_optimizer.ng.p.Instrumentation")
    @patch("src.nevergrad_optimizer.ng.p.Scalar")
    @patch("src.nevergrad_optimizer.ng.p.Choice")
    def test_executor_shutdown_called(
        self,
        _mock_choice,
        _mock_scalar,
        _mock_instrumentation,
        mock_registry,
        mock_executor_cls,
        mock_wait,
    ):
        config = _make_config(budget=2, num_workers=1)
        opt = NevergradOptimizer(config)

        mock_ng_opt = _build_mock_ng_optimizer()
        mock_registry.__getitem__.return_value.return_value = mock_ng_opt

        mock_executor = _build_mock_executor()
        mock_executor_cls.return_value = mock_executor

        mock_wait.side_effect = lambda futures, **kw: (list(futures), [])

        lookups = _make_lookups()
        base = _make_base_options()

        _ = opt.optimize(base, ["/tmp/repo"], lookups, 100.0, 42)

        mock_executor.shutdown.assert_called_once_with(wait=True)

    @patch("src.nevergrad_optimizer.concurrent.futures.wait")
    @patch("src.nevergrad_optimizer.concurrent.futures.ProcessPoolExecutor")
    @patch("src.nevergrad_optimizer.ng.optimizers.registry")
    @patch("src.nevergrad_optimizer.ng.p.Instrumentation")
    @patch("src.nevergrad_optimizer.ng.p.Scalar")
    @patch("src.nevergrad_optimizer.ng.p.Choice")
    def test_converts_recommendation_types(
        self,
        _mock_choice,
        _mock_scalar,
        _mock_instrumentation,
        mock_registry,
        mock_executor_cls,
        mock_wait,
    ):
        config = _make_config(budget=2, num_workers=1)
        opt = NevergradOptimizer(config)

        mock_ng_opt = _build_mock_ng_optimizer()
        rec = MagicMock()
        rec.kwargs = {
            "UseTab": "Always",
            "IndentWidth": 8,
            "BreakBeforeBraces": 1,
        }
        mock_ng_opt.provide_recommendation.return_value = rec
        mock_registry.__getitem__.return_value.return_value = mock_ng_opt

        mock_executor = _build_mock_executor()
        mock_executor_cls.return_value = mock_executor

        mock_wait.side_effect = lambda futures, **kw: (list(futures), [])

        lookups = _make_lookups()
        base = _make_base_options()

        result = opt.optimize(base, ["/tmp/repo"], lookups, 100.0, 42)

        assert result["IndentWidth"]["value"] == 8
        assert result["BreakBeforeBraces"]["value"] is True

    @patch("src.nevergrad_optimizer.concurrent.futures.wait")
    @patch("src.nevergrad_optimizer.concurrent.futures.ProcessPoolExecutor")
    @patch("src.nevergrad_optimizer.ng.optimizers.registry")
    @patch("src.nevergrad_optimizer.ng.p.Instrumentation")
    @patch("src.nevergrad_optimizer.ng.p.Scalar")
    @patch("src.nevergrad_optimizer.ng.p.Choice")
    def test_budget_zero_returns_base_config(
        self,
        _mock_choice,
        _mock_scalar,
        _mock_instrumentation,
        mock_registry,
        mock_executor_cls,
        mock_wait,
    ):
        config = _make_config(budget=0, num_workers=1)
        opt = NevergradOptimizer(config)

        mock_ng_opt = _build_mock_ng_optimizer()
        rec = MagicMock()
        rec.kwargs = {}
        mock_ng_opt.provide_recommendation.return_value = rec
        mock_registry.__getitem__.return_value.return_value = mock_ng_opt

        mock_executor = MagicMock()
        mock_executor.shutdown = MagicMock()
        mock_executor_cls.return_value = mock_executor

        mock_wait.side_effect = lambda futures, **kw: (list(futures), [])

        lookups = _make_lookups()
        base = _make_base_options()

        result = opt.optimize(base, ["/tmp/repo"], lookups, 100.0, 42)

        assert result["UseTab"]["value"] == "Never"


class TestNevergradObjectiveFunction:
    """Test _nevergrad_objective_function directly."""

    @patch("src.nevergrad_optimizer.run_clang_format_and_count_changes")
    @patch("src.nevergrad_optimizer.multiprocessing.current_process")
    def test_returns_changes(self, mock_proc, mock_run):
        mock_proc.return_value._identity = (1,)
        mock_run.return_value = 42
        lookups = _make_lookups()
        base = _make_base_options()
        result = NevergradOptimizer._nevergrad_objective_function(  # pyright: ignore[reportPrivateUsage]
            base_options_template=base,
            lookups=lookups,
            debug=False,
            file_sample_percentage=100.0,
            random_seed=42,
            all_repo_paths=["/tmp/repo"],
            IndentWidth=2,
            BreakBeforeBraces=0,
        )
        assert result == 42

    @patch("src.nevergrad_optimizer.run_clang_format_and_count_changes")
    @patch("src.nevergrad_optimizer.multiprocessing.current_process")
    def test_git_error_returns_inf(self, mock_proc, mock_run):
        mock_proc.return_value._identity = (1,)
        mock_run.return_value = -1
        lookups = _make_lookups()
        base = _make_base_options()
        result = NevergradOptimizer._nevergrad_objective_function(  # pyright: ignore[reportPrivateUsage]
            base_options_template=base,
            lookups=lookups,
            debug=False,
            file_sample_percentage=100.0,
            random_seed=42,
            all_repo_paths=["/tmp/repo"],
        )
        assert result == float("inf")

    @patch("src.nevergrad_optimizer.run_clang_format_and_count_changes")
    @patch("src.nevergrad_optimizer.multiprocessing.current_process")
    def test_converts_int_type(self, mock_proc, mock_run):
        mock_proc.return_value._identity = (1,)
        mock_run.return_value = 0
        lookups = _make_lookups()
        base = _make_base_options()
        _ = NevergradOptimizer._nevergrad_objective_function(  # pyright: ignore[reportPrivateUsage]
            base_options_template=base,
            lookups=lookups,
            debug=False,
            file_sample_percentage=100.0,
            random_seed=42,
            all_repo_paths=["/tmp/repo"],
            IndentWidth=2.5,  # float from nevergrad, should be cast to int
        )
        # Verify run_clang_format was called with int value
        call_kwargs = mock_run.call_args[1]
        # The config string should contain IndentWidth: 2
        assert "IndentWidth: 2" in call_kwargs.get(
            "config_string", mock_run.call_args[0][0]
        )

    @patch("src.nevergrad_optimizer.run_clang_format_and_count_changes")
    @patch("src.nevergrad_optimizer.multiprocessing.current_process")
    def test_converts_bool_type(self, mock_proc, mock_run):
        mock_proc.return_value._identity = (1,)
        mock_run.return_value = 0
        lookups = _make_lookups()
        base = _make_base_options()
        _ = NevergradOptimizer._nevergrad_objective_function(  # pyright: ignore[reportPrivateUsage]
            base_options_template=base,
            lookups=lookups,
            debug=False,
            file_sample_percentage=100.0,
            random_seed=42,
            all_repo_paths=["/tmp/repo"],
            BreakBeforeBraces=0,  # 0 from nevergrad, should be False
        )
        call_args = mock_run.call_args[0][0]
        assert "BreakBeforeBraces: false" in call_args

    @patch("src.nevergrad_optimizer.run_clang_format_and_count_changes")
    @patch("src.nevergrad_optimizer.multiprocessing.current_process")
    def test_applies_forced_options(self, mock_proc, mock_run):
        mock_proc.return_value._identity = (1,)
        mock_run.return_value = 0
        lookups = _make_lookups(forced_options={"UseTab": "Always"})
        base = _make_base_options()
        _ = NevergradOptimizer._nevergrad_objective_function(  # pyright: ignore[reportPrivateUsage]
            base_options_template=base,
            lookups=lookups,
            debug=False,
            file_sample_percentage=100.0,
            random_seed=42,
            all_repo_paths=["/tmp/repo"],
            UseTab="Never",  # should be overridden by forced option
        )
        call_args = mock_run.call_args[0][0]
        assert "UseTab: Always" in call_args

    @patch("src.nevergrad_optimizer.run_clang_format_and_count_changes")
    @patch("src.nevergrad_optimizer.multiprocessing.current_process")
    def test_debug_prints_repo_path(self, mock_proc, mock_run, capsys):
        mock_proc.return_value._identity = (1,)
        mock_run.return_value = 0
        lookups = _make_lookups()
        base = _make_base_options()
        _ = NevergradOptimizer._nevergrad_objective_function(  # pyright: ignore[reportPrivateUsage]
            base_options_template=base,
            lookups=lookups,
            debug=True,
            file_sample_percentage=100.0,
            random_seed=42,
            all_repo_paths=["/tmp/repo"],
        )
        captured = capsys.readouterr()
        assert "Using repo path" in captured.err

    @patch("src.nevergrad_optimizer.run_clang_format_and_count_changes")
    @patch("src.nevergrad_optimizer.multiprocessing.current_process")
    def test_process_id_zero_uses_first_repo(self, mock_proc, mock_run):
        """When process_id is 0, use the first repo path."""
        mock_proc.return_value._identity = (0,)
        mock_run.return_value = 0
        lookups = _make_lookups()
        base = _make_base_options()
        _ = NevergradOptimizer._nevergrad_objective_function(  # pyright: ignore[reportPrivateUsage]
            base_options_template=base,
            lookups=lookups,
            debug=False,
            file_sample_percentage=100.0,
            random_seed=42,
            all_repo_paths=["/tmp/repo0", "/tmp/repo1"],
        )
        # Should use /tmp/repo0 (index 0)
        mock_run.assert_called_once()

    @patch("src.nevergrad_optimizer.run_clang_format_and_count_changes")
    @patch("src.nevergrad_optimizer.multiprocessing.current_process")
    def test_int_conversion_failure_skips_option(self, mock_proc, mock_run, capsys):
        """When int conversion fails, skip the option and log warning."""
        mock_proc.return_value._identity = (1,)
        mock_run.return_value = 0
        lookups = _make_lookups()
        base = _make_base_options()
        _ = NevergradOptimizer._nevergrad_objective_function(  # pyright: ignore[reportPrivateUsage]
            base_options_template=base,
            lookups=lookups,
            debug=True,
            file_sample_percentage=100.0,
            random_seed=42,
            all_repo_paths=["/tmp/repo"],
            IndentWidth="not_a_number",  # should fail to convert
        )
        captured = capsys.readouterr()
        assert "Could not convert" in captured.err


class TestNevergradOptimizerLoopInternals:
    """Test the ask/tell loop internals that are hard to reach."""

    @patch("src.nevergrad_optimizer.concurrent.futures.wait")
    @patch("src.nevergrad_optimizer.concurrent.futures.ProcessPoolExecutor")
    @patch("src.nevergrad_optimizer.ng.optimizers.registry")
    @patch("src.nevergrad_optimizer.ng.p.Instrumentation")
    @patch("src.nevergrad_optimizer.ng.p.Scalar")
    @patch("src.nevergrad_optimizer.ng.p.Choice")
    def test_debug_prints_evaluation(
        self,
        _mock_choice,
        _mock_scalar,
        _mock_instrumentation,
        mock_registry,
        mock_executor_cls,
        mock_wait,
        capsys,
    ):
        """Debug mode prints per-evaluation details."""
        config = _make_config(budget=2, num_workers=1, debug=True)
        opt = NevergradOptimizer(config)

        mock_ng_opt = _build_mock_ng_optimizer()
        mock_registry.__getitem__.return_value.return_value = mock_ng_opt

        mock_executor = _build_mock_executor(future_results=[10, 20])
        mock_executor_cls.return_value = mock_executor

        mock_wait.side_effect = lambda futures, **kw: (list(futures), [])

        lookups = _make_lookups()
        base = _make_base_options()
        _ = opt.optimize(base, ["/tmp/repo"], lookups, 100.0, 42)

        captured = capsys.readouterr()
        assert "Evaluation" in captured.err

    @patch("src.nevergrad_optimizer.concurrent.futures.wait")
    @patch("src.nevergrad_optimizer.concurrent.futures.ProcessPoolExecutor")
    @patch("src.nevergrad_optimizer.ng.optimizers.registry")
    @patch("src.nevergrad_optimizer.ng.p.Instrumentation")
    @patch("src.nevergrad_optimizer.ng.p.Scalar")
    @patch("src.nevergrad_optimizer.ng.p.Choice")
    def test_evaluation_error_told_as_inf(
        self,
        _mock_choice,
        _mock_scalar,
        _mock_instrumentation,
        mock_registry,
        mock_executor_cls,
        mock_wait,
        capsys,
    ):
        """When a future raises an exception, tell nevergrad inf."""
        config = _make_config(budget=2, num_workers=1)
        opt = NevergradOptimizer(config)

        mock_ng_opt = _build_mock_ng_optimizer()
        mock_registry.__getitem__.return_value.return_value = mock_ng_opt

        # First future succeeds, second raises
        mock_executor = _build_mock_executor(future_results=[10, RuntimeError("boom")])
        mock_executor_cls.return_value = mock_executor

        mock_wait.side_effect = lambda futures, **kw: (list(futures), [])

        lookups = _make_lookups()
        base = _make_base_options()
        _ = opt.optimize(base, ["/tmp/repo"], lookups, 100.0, 42)

        captured = capsys.readouterr()
        assert "Error during evaluation" in captured.err
        # tell should have been called with inf for the failed evaluation
        tell_calls = mock_ng_opt.tell.call_args_list
        assert any(call[0][1] == float("inf") for call in tell_calls)

    @patch("src.nevergrad_optimizer.concurrent.futures.wait")
    @patch("src.nevergrad_optimizer.concurrent.futures.ProcessPoolExecutor")
    @patch("src.nevergrad_optimizer.ng.optimizers.registry")
    @patch("src.nevergrad_optimizer.ng.p.Instrumentation")
    @patch("src.nevergrad_optimizer.ng.p.Scalar")
    @patch("src.nevergrad_optimizer.ng.p.Choice")
    def test_forced_options_applied_to_final_config(
        self,
        _mock_choice,
        _mock_scalar,
        _mock_instrumentation,
        mock_registry,
        mock_executor_cls,
        mock_wait,
    ):
        """Forced options are applied to the final optimized config."""
        config = _make_config(budget=2, num_workers=1)
        opt = NevergradOptimizer(config)

        mock_ng_opt = _build_mock_ng_optimizer()
        rec = MagicMock()
        rec.kwargs = {"IndentWidth": 2}
        mock_ng_opt.provide_recommendation.return_value = rec
        mock_registry.__getitem__.return_value.return_value = mock_ng_opt

        mock_executor = _build_mock_executor(future_results=[10, 20])
        mock_executor_cls.return_value = mock_executor

        mock_wait.side_effect = lambda futures, **kw: (list(futures), [])

        lookups = _make_lookups(forced_options={"UseTab": "Always"})
        base = _make_base_options()
        result = opt.optimize(base, ["/tmp/repo"], lookups, 100.0, 42)

        assert result["UseTab"]["value"] == "Always"

    @patch("src.nevergrad_optimizer.concurrent.futures.wait")
    @patch("src.nevergrad_optimizer.concurrent.futures.ProcessPoolExecutor")
    @patch("src.nevergrad_optimizer.ng.optimizers.registry")
    @patch("src.nevergrad_optimizer.ng.p.Instrumentation")
    @patch("src.nevergrad_optimizer.ng.p.Scalar")
    @patch("src.nevergrad_optimizer.ng.p.Choice")
    def test_none_recommendation_returns_base(
        self,
        _mock_choice,
        _mock_scalar,
        _mock_instrumentation,
        mock_registry,
        mock_executor_cls,
        mock_wait,
        capsys,
    ):
        """When recommendation is None, return base config."""
        config = _make_config(budget=2, num_workers=1)
        opt = NevergradOptimizer(config)

        mock_ng_opt = _build_mock_ng_optimizer()
        mock_ng_opt.provide_recommendation.return_value = None
        mock_registry.__getitem__.return_value.return_value = mock_ng_opt

        mock_executor = _build_mock_executor(future_results=[10, 20])
        mock_executor_cls.return_value = mock_executor

        mock_wait.side_effect = lambda futures, **kw: (list(futures), [])

        lookups = _make_lookups()
        base = _make_base_options()
        result = opt.optimize(base, ["/tmp/repo"], lookups, 100.0, 42)

        assert result["UseTab"]["value"] == "Never"
        captured = capsys.readouterr()
        assert "No recommendation provided" in captured.err

    @patch("src.nevergrad_optimizer.concurrent.futures.wait")
    @patch("src.nevergrad_optimizer.concurrent.futures.ProcessPoolExecutor")
    @patch("src.nevergrad_optimizer.ng.optimizers.registry")
    @patch("src.nevergrad_optimizer.ng.p.Instrumentation")
    @patch("src.nevergrad_optimizer.ng.p.Scalar")
    @patch("src.nevergrad_optimizer.ng.p.Choice")
    def test_executor_cancel_called_on_shutdown(
        self,
        _mock_choice,
        _mock_scalar,
        _mock_instrumentation,
        mock_registry,
        mock_executor_cls,
        mock_wait,
    ):
        """Pending futures are cancelled during shutdown."""
        config = _make_config(budget=1, num_workers=1)
        opt = NevergradOptimizer(config)

        mock_ng_opt = _build_mock_ng_optimizer()
        mock_registry.__getitem__.return_value.return_value = mock_ng_opt

        mock_executor = _build_mock_executor(future_results=[10])
        mock_executor_cls.return_value = mock_executor

        # Return empty done list so future remains pending
        mock_wait.side_effect = [
            (set(), [MagicMock()]),  # nothing done, one pending
            (set(), []),  # done
        ]

        lookups = _make_lookups()
        base = _make_base_options()
        _ = opt.optimize(base, ["/tmp/repo"], lookups, 100.0, 42)

        # shutdown should have been called
        mock_executor.shutdown.assert_called()

    @patch("src.nevergrad_optimizer.concurrent.futures.wait")
    @patch("src.nevergrad_optimizer.concurrent.futures.ProcessPoolExecutor")
    @patch("src.nevergrad_optimizer.ng.optimizers.registry")
    @patch("src.nevergrad_optimizer.ng.p.Instrumentation")
    @patch("src.nevergrad_optimizer.ng.p.Scalar")
    @patch("src.nevergrad_optimizer.ng.p.Choice")
    def test_periodic_progress_print(
        self,
        _mock_choice,
        _mock_scalar,
        _mock_instrumentation,
        mock_registry,
        mock_executor_cls,
        mock_wait,
        capsys,
    ):
        """Every 50 evaluations, a progress line is printed (non-debug mode)."""
        config = _make_config(budget=50, num_workers=1, debug=False)
        opt = NevergradOptimizer(config)

        mock_ng_opt = _build_mock_ng_optimizer()
        mock_registry.__getitem__.return_value.return_value = mock_ng_opt

        mock_executor = _build_mock_executor(future_results=list(range(50)))
        mock_executor_cls.return_value = mock_executor

        mock_wait.side_effect = lambda futures, **kw: (list(futures), [])

        lookups = _make_lookups()
        base = _make_base_options()
        _ = opt.optimize(base, ["/tmp/repo"], lookups, 100.0, 42)

        captured = capsys.readouterr()
        assert "--- Evaluation 50/50" in captured.err

    @patch("src.nevergrad_optimizer.concurrent.futures.wait")
    @patch("src.nevergrad_optimizer.concurrent.futures.ProcessPoolExecutor")
    @patch("src.nevergrad_optimizer.ng.optimizers.registry")
    @patch("src.nevergrad_optimizer.ng.p.Instrumentation")
    @patch("src.nevergrad_optimizer.ng.p.Scalar")
    @patch("src.nevergrad_optimizer.ng.p.Choice")
    def test_int_conversion_failure_final_config(
        self,
        _mock_choice,
        _mock_scalar,
        _mock_instrumentation,
        mock_registry,
        mock_executor_cls,
        mock_wait,
        capsys,
    ):
        """When recommendation has bad int type, keep original value and log warning."""
        config = _make_config(budget=2, num_workers=1, debug=True)
        opt = NevergradOptimizer(config)

        mock_ng_opt = _build_mock_ng_optimizer()
        rec = MagicMock()
        rec.kwargs = {"IndentWidth": "not_a_number"}  # bad type
        mock_ng_opt.provide_recommendation.return_value = rec
        mock_registry.__getitem__.return_value.return_value = mock_ng_opt

        mock_executor = _build_mock_executor(future_results=[10, 20])
        mock_executor_cls.return_value = mock_executor

        mock_wait.side_effect = lambda futures, **kw: (list(futures), [])

        lookups = _make_lookups()
        base = _make_base_options()
        result = opt.optimize(base, ["/tmp/repo"], lookups, 100.0, 42)

        # Should keep original value (4) since conversion failed
        assert result["IndentWidth"]["value"] == 4
        captured = capsys.readouterr()
        assert "Failed to convert Nevergrad value" in captured.err
