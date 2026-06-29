"""Generic Nevergrad optimization wrapper with no domain-specific dependencies.

Provides a generic ask/tell loop, instrumentation builder from SearchSpace,
and progress reporting. The caller supplies the fitness objective function.
"""

from __future__ import annotations

import concurrent.futures
import sys
import threading
from collections.abc import Callable
from typing import Any

import nevergrad as ng  # type: ignore[import-untyped]

from ..utils import dbg
from .types import OptimizationResult, SearchSpace


def build_instrumentation(search_space: SearchSpace) -> ng.p.Instrumentation:
    """Build Nevergrad instrumentation from a generic SearchSpace.

    Maps ParameterDef types to Nevergrad parameter types:
    - bool -> Scalar with integer casting (0/1)
    - int with possible_values -> Choice of ints
    - str with possible_values -> Choice of strings
    - fixed or no values -> excluded from instrumentation
    """
    params: dict[str, ng.p.Parameter] = {}
    for param in search_space.mutable_parameters:
        if param.param_type == "bool":
            params[param.name] = ng.p.Scalar(
                init=0.0, lower=0.0, upper=1.0
            ).set_integer_casting()
        elif param.param_type == "int" and param.possible_values:
            params[param.name] = ng.p.Choice([int(v) for v in param.possible_values])
        elif param.param_type == "str" and param.possible_values:
            params[param.name] = ng.p.Choice(list(param.possible_values))
        # else: skip (fixed or no values)
    return ng.p.Instrumentation(**params)


def _convert_param_types(
    config: dict[str, Any], search_space: SearchSpace
) -> dict[str, Any]:
    """Convert Nevergrad recommendation values to proper types."""
    result = dict(config)
    for name, value in result.items():
        param = search_space.parameters.get(name)
        if param is None:
            continue
        if param.param_type == "int":
            try:
                result[name] = int(value)
            except (ValueError, TypeError):
                pass  # keep original
        elif param.param_type == "bool":
            result[name] = bool(value)
    return result


class _ObjectiveWrapper:
    """Picklable wrapper that converts nevergrad params to a config dict.

    Must be a module-level class so ProcessPoolExecutor can pickle it.
    """

    _objective: Callable[[dict[str, Any]], float]
    _search_space: SearchSpace
    _initial_config: dict[str, Any] | None
    _tag: str

    def __init__(
        self,
        objective: Callable[[dict[str, Any]], float],
        search_space: SearchSpace,
        initial_config: dict[str, Any] | None,
        tag: str = "",
    ) -> None:
        self._objective = objective
        self._search_space = search_space
        self._initial_config = initial_config
        self._tag = tag

    def __call__(self, **ng_params: Any) -> float:
        config = dict(self._initial_config) if self._initial_config else {}
        config.update(_convert_param_types(ng_params, self._search_space))
        return self._objective(config)


def run_nevergrad_optimization(
    search_space: SearchSpace,
    objective: Callable[[dict[str, Any]], float],
    budget: int,
    num_workers: int,
    optimizer_name: str = "TwoPointsDE",
    debug: bool = False,
    initial_config: dict[str, Any] | None = None,
    convergence_threshold: int | None = None,
    tag: str = "",
) -> OptimizationResult:
    """Run a generic Nevergrad optimization loop.

    Args:
        search_space: Defines the tunable parameters and their types.
        objective: Callable that takes a config dict and returns fitness (lower is better).
        budget: Maximum number of evaluations.
        num_workers: Number of parallel workers.
        optimizer_name: Nevergrad optimizer to use.
        debug: If True, print detailed progress.
        initial_config: Base config to merge with optimized values.
        convergence_threshold: If set, stop early when no improvement occurs
            for this many consecutive evaluations. None means run full budget.
        tag: Tag used for debug output (e.g., phase name).

    Returns:
        OptimizationResult with best config and fitness.
    """
    if budget <= 0:
        base_config = initial_config or {}
        return OptimizationResult(best_config=base_config, best_fitness=float("inf"))

    # Build instrumentation
    instrumentation = build_instrumentation(search_space)

    # Create optimizer
    try:
        optimizer = ng.optimizers.registry[optimizer_name](
            parametrization=instrumentation,
            budget=budget,
            num_workers=num_workers,
        )
    except KeyError:
        available = list(ng.optimizers.registry.keys())
        print(
            f"Error: Nevergrad optimizer '{optimizer_name}' not found. Available: {available}",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:  # pragma: no cover
        print(f"Error initializing Nevergrad optimizer: {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"[{tag}] Starting Nevergrad optimization with {optimizer_name}...",
        file=sys.stderr,
    )
    print(f"[{tag}] Budget: {budget}, Workers: {num_workers}", file=sys.stderr)

    # State
    best_overall_fitness = float("inf")
    best_fitness_history: list[float] = []
    current_eval_count = 0
    interrupted = False
    # Convergence tracking
    no_improve_count = 0

    pending_futures: dict[concurrent.futures.Future[float], ng.p.Parameter] = {}

    # Use a module-level callable so ThreadPoolExecutor can invoke it.
    wrapper = _ObjectiveWrapper(objective, search_space, initial_config, tag)

    executor: concurrent.futures.ThreadPoolExecutor | None = None

    # Track evaluation numbers per candidate for debug tracing.
    eval_counter = 0
    candidate_evals: dict[ng.p.Parameter, int] = {}

    def _submit_next_evaluation() -> bool:
        nonlocal current_eval_count, eval_counter
        if current_eval_count < budget:
            assert executor is not None
            candidate = optimizer.ask()
            eval_counter += 1
            candidate_evals[candidate] = eval_counter
            future = executor.submit(wrapper, **candidate.kwargs)
            pending_futures[future] = candidate
            current_eval_count += 1
            if debug:
                dbg(
                    tag,
                    f"Submitted eval {eval_counter}. Active: {len(pending_futures)}/{num_workers}. Total submitted: {current_eval_count}/{budget}",
                )
            return True
        return False

    def _refill_queue() -> None:
        while len(pending_futures) < num_workers:
            if not _submit_next_evaluation():
                break

    try:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
        print(
            f"[{tag}] Nevergrad: Using ThreadPoolExecutor with {num_workers} workers.",
            file=sys.stderr,
        )

        _refill_queue()

        while (current_eval_count < budget or pending_futures) and not interrupted:
            done, _ = concurrent.futures.wait(
                pending_futures.keys(),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

            completed_batch = list(done)
            for f in list(pending_futures.keys()):
                if f.done() and f not in completed_batch:  # pragma: no cover
                    completed_batch.append(f)

            for completed_future in completed_batch:
                candidate = pending_futures.pop(completed_future)
                eval_num = candidate_evals.pop(candidate, len(best_fitness_history) + 1)
                thread_id = threading.current_thread().ident or 0
                try:
                    loss = completed_future.result()
                    optimizer.tell(candidate, loss)
                    best_fitness_history.append(loss)

                    if loss < best_overall_fitness:
                        best_overall_fitness = loss
                        no_improve_count = 0
                        dbg(
                            tag,
                            f"Eval {eval_num} (thread={thread_id}): New best fitness: {best_overall_fitness}",
                            summary=True,
                        )
                    else:
                        no_improve_count += 1

                    # Check convergence
                    if (
                        convergence_threshold is not None
                        and no_improve_count >= convergence_threshold
                    ):
                        print(
                            f"[{tag}] Nevergrad: Converged after {len(best_fitness_history)} evaluations (no improvement for {convergence_threshold} consecutive evaluations).",
                            file=sys.stderr,
                        )
                        interrupted = True

                    if debug:
                        dbg(
                            tag,
                            f"Eval {eval_num} (thread={thread_id}) completed. Loss: {loss}, Best: {best_overall_fitness}",
                        )
                    elif len(best_fitness_history) % 50 == 0:
                        print(
                            f"[{tag}] --- Evaluation {len(best_fitness_history)}/{budget} (Best: {best_overall_fitness}) ---",
                            file=sys.stderr,
                        )

                except KeyboardInterrupt:  # pragma: no cover
                    print(f"\n[{tag}] Ctrl-C detected. Terminating...", file=sys.stderr)
                    interrupted = True
                    break
                except Exception as e:
                    print(
                        f"[{tag}] Nevergrad: Error during evaluation: {e}",
                        file=sys.stderr,
                    )
                    optimizer.tell(candidate, float("inf"))

            if interrupted:  # pragma: no cover
                break
            _refill_queue()

        recommendation = optimizer.provide_recommendation()

    except KeyboardInterrupt:  # pragma: no cover
        print(f"\n[{tag}] Ctrl-C detected. Terminating...", file=sys.stderr)
        interrupted = True
        recommendation = optimizer.provide_recommendation()
    except Exception as e:  # pragma: no cover
        print(f"[{tag}] Error during Nevergrad optimization: {e}", file=sys.stderr)
        recommendation = optimizer.provide_recommendation()
    finally:
        if executor:
            print(f"[{tag}] Shutting down ThreadPoolExecutor...", file=sys.stderr)
            for future in pending_futures.keys():  # pragma: no cover
                _ = future.cancel()
            executor.shutdown(wait=True)
            print(f"[{tag}] ThreadPoolExecutor shut down.", file=sys.stderr)

    # Build result
    if recommendation is None:  # pyright: ignore[reportUnnecessaryComparison]
        print(
            f"[{tag}] Warning: No recommendation from Nevergrad. Returning initial config.",
            file=sys.stderr,
        )
        base_config = dict(initial_config) if initial_config else {}
        return OptimizationResult(best_config=base_config, best_fitness=float("inf"))

    best_config = dict(initial_config) if initial_config else {}
    best_config.update(_convert_param_types(recommendation.kwargs, search_space))

    print(
        f"\n[{tag}] Nevergrad optimization finished. Best fitness: {best_overall_fitness}",
        file=sys.stderr,
    )
    return OptimizationResult(
        best_config=best_config, best_fitness=best_overall_fitness
    )
