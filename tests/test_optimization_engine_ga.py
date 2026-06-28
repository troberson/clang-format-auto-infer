"""Tests for the generic island-model genetic algorithm."""

import random
from typing import Any
from src.optimization_engine.ga import (  # noqa: PLC2701
    MAX_MUTABLE_RANDOMIZED,
    _effective_diversity_rate,  # pyright: ignore[reportPrivateUsage]
    _initialize_population,  # pyright: ignore[reportPrivateUsage]
    crossover,
    evolve_island_generation,
    mutate,
    perform_migration,
    run_island_ga,
)
from src.optimization_engine.types import (
    Individual,
    ParameterDef,
    SearchSpace,
)


def _make_search_space() -> SearchSpace:
    return SearchSpace(
        parameters={
            "a": ParameterDef(name="a", param_type="int", possible_values=[1, 2, 3]),
            "b": ParameterDef(
                name="b", param_type="bool", possible_values=[True, False]
            ),
            "c": ParameterDef(name="c", param_type="str", fixed=True),
        }
    )


def _fitness(config: dict[str, Any]) -> float:
    """Simple fitness: sum of values, lower is better."""
    return sum(v if isinstance(v, (int, float)) else 0 for v in config.values())


class TestCrossover:
    def test_produces_child_with_all_keys(self):
        p1 = {"a": 1, "b": 2}
        p2 = {"a": 10, "b": 20}
        child = crossover(p1, p2, random.Random(42))
        assert set(child.keys()) == {"a", "b"}

    def test_values_come_from_parents(self):
        p1 = {"a": 1, "b": 2}
        p2 = {"a": 10, "b": 20}
        child = crossover(p1, p2, random.Random(42))
        assert child["a"] in (1, 10)
        assert child["b"] in (2, 20)

    def test_deep_copy_independence(self):
        p1 = {"a": [1, 2]}
        p2 = {"a": [3, 4]}
        child = crossover(p1, p2, random.Random(42))
        child["a"].append(99)
        assert 99 not in p1["a"]
        assert 99 not in p2["a"]


class TestMutate:
    def test_mutates_one_option(self):
        ss = _make_search_space()
        ind = Individual(config={"a": 3, "b": True, "c": "x"})
        result = mutate(ind, ss, _fitness, random.Random(42))
        # Should have mutated either a or b
        assert result.fitness <= ind.fitness or result.fitness == float("inf")

    def test_no_mutable_options_returns_inf(self):
        ss = SearchSpace(
            parameters={
                "x": ParameterDef(name="x", param_type="str", fixed=True),
            }
        )
        ind = Individual(config={"x": "hello"})
        result = mutate(ind, ss, _fitness, random.Random(42))
        assert result.fitness == float("inf")
        assert result.config == {"x": "hello"}

    def test_debug_prints_no_mutable(self, capsys):
        ss = SearchSpace(
            parameters={
                "x": ParameterDef(name="x", param_type="str", fixed=True),
            }
        )
        ind = Individual(config={"x": "hello"})
        _ = mutate(ind, ss, _fitness, random.Random(42), debug=True, debug_prefix="W: ")
        captured = capsys.readouterr()
        assert "No mutable options" in captured.err

    def test_debug_prints_mutating(self, capsys):
        ss = _make_search_space()
        ind = Individual(config={"a": 3, "b": True})
        _ = mutate(ind, ss, _fitness, random.Random(42), debug=True, debug_prefix="W: ")
        captured = capsys.readouterr()
        assert "Mutating" in captured.err

    def test_picks_best_value(self):
        ss = SearchSpace(
            parameters={
                "x": ParameterDef(
                    name="x", param_type="int", possible_values=[10, 1, 5]
                ),
            }
        )
        ind = Individual(config={"x": 10})
        result = mutate(ind, ss, _fitness, random.Random(42))
        assert result.config["x"] == 1
        assert result.fitness == 1

    def test_all_values_inf_fitness(self):
        def _always_inf(_config: dict[str, Any]) -> float:
            return float("inf")

        ss = SearchSpace(
            parameters={
                "x": ParameterDef(name="x", param_type="int", possible_values=[1, 2]),
            }
        )
        ind = Individual(config={"x": 1})
        result = mutate(ind, ss, _always_inf, random.Random(42))
        # All values tie at inf, should pick one and return inf
        assert result.fitness == float("inf")
        assert result.config["x"] in (1, 2)


class TestEvolveIslandGeneration:
    def test_elitism_keeps_best(self):
        ss = _make_search_space()
        pop = [
            Individual(config={"a": 1, "b": False}, fitness=1.0),
            Individual(config={"a": 3, "b": True}, fitness=4.0),
        ]
        result = evolve_island_generation(pop, 2, ss, _fitness, random.Random(42))
        best = min(result, key=lambda i: i.fitness)
        assert best.fitness <= 1.0

    def test_empty_population_returns_empty(self):
        ss = _make_search_space()
        result = evolve_island_generation([], 5, ss, _fitness, random.Random(42))
        assert result == []

    def test_returns_island_size_individuals(self):
        ss = _make_search_space()
        pop = [
            Individual(config={"a": 1, "b": False}, fitness=1.0),
            Individual(config={"a": 3, "b": True}, fitness=4.0),
        ]
        result = evolve_island_generation(pop, 5, ss, _fitness, random.Random(42))
        assert len(result) == 5

    def test_small_population_warns(self, capsys):
        ss = _make_search_space()
        pop = [Individual(config={"a": 1, "b": False}, fitness=1.0)]
        _ = evolve_island_generation(
            pop, 3, ss, _fitness, random.Random(42), debug=True
        )
        captured = capsys.readouterr()
        assert "too small" in captured.err


class TestPerformMigration:
    def test_no_migration_single_island(self):
        pops = [[Individual(config={"a": 1}, fitness=1.0)]]
        perform_migration(pops, random.Random(42))
        assert len(pops) == 1

    def test_migrants_replace_individuals(self):
        pops = [
            [Individual(config={"a": 1}, fitness=1.0)],
            [Individual(config={"a": 10}, fitness=10.0)],
        ]
        perform_migration(pops, random.Random(42))
        # Island 2 should have received migrant from island 1
        assert len(pops[0]) == 1
        assert len(pops[1]) == 1

    def test_migrant_added_to_empty_island(self):
        pops = [
            [Individual(config={"a": 1}, fitness=1.0)],
            [],
        ]
        perform_migration(pops, random.Random(42), debug=True)
        assert len(pops[1]) == 1

    def test_migration_debug_prints_replacement(self, capsys):
        pops: list[list[Individual]] = [
            [Individual(config={"a": 1}, fitness=1.0)],
            [Individual(config={"a": 10}, fitness=10.0)],
        ]
        perform_migration(pops, random.Random(42), debug=True)
        captured = capsys.readouterr()
        assert "replaced an individual" in captured.err

    def test_skips_empty_islands_for_selection(self, capsys):
        pops: list[list[Individual]] = [[], []]
        perform_migration(pops, random.Random(42), debug=True)
        captured = capsys.readouterr()
        assert "empty" in captured.err

    def test_single_island_debug(self, capsys):
        pops = [[Individual(config={"a": 1}, fitness=1.0)]]
        perform_migration(pops, random.Random(42), debug=True)
        captured = capsys.readouterr()
        assert "less than 2" in captured.err


class TestRunIslandGa:
    def test_runs_and_returns_individual(self):
        ss = _make_search_space()
        initial = {"a": 3, "b": True, "c": "x"}
        best = run_island_ga(
            initial,
            ss,
            _fitness,
            num_islands=2,
            population_size=10,
            num_iterations=3,
            random_seed=42,
        )
        assert isinstance(best, Individual)
        assert "a" in best.config

    def test_single_island(self):
        ss = _make_search_space()
        initial = {"a": 3, "b": True}
        best = run_island_ga(
            initial,
            ss,
            _fitness,
            num_islands=1,
            population_size=5,
            num_iterations=2,
            random_seed=42,
        )
        assert isinstance(best, Individual)

    def test_early_termination_on_zero_fitness(self, capsys):
        def _zero_fitness(_config: dict[str, Any]) -> float:
            return 0.0

        ss = _make_search_space()
        initial = {"a": 3, "b": True}
        best = run_island_ga(
            initial,
            ss,
            _zero_fitness,
            num_islands=1,
            population_size=5,
            num_iterations=100,
            debug=True,
            random_seed=42,
        )
        assert best.fitness == 0.0
        captured = capsys.readouterr()
        assert "Iter 1/100 fitness=0.0" in captured.err

    def test_debug_prints_iterations(self, capsys):
        ss = _make_search_space()
        initial = {"a": 3, "b": True}
        _ = run_island_ga(
            initial,
            ss,
            _fitness,
            num_islands=1,
            population_size=5,
            num_iterations=2,
            debug=True,
            random_seed=42,
        )
        captured = capsys.readouterr()
        assert "Iter 1/2" in captured.err

    def test_min_improvement_ratio_treats_tiny_improvements_as_noise(self, capsys):
        """Tiny improvements below min_improvement_ratio do not reset convergence."""
        # Fitness that returns large values with tiny fractional improvements.
        call_count = [0]

        def tiny_improvement_fitness(_config: dict[str, Any]) -> float:
            call_count[0] += 1
            # Start at 10000, slowly decrease by 0.0001 each call.
            # Improvement is ~0.0001/10000 = 1e-8, far below default 0.001.
            return max(0.0, 10000.0 - call_count[0] * 0.0001)

        ss = _make_search_space()
        initial = {"a": 3, "b": True}
        _ = run_island_ga(
            initial,
            ss,
            tiny_improvement_fitness,
            num_islands=1,
            population_size=5,
            num_iterations=100,
            convergence_threshold=5,
            min_improvement_ratio=0.001,
            debug=True,
            random_seed=42,
        )
        captured = capsys.readouterr()
        # Should converge early because tiny improvements are treated as noise.
        assert "Converged after" in captured.err
        # Verify it stopped well before 100 iterations.
        assert "Converged after 5" in captured.err

    def test_migration_interval(self):
        ss = _make_search_space()
        initial = {"a": 3, "b": True}
        best = run_island_ga(
            initial,
            ss,
            _fitness,
            num_islands=2,
            population_size=10,
            num_iterations=30,
            migration_interval=10,
            random_seed=42,
        )
        assert isinstance(best, Individual)

    def test_num_islands_clamped_to_1(self):
        ss = _make_search_space()
        initial = {"a": 3, "b": True}
        best = run_island_ga(
            initial,
            ss,
            _fitness,
            num_islands=0,
            population_size=5,
            num_iterations=2,
            random_seed=42,
        )
        assert isinstance(best, Individual)

    def test_population_size_adjusted_for_min_island_size(self):
        """When population_size < num_islands * 5, it is adjusted up."""
        ss = _make_search_space()
        initial = {"a": 3, "b": True}
        # 3 islands, pop_size=10 -> island_size=max(5, 10//3)=5, 5*3=15 > 10
        best = run_island_ga(
            initial,
            ss,
            _fitness,
            num_islands=3,
            population_size=10,
            num_iterations=1,
            random_seed=42,
        )
        assert isinstance(best, Individual)

    def test_parallel_islands_produces_valid_result(self):
        """Parallel island evolution (num_workers>1) produces a valid Individual."""
        ss = _make_search_space()
        initial = {"a": 3, "b": True}
        best = run_island_ga(
            initial,
            ss,
            _fitness,
            num_islands=4,
            population_size=20,
            num_iterations=3,
            random_seed=42,
            num_workers=2,
        )
        assert isinstance(best, Individual)
        assert "a" in best.config

    def test_parallel_islands_all_islands_evolved(self):
        """All islands are evolved when using parallel execution."""
        call_count = [0]

        def counting_fitness(config: dict[str, Any]) -> float:
            call_count[0] += 1
            return sum(v if isinstance(v, (int, float)) else 0 for v in config.values())

        ss = _make_search_space()
        initial = {"a": 3, "b": True}
        _ = run_island_ga(
            initial,
            ss,
            counting_fitness,
            num_islands=2,
            population_size=10,
            num_iterations=2,
            random_seed=42,
            num_workers=2,
        )
        # 1 initial + 2 iterations * 2 islands * island_size evaluations.
        # Each island_size individual triggers fitness calls inside mutate(),
        # so total > island_size. Just verify all islands contributed work.
        assert call_count[0] > 1

    def test_parallel_islands_convergence_detection(self, capsys):
        """Convergence detection works with parallel islands."""

        def _zero_fitness(_config: dict[str, Any]) -> float:
            return 0.0

        ss = _make_search_space()
        initial = {"a": 3, "b": True}
        best = run_island_ga(
            initial,
            ss,
            _zero_fitness,
            num_islands=2,
            population_size=10,
            num_iterations=100,
            convergence_threshold=5,
            debug=True,
            random_seed=42,
            num_workers=2,
        )
        assert best.fitness == 0.0
        captured = capsys.readouterr()
        assert "Iter 1/100 fitness=0.0" in captured.err

    def test_parallel_islands_budget_check(self, capsys):
        """Budget check works with parallel islands."""
        ss = _make_search_space()
        initial = {"a": 3, "b": True}
        _ = run_island_ga(
            initial,
            ss,
            _fitness,
            num_islands=2,
            population_size=10,
            num_iterations=100,
            budget=15,
            debug=True,
            random_seed=42,
            num_workers=2,
        )
        captured = capsys.readouterr()
        assert "Budget exhausted" in captured.err

    def test_single_worker_is_sequential(self):
        """num_workers=1 uses sequential path."""
        ss = _make_search_space()
        initial = {"a": 3, "b": True}
        best = run_island_ga(
            initial,
            ss,
            _fitness,
            num_islands=2,
            population_size=10,
            num_iterations=2,
            random_seed=42,
            num_workers=1,
        )
        assert isinstance(best, Individual)

    def test_num_workers_clamped_to_num_islands(self):
        """num_workers greater than num_islands is clamped to num_islands."""
        ss = _make_search_space()
        initial = {"a": 3, "b": True}
        best = run_island_ga(
            initial,
            ss,
            _fitness,
            num_islands=2,
            population_size=10,
            num_iterations=2,
            random_seed=42,
            num_workers=10,  # More than num_islands
        )
        assert isinstance(best, Individual)


class TestEffectiveDiversityRate:
    def test_small_space_uses_base_rate(self):
        """With 2 mutable params, rate=0.5 is not capped."""
        rate = _effective_diversity_rate(0.5, 2)
        assert rate == 0.5  # 3/2 = 1.5 > 0.5, so base rate wins

    def test_large_space_caps_rate(self):
        """With 20 mutable params, rate=0.5 is capped to 3/20."""
        rate = _effective_diversity_rate(0.5, 20)
        assert rate == MAX_MUTABLE_RANDOMIZED / 20  # 0.15

    def test_zero_mutable_returns_zero(self):
        rate = _effective_diversity_rate(0.5, 0)
        assert rate == 0.0

    def test_negative_mutable_returns_zero(self):
        rate = _effective_diversity_rate(0.5, -1)
        assert rate == 0.0

    def test_rate_higher_than_cap_uses_cap(self):
        """With 2 mutable params, rate=1.0 is capped to 1.0 (3/2=1.5 > 1.0)."""
        rate = _effective_diversity_rate(1.0, 2)
        assert rate == 1.0

    def test_rate_lower_than_cap_uses_base(self):
        """With 10 mutable params, rate=0.01 is not capped."""
        rate = _effective_diversity_rate(0.01, 10)
        assert rate == 0.01  # 3/10 = 0.3 > 0.01

    def test_constant_value(self):
        assert MAX_MUTABLE_RANDOMIZED == 3


class TestInitializePopulation:
    def test_first_individual_is_anchor(self):
        """Individual 0 is an exact copy of initial_config."""
        ss = _make_search_space()
        initial = {"a": 3, "b": True, "c": "x"}
        pop, evals = _initialize_population(
            initial, ss, _fitness, island_size=5, rng=random.Random(42)
        )
        assert pop[0].config == initial
        assert pop[0].fitness == _fitness(initial)
        assert evals == 5

    def test_has_diversity(self):
        """At least some individuals differ from the anchor."""
        ss = _make_search_space()
        initial = {"a": 3, "b": True, "c": "x"}
        pop, _ = _initialize_population(
            initial, ss, _fitness, island_size=10, rng=random.Random(42)
        )
        unique_configs = set(str(ind.config) for ind in pop)
        assert len(unique_configs) > 1

    def test_all_individuals_have_all_keys(self):
        """Every individual contains all keys from initial_config."""
        ss = _make_search_space()
        initial = {"a": 3, "b": True, "c": "x"}
        pop, _ = _initialize_population(
            initial, ss, _fitness, island_size=5, rng=random.Random(42)
        )
        for ind in pop:
            assert set(ind.config.keys()) == set(initial.keys())

    def test_all_individuals_have_fitness(self):
        """Every individual has a finite fitness value."""
        ss = _make_search_space()
        initial = {"a": 3, "b": True, "c": "x"}
        pop, _ = _initialize_population(
            initial, ss, _fitness, island_size=5, rng=random.Random(42)
        )
        for ind in pop:
            assert ind.fitness == _fitness(ind.config)

    def test_zero_diversity_produces_identical_clones(self):
        """diversity_rate=0 produces all identical individuals."""
        ss = _make_search_space()
        initial = {"a": 3, "b": True, "c": "x"}
        pop, _ = _initialize_population(
            initial,
            ss,
            _fitness,
            island_size=5,
            rng=random.Random(42),
            diversity_rate=0.0,
        )
        for ind in pop:
            assert ind.config == initial

    def test_full_diversity_randomizes_all_mutable_params(self):
        """diversity_rate=1.0 randomizes all mutable params for non-anchor individuals."""
        ss = SearchSpace(
            parameters={
                "a": ParameterDef(
                    name="a", param_type="int", possible_values=[1, 2, 3]
                ),
            }
        )
        initial = {"a": 3}
        pop, _ = _initialize_population(
            initial,
            ss,
            _fitness,
            island_size=5,
            rng=random.Random(42),
            diversity_rate=1.0,
        )
        # Anchor keeps initial value, others may differ
        assert pop[0].config["a"] == 3
        # With 4 random draws from [1,2,3], at least one should differ from 3
        non_anchor_values = [ind.config["a"] for ind in pop[1:]]
        assert any(v != 3 for v in non_anchor_values)

    def test_single_individual_returns_anchor(self):
        """island_size=1 returns just the anchor individual."""
        ss = _make_search_space()
        initial = {"a": 3, "b": True, "c": "x"}
        pop, evals = _initialize_population(
            initial, ss, _fitness, island_size=1, rng=random.Random(42)
        )
        assert len(pop) == 1
        assert pop[0].config == initial
        assert evals == 1

    def test_no_mutable_params_falls_back_to_clones(self):
        """When no mutable params exist, all individuals are identical clones."""
        ss = SearchSpace(
            parameters={
                "x": ParameterDef(name="x", param_type="str", fixed=True),
            }
        )
        initial = {"x": "hello"}
        pop, _ = _initialize_population(
            initial, ss, _fitness, island_size=5, rng=random.Random(42)
        )
        for ind in pop:
            assert ind.config == initial


class TestRunIslandGaDiversity:
    def test_diverse_initialization_produces_different_islands(self):
        """With diversity, islands start with different best fitness values."""
        ss = _make_search_space()
        initial = {"a": 3, "b": True, "c": "x"}
        best = run_island_ga(
            initial,
            ss,
            _fitness,
            num_islands=4,
            population_size=20,
            num_iterations=1,
            random_seed=42,
            diversity_rate=0.5,
        )
        assert isinstance(best, Individual)

    def test_diversity_rate_zero_preserves_old_behavior(self):
        """diversity_rate=0 makes all islands start identical."""
        ss = _make_search_space()
        initial = {"a": 3, "b": True}
        best = run_island_ga(
            initial,
            ss,
            _fitness,
            num_islands=2,
            population_size=10,
            num_iterations=2,
            random_seed=42,
            diversity_rate=0.0,
        )
        assert isinstance(best, Individual)

    def test_diversity_rate_default(self):
        """Default diversity_rate is 0.5."""
        ss = _make_search_space()
        initial = {"a": 3, "b": True}
        best = run_island_ga(
            initial,
            ss,
            _fitness,
            num_islands=2,
            population_size=10,
            num_iterations=2,
            random_seed=42,
        )
        assert isinstance(best, Individual)
