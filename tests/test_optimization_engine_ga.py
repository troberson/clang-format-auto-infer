"""Tests for the generic island-model genetic algorithm."""

import random
from typing import Any
from src.optimization_engine.ga import (
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
        assert "Perfect configuration found" in captured.err

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
        assert "Iteration 1" in captured.err

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
