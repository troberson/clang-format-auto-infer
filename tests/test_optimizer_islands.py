"""Tests for optimizer island evolution, migration, and polish helpers."""

from unittest.mock import patch, MagicMock


from src.optimizer import (
    _evolve_island_generation_task,  # pyright: ignore[reportPrivateUsage]
    _perform_migration,  # pyright: ignore[reportPrivateUsage]
    _polish_coordinate_descent,  # pyright: ignore[reportPrivateUsage]
    _polish_single_option_task,  # pyright: ignore[reportPrivateUsage]
    _island_evolution_task_wrapper,  # pyright: ignore[reportPrivateUsage]
    GeneticAlgorithmOptimizer,
)
from src.clang_format_parser import save_checkpoint
from src.data_classes import (
    GeneticOptimizationConfig,
    IslandEvolutionArgs,
    GeneticAlgorithmLookups,
    WorkerContext,
)


def _make_lookups(json_options=None, forced_options=None):
    return GeneticAlgorithmLookups(
        json_options_lookup=json_options or {},
        forced_options_lookup=forced_options or {},
    )


def _make_island_args(population, lookups, island_index=0, debug=False):
    return IslandEvolutionArgs(
        population=population,
        island_population_size=len(population),
        island_index=island_index,
        lookups=lookups,
        debug=debug,
        file_sample_percentage=100.0,
        random_seed=42,
    )


def _make_worker_context(repo_path="/tmp/fake"):
    return WorkerContext(repo_path=repo_path, process_id=1)


# ---------------------------------------------------------------------------
# _evolve_island_generation_task
# ---------------------------------------------------------------------------


class TestEvolveIslandGenerationTask:
    @patch("src.optimizer.mutate")
    def test_elitism_keeps_best_individual(self, mock_mutate):
        """The best individual from the current population is carried forward."""
        mock_mutate.return_value = ({"UseTab": {"type": "str", "value": "Always"}}, 10)

        population = [
            {"config": {"UseTab": {"type": "str", "value": "Never"}}, "fitness": 5},
            {
                "config": {"UseTab": {"type": "str", "value": "ForIndentation"}},
                "fitness": 20,
            },
        ]
        lookups = _make_lookups()
        args = _make_island_args(population, lookups)
        ctx = _make_worker_context()

        new_pop, best = _evolve_island_generation_task(args, ctx)

        # Best (fitness 5) should be in new population
        assert any(ind["fitness"] == 5 for ind in new_pop)
        assert best["fitness"] == 5

    @patch("src.optimizer.mutate")
    @patch("src.optimizer.crossover")
    def test_generates_new_individuals_via_crossover_and_mutation(
        self, mock_crossover, mock_mutate
    ):
        mock_crossover.return_value = {"UseTab": {"type": "str", "value": "Never"}}
        mock_mutate.return_value = (
            {"UseTab": {"type": "str", "value": "Always"}},
            7,
        )

        population = [
            {"config": {"UseTab": {"type": "str", "value": "Never"}}, "fitness": 10},
            {
                "config": {"UseTab": {"type": "str", "value": "ForIndentation"}},
                "fitness": 15,
            },
        ]
        lookups = _make_lookups()
        args = _make_island_args(population, lookups)
        ctx = _make_worker_context()

        _ = _evolve_island_generation_task(args, ctx)

        # 1 elitist + 1 new = 2 total, so 1 crossover + 1 mutate call
        assert mock_crossover.call_count == 1
        assert mock_mutate.call_count == 1

    def test_empty_population_returns_empty(self):
        lookups = _make_lookups()
        args = _make_island_args([], lookups)
        args.island_population_size = 0
        ctx = _make_worker_context()

        new_pop, best = _evolve_island_generation_task(args, ctx)

        assert new_pop == []
        assert best["fitness"] == float("inf")

    @patch("src.optimizer.mutate")
    def test_selects_top_n_by_fitness(self, mock_mutate):
        """New population is sorted by fitness and trimmed to island_population_size."""
        mock_mutate.return_value = ({"UseTab": {"type": "str", "value": "Always"}}, 100)

        population = [
            {"config": {"UseTab": {"type": "str", "value": "Never"}}, "fitness": 5},
            {
                "config": {"UseTab": {"type": "str", "value": "ForIndentation"}},
                "fitness": 20,
            },
        ]
        lookups = _make_lookups()
        args = _make_island_args(population, lookups)
        ctx = _make_worker_context()

        new_pop, _best = _evolve_island_generation_task(args, ctx)

        # Should be sorted ascending by fitness
        assert len(new_pop) == 2
        assert new_pop[0]["fitness"] <= new_pop[1]["fitness"]


# ---------------------------------------------------------------------------
# _perform_migration
# ---------------------------------------------------------------------------


class TestPerformMigration:
    def test_no_migration_with_single_island(self):
        populations = [
            [
                {"config": {"A": 1}, "fitness": 5},
                {"config": {"A": 2}, "fitness": 10},
            ]
        ]
        _perform_migration(populations, debug=False)
        # No change expected
        assert len(populations[0]) == 2

    def test_migrants_replace_individuals_in_other_islands(self):
        populations = [
            [
                {"config": {"A": 1}, "fitness": 5},
                {"config": {"A": 2}, "fitness": 10},
            ],
            [
                {"config": {"B": 1}, "fitness": 8},
                {"config": {"B": 2}, "fitness": 12},
            ],
        ]
        original_len_0 = len(populations[0])
        original_len_1 = len(populations[1])
        # Island 0 -> target 1, Island 1 -> target 0
        with patch("src.optimizer.random.randrange", side_effect=[1, 0]):
            with patch(
                "src.optimizer.random.choice",
                side_effect=[populations[1][0], populations[0][0]],
            ):
                _perform_migration(populations, debug=False)

        # Both islands should still have same size
        assert len(populations[0]) == original_len_0
        assert len(populations[1]) == original_len_1

    def test_migrant_added_to_empty_island(self):
        populations = [
            [
                {"config": {"A": 1}, "fitness": 5},
            ],
            [],
        ]
        with patch("src.optimizer.random.randrange", return_value=1):
            _perform_migration(populations, debug=False)

        assert len(populations[1]) == 1
        assert populations[1][0]["fitness"] == 5

    def test_skips_empty_islands_for_migrant_selection(self):
        populations = [
            [],
            [
                {"config": {"B": 1}, "fitness": 8},
            ],
        ]
        with patch("src.optimizer.random.randrange", return_value=0):
            _perform_migration(populations, debug=False)

        # Island 0 was empty, so island 1's migrant should have migrated to island 0
        assert len(populations[0]) == 1
        assert populations[0][0]["fitness"] == 8

    def test_migration_preserves_population_sizes(self):
        populations = [
            [{"config": {"A": i}, "fitness": i} for i in range(5)],
            [{"config": {"B": i}, "fitness": i + 10} for i in range(5)],
        ]
        with patch("src.optimizer.random.randrange", side_effect=[1, 0]):
            with patch(
                "src.optimizer.random.choice",
                side_effect=[populations[1][0], populations[0][0]],
            ):
                _perform_migration(populations, debug=False)

        assert len(populations[0]) == 5
        assert len(populations[1]) == 5


# ---------------------------------------------------------------------------
# _polish_coordinate_descent
# ---------------------------------------------------------------------------


class TestPolishCoordinateDescent:
    @patch("src.optimizer.optimize_option_with_values")
    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer.generate_clang_format_config")
    def test_improves_fitness_when_possible(self, mock_gen, mock_fitness, mock_opt):
        mock_gen.return_value = "config"
        mock_fitness.return_value = 100  # baseline
        mock_opt.return_value = 50  # improvement

        config = {
            "UseTab": {"type": "str", "value": "Never"},
        }
        lookups = _make_lookups(
            json_options={
                "UseTab": {"type": "str", "possible_values": ["Never", "Always"]}
            }
        )
        _best_config, final_fitness = _polish_coordinate_descent(
            config,
            "/tmp/fake",
            lookups,
            debug=False,
            file_sample_percentage=100.0,
            random_seed=42,
            max_passes=2,
        )

        assert final_fitness == 50

    @patch("src.optimizer.optimize_option_with_values")
    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer.generate_clang_format_config")
    def test_stops_when_no_improvements(self, mock_gen, mock_fitness, mock_opt):
        mock_gen.return_value = "config"
        mock_fitness.return_value = 10  # already good
        mock_opt.return_value = 10  # no improvement

        config = {
            "UseTab": {"type": "str", "value": "Never"},
        }
        lookups = _make_lookups(
            json_options={
                "UseTab": {"type": "str", "possible_values": ["Never", "Always"]}
            }
        )
        _best_config, final_fitness = _polish_coordinate_descent(
            config,
            "/tmp/fake",
            lookups,
            debug=False,
            file_sample_percentage=100.0,
            random_seed=42,
            max_passes=3,
        )

        # Should converge after 1 pass
        assert final_fitness == 10

    def test_no_mutable_options_returns_original(self):
        config = {
            "UseTab": {"type": "str", "value": "Never"},
        }
        lookups = _make_lookups(
            json_options={},
            forced_options={"UseTab": "Never"},
        )
        best_config, _final_fitness = _polish_coordinate_descent(
            config,
            "/tmp/fake",
            lookups,
            debug=False,
            file_sample_percentage=100.0,
            random_seed=42,
            max_passes=1,
        )

        assert best_config["UseTab"]["value"] == "Never"

    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer.generate_clang_format_config")
    @patch("src.optimizer.optimize_option_with_values")
    def test_max_passes_reached(self, mock_opt, mock_gen, mock_run, capsys):
        mock_gen.return_value = "config\n"
        mock_run.return_value = 100
        fitness_values = [90, 80, 70, 60, 50, 40]
        mock_opt.side_effect = fitness_values

        lookups = _make_lookups(
            json_options={"IndentWidth": {"possible_values": ["2", "4"]}}
        )
        config = {
            "UseTab": {"type": "str", "value": "Never"},
            "IndentWidth": {"type": "int", "value": 4},
            "BreakBeforeBraces": {"type": "bool", "value": True},
        }
        _ = _polish_coordinate_descent(
            config, "/tmp/repo", lookups, False, 100.0, 42, max_passes=2
        )
        captured = capsys.readouterr()
        assert "reached max passes" in captured.err

    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer.generate_clang_format_config")
    @patch("src.optimizer.optimize_option_with_values")
    def test_bool_possible_values_in_polish(self, mock_opt, mock_gen, mock_run):
        mock_gen.return_value = "config\n"
        mock_run.return_value = 100
        mock_opt.return_value = 90

        lookups = _make_lookups()
        config = {
            "UseTab": {"type": "str", "value": "Never"},
            "IndentWidth": {"type": "int", "value": 4},
            "BreakBeforeBraces": {"type": "bool", "value": True},
            "AlignConsecutiveAssignments": {"type": "bool", "value": False},
        }
        _ = _polish_coordinate_descent(
            config, "/tmp/repo", lookups, False, 100.0, 42, max_passes=1
        )
        assert mock_opt.call_count >= 2

    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer.generate_clang_format_config")
    @patch("src.optimizer.optimize_option_with_values")
    def test_polish_skips_option_no_values(self, mock_opt, mock_gen, mock_run):
        mock_gen.return_value = "config\n"
        mock_run.return_value = 100
        mock_opt.return_value = 90

        lookups = _make_lookups(json_options={"IndentWidth": {"possible_values": []}})
        config = {
            "IndentWidth": {"type": "int", "value": 4},
            "BreakBeforeBraces": {"type": "bool", "value": True},
        }
        _ = _polish_coordinate_descent(
            config, "/tmp/repo", lookups, False, 100.0, 42, max_passes=1
        )
        for call_args in mock_opt.call_args_list:
            assert call_args[0][1] != "IndentWidth"


# ---------------------------------------------------------------------------
# _evolve_island_generation_task edge cases
# ---------------------------------------------------------------------------


class TestEvolveIslandGenerationTaskEdgeCases:
    @patch("src.optimizer.mutate")
    @patch("src.optimizer.crossover")
    def test_population_too_small_for_crossover(self, _mock_cross, mock_mutate, capsys):
        mock_mutate.return_value = ({"config": {}}, 10)
        island_args = IslandEvolutionArgs(
            population=[{"config": {"a": 1}, "fitness": 5}],
            island_population_size=3,
            island_index=0,
            lookups=_make_lookups(),
            debug=True,
            file_sample_percentage=100.0,
            random_seed=42,
        )

        _ = _evolve_island_generation_task(island_args, _make_worker_context())
        captured = capsys.readouterr()
        assert "population too small" in captured.err

    @patch("src.optimizer.mutate")
    @patch("src.optimizer.crossover")
    def test_num_to_generate_clamped(self, mock_crossover, mock_mutate):
        mock_mutate.return_value = ({"config": {}}, 100)
        mock_crossover.return_value = {"config": {}}

        population = [
            {"config": {"a": {"type": "str", "value": "1"}}, "fitness": 10}
            for _ in range(10)
        ]
        island_args = IslandEvolutionArgs(
            population=population,
            island_population_size=5,
            island_index=0,
            lookups=_make_lookups(),
            debug=False,
            file_sample_percentage=100.0,
            random_seed=42,
        )
        new_pop, _best = _evolve_island_generation_task(
            island_args, _make_worker_context()
        )
        assert len(new_pop) == 5


# ---------------------------------------------------------------------------
# _perform_migration edge cases
# ---------------------------------------------------------------------------


class TestPerformMigrationEdgeCases:
    def test_skips_migration_less_than_two_islands(self, capsys):
        _perform_migration([[]], debug=True)
        captured = capsys.readouterr()
        assert "Skipping migration" in captured.err

    def test_debug_prints_performing_migration(self, capsys):
        populations = [
            [{"config": {}, "fitness": 10}],
            [{"config": {}, "fitness": 20}],
        ]
        _perform_migration(populations, debug=True)
        captured = capsys.readouterr()
        assert "Performing migration" in captured.err

    def test_empty_island_warning(self, capsys):
        populations = [
            [],
            [{"config": {}, "fitness": 20}],
        ]
        _perform_migration(populations, debug=True)
        captured = capsys.readouterr()
        assert "empty" in captured.err

    def test_migrant_added_to_empty_target(self, capsys):
        populations = [
            [{"config": {}, "fitness": 10}],
            [],
        ]
        _perform_migration(populations, debug=True)
        captured = capsys.readouterr()
        assert "added to empty" in captured.err

    def test_migrant_replaces_individual(self, capsys):
        populations = [
            [{"config": {}, "fitness": 10}],
            [{"config": {}, "fitness": 20}],
        ]
        _perform_migration(populations, debug=True)
        captured = capsys.readouterr()
        assert "replaced an individual" in captured.err


# ---------------------------------------------------------------------------
# _island_evolution_task_wrapper
# ---------------------------------------------------------------------------


class TestIslandEvolutionTaskWrapper:
    @patch("src.optimizer._evolve_island_generation_task")
    @patch("src.optimizer.multiprocessing.current_process")
    def test_wrapper_calls_evolve_with_worker_context(
        self, mock_current_process, mock_evolve
    ):
        mock_current_process.return_value._identity = [3]
        mock_evolve.return_value = (
            [{"config": {}, "fitness": 50}],
            {"config": {}, "fitness": 50},
        )

        island_args = IslandEvolutionArgs(
            population=[],
            island_population_size=5,
            island_index=0,
            lookups=_make_lookups(
                json_options={"IndentWidth": {"possible_values": ["2"]}}
            ),
            debug=False,
            file_sample_percentage=100.0,
            random_seed=42,
        )
        args_tuple = (
            island_args,
            ["/tmp/repo1", "/tmp/repo2", "/tmp/repo3"],
            _make_lookups(json_options={"IndentWidth": {"possible_values": ["2"]}}),
            True,
            50.0,
            42,
        )
        _ = _island_evolution_task_wrapper(args_tuple)

        mock_evolve.assert_called_once()
        call_args = mock_evolve.call_args[0]
        evolved_island_args, worker_context = call_args
        assert worker_context.process_id == 3
        assert worker_context.repo_path == "/tmp/repo3"
        assert evolved_island_args.debug is True
        assert evolved_island_args.file_sample_percentage == 50.0
        assert evolved_island_args.random_seed == 42 + 3  # base_seed + process_id
        assert evolved_island_args.lookups is not None


# ---------------------------------------------------------------------------
# GeneticAlgorithmOptimizer.optimize() — mock multiprocessing.Pool
# ---------------------------------------------------------------------------


class TestGeneticAlgorithmOptimizerOptimize:
    def _make_config(
        self,
        num_iterations: int = 2,
        total_population_size: int = 10,
        num_islands: int = 2,
        debug: bool = False,
        plot_fitness: bool = False,
        polish_passes: int = 0,
    ) -> GeneticOptimizationConfig:
        return GeneticOptimizationConfig(
            num_iterations=num_iterations,
            total_population_size=total_population_size,
            num_islands=num_islands,
            debug=debug,
            plot_fitness=plot_fitness,
            polish_passes=polish_passes,
        )

    def _make_base_config(self):
        return {
            "UseTab": {"type": "str", "value": "Never"},
            "IndentWidth": {"type": "int", "value": 4},
            "BreakBeforeBraces": {"type": "bool", "value": True},
            "AlignConsecutiveAssignments": {"type": "bool", "value": False},
        }

    @patch("src.optimizer.run_clang_format_and_count_changes")
    def test_initialize_populations_structure(self, mock_run):
        mock_run.return_value = 100

        lookups = _make_lookups()
        base = self._make_base_config()
        opt = GeneticAlgorithmOptimizer(self._make_config(num_islands=2))

        (
            populations,
            best_overall,
            fitness_history,
            island_pop_size,
            num_islands,
        ) = opt._initialize_populations(  # pyright: ignore[reportPrivateUsage]
            base,
            ["/tmp/repo"],
            lookups,
            2,
            10,
            False,
            100.0,
            42,
        )

        assert num_islands == 2
        assert island_pop_size == 5
        assert len(populations) == 2
        assert len(populations[0]) == 5
        assert len(populations[1]) == 5
        assert best_overall["fitness"] == 100
        assert len(fitness_history) == 2
        assert fitness_history[0] == [100]
        assert fitness_history[1] == [100]

    def test_initialize_populations_no_repo_paths(self):
        lookups = _make_lookups()
        base = self._make_base_config()
        opt = GeneticAlgorithmOptimizer(self._make_config(num_islands=1))

        populations, best, history, pop_size, num = opt._initialize_populations(  # pyright: ignore[reportPrivateUsage]
            base,
            [],
            lookups,
            1,
            5,
            False,
            100.0,
            42,
        )

        assert populations == []
        assert best == {}
        assert history == []
        assert pop_size == 0
        assert num == 1

    @patch("src.optimizer.multiprocessing.Pool")
    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer._island_evolution_task_wrapper")
    def test_runs_evolution_loop(self, mock_wrapper, mock_run, mock_pool_cls, capsys):  # pyright: ignore[reportUnusedParameter]
        mock_run.return_value = 100
        mock_wrapper.return_value = (
            0,
            [{"config": {}, "fitness": 90}],
            {"config": {}, "fitness": 90},
        )
        mock_pool = MagicMock()
        mock_pool.imap_unordered.return_value = iter(
            [
                (
                    0,
                    [{"config": {}, "fitness": 90}],
                    {"config": {}, "fitness": 90},
                ),
                (
                    1,
                    [{"config": {}, "fitness": 95}],
                    {"config": {}, "fitness": 95},
                ),
            ]
        )
        mock_pool_cls.return_value = mock_pool

        lookups = _make_lookups()
        base = self._make_base_config()
        opt = GeneticAlgorithmOptimizer(self._make_config())
        _ = opt.optimize(base, ["/tmp/repo"], lookups, 100.0, 42)

        mock_pool.imap_unordered.assert_called()
        mock_pool.close.assert_called()
        mock_pool.join.assert_called()

    @patch("src.optimizer.multiprocessing.Pool")
    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer._island_evolution_task_wrapper")
    def test_early_termination_on_fitness_zero(
        self, _mock_wrapper, mock_run, mock_pool_cls, capsys
    ):
        mock_run.return_value = 100
        mock_pool = MagicMock()
        mock_pool.imap_unordered.return_value = iter(
            [
                (
                    0,
                    [{"config": {}, "fitness": 0}],
                    {"config": {}, "fitness": 0},
                ),
                (
                    1,
                    [{"config": {}, "fitness": 5}],
                    {"config": {}, "fitness": 5},
                ),
            ]
        )
        mock_pool_cls.return_value = mock_pool

        lookups = _make_lookups()
        base = self._make_base_config()
        opt = GeneticAlgorithmOptimizer(self._make_config(num_iterations=10))
        _ = opt.optimize(base, ["/tmp/repo"], lookups, 100.0, 42)

        captured = capsys.readouterr()
        assert "Perfect configuration found" in captured.err
        assert mock_pool.imap_unordered.call_count == 1

    @patch("src.optimizer.multiprocessing.Pool")
    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer._island_evolution_task_wrapper")
    def test_polish_improves_fitness(
        self, _mock_wrapper, mock_run, mock_pool_cls, capsys
    ):
        mock_run.return_value = 100
        mock_pool = MagicMock()
        mock_pool.imap_unordered.return_value = iter(
            [
                (
                    0,
                    [{"config": {}, "fitness": 90}],
                    {"config": {}, "fitness": 90},
                ),
                (
                    1,
                    [{"config": {}, "fitness": 95}],
                    {"config": {}, "fitness": 95},
                ),
            ]
        )
        mock_pool_cls.return_value = mock_pool

        with patch("src.optimizer._polish_coordinate_descent") as mock_polish:
            mock_polish.return_value = ({"config": {}}, 80)

            lookups = _make_lookups()
            base = self._make_base_config()
            opt = GeneticAlgorithmOptimizer(self._make_config(polish_passes=3))
            _ = opt.optimize(base, ["/tmp/repo"], lookups, 100.0, 42)

            captured = capsys.readouterr()
            assert "Polish improved fitness" in captured.err

    @patch("src.optimizer.multiprocessing.Pool")
    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer._island_evolution_task_wrapper")
    def test_polish_no_improvement(
        self, _mock_wrapper, mock_run, mock_pool_cls, capsys
    ):
        mock_run.return_value = 100
        mock_pool = MagicMock()
        mock_pool.imap_unordered.return_value = iter(
            [
                (
                    0,
                    [{"config": {}, "fitness": 90}],
                    {"config": {}, "fitness": 90},
                ),
                (
                    1,
                    [{"config": {}, "fitness": 95}],
                    {"config": {}, "fitness": 95},
                ),
            ]
        )
        mock_pool_cls.return_value = mock_pool

        with patch("src.optimizer._polish_coordinate_descent") as mock_polish:
            mock_polish.return_value = ({"config": {}}, 95)

            lookups = _make_lookups()
            base = self._make_base_config()
            opt = GeneticAlgorithmOptimizer(self._make_config(polish_passes=3))
            _ = opt.optimize(base, ["/tmp/repo"], lookups, 100.0, 42)

            captured = capsys.readouterr()
            assert "Polish did not improve" in captured.err

    @patch("src.optimizer.multiprocessing.Pool")
    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer._island_evolution_task_wrapper")
    def test_no_repo_paths_returns_empty(
        self, _mock_wrapper, _mock_run, _mock_pool_cls, capsys
    ):
        lookups = _make_lookups()
        base = self._make_base_config()
        opt = GeneticAlgorithmOptimizer(self._make_config())
        result = opt.optimize(base, [], lookups, 100.0, 42)

        assert result == {}
        captured = capsys.readouterr()
        assert "No repository paths" in captured.err

    @patch("src.optimizer.multiprocessing.Pool")
    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer._island_evolution_task_wrapper")
    def test_num_islands_less_than_one_adjusted(
        self, _mock_wrapper, mock_run, mock_pool_cls, capsys
    ):
        mock_run.return_value = 100
        mock_pool = MagicMock()
        mock_pool.imap_unordered.return_value = iter(
            [
                (
                    0,
                    [{"config": {}, "fitness": 90}],
                    {"config": {}, "fitness": 90},
                ),
            ]
        )
        mock_pool_cls.return_value = mock_pool

        lookups = _make_lookups()
        base = self._make_base_config()
        opt = GeneticAlgorithmOptimizer(self._make_config(num_islands=0))
        _ = opt.optimize(base, ["/tmp/repo"], lookups, 100.0, 42)

        captured = capsys.readouterr()
        assert "must be at least 1" in captured.err

    @patch("src.optimizer.multiprocessing.Pool")
    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer._island_evolution_task_wrapper")
    def test_population_adjusted_for_minimum(
        self, _mock_wrapper, mock_run, mock_pool_cls, capsys
    ):
        mock_run.return_value = 100
        mock_pool = MagicMock()
        mock_pool.imap_unordered.return_value = iter(
            [
                (
                    0,
                    [{"config": {}, "fitness": 90}],
                    {"config": {}, "fitness": 90},
                ),
                (
                    1,
                    [{"config": {}, "fitness": 90}],
                    {"config": {}, "fitness": 90},
                ),
            ]
        )
        mock_pool_cls.return_value = mock_pool

        lookups = _make_lookups()
        base = self._make_base_config()
        opt = GeneticAlgorithmOptimizer(
            self._make_config(total_population_size=2, num_islands=2)
        )
        _ = opt.optimize(base, ["/tmp/repo"], lookups, 100.0, 42)

        captured = capsys.readouterr()
        assert "Adjusted total population size" in captured.err

    @patch("src.optimizer.multiprocessing.Pool")
    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer._island_evolution_task_wrapper")
    def test_migration_performed_periodically(
        self,
        _mock_wrapper,
        mock_run,
        mock_pool_cls,
        capsys,  # pyright: ignore[reportUnusedParameter]
    ):
        mock_run.return_value = 100
        mock_pool = MagicMock()
        mock_pool.imap_unordered.return_value = iter(
            [
                (
                    0,
                    [{"config": {}, "fitness": 90}],
                    {"config": {}, "fitness": 90},
                ),
                (
                    1,
                    [{"config": {}, "fitness": 95}],
                    {"config": {}, "fitness": 95},
                ),
            ]
        )
        mock_pool_cls.return_value = mock_pool

        with patch("src.optimizer._perform_migration") as mock_migrate:
            lookups = _make_lookups()
            base = self._make_base_config()
            opt = GeneticAlgorithmOptimizer(
                self._make_config(num_iterations=16, num_islands=2)
            )
            _ = opt.optimize(base, ["/tmp/repo"], lookups, 100.0, 42)

            mock_migrate.assert_called()

    @patch("src.optimizer.multiprocessing.Pool")
    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer._island_evolution_task_wrapper")
    def test_interrupted_terminates_pool(
        self,
        _mock_wrapper,
        mock_run,
        mock_pool_cls,
        capsys,  # pyright: ignore[reportUnusedParameter]
    ):
        mock_run.return_value = 100
        mock_pool = MagicMock()

        def raise_keyboard_interrupt(*_args):
            raise KeyboardInterrupt()

        mock_pool.imap_unordered.side_effect = raise_keyboard_interrupt
        mock_pool_cls.return_value = mock_pool

        lookups = _make_lookups()
        base = self._make_base_config()
        opt = GeneticAlgorithmOptimizer(self._make_config())
        _ = opt.optimize(base, ["/tmp/repo"], lookups, 100.0, 42)

        mock_pool.terminate.assert_called()
        mock_pool.join.assert_called()

    @patch("src.optimizer.multiprocessing.Pool")
    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer._island_evolution_task_wrapper")
    def test_polish_called_with_repo_paths(
        self,
        _mock_wrapper,
        mock_run,
        mock_pool_cls,
        capsys,  # pyright: ignore[reportUnusedParameter]
    ):
        mock_run.return_value = 100
        mock_pool = MagicMock()
        mock_pool.imap_unordered.return_value = iter(
            [
                (
                    0,
                    [{"config": {}, "fitness": 90}],
                    {"config": {}, "fitness": 90},
                ),
            ]
        )
        mock_pool_cls.return_value = mock_pool

        with patch("src.optimizer._polish_coordinate_descent") as mock_polish:
            mock_polish.return_value = ({"config": {}}, 80)
            lookups = _make_lookups()
            base = self._make_base_config()
            opt = GeneticAlgorithmOptimizer(
                self._make_config(num_islands=1, polish_passes=3)
            )
            _ = opt.optimize(base, ["/tmp/repo"], lookups, 100.0, 42)
            mock_polish.assert_called()


# ---------------------------------------------------------------------------
# _polish_single_option_task
# ---------------------------------------------------------------------------


class TestPolishSingleOptionTask:
    """Tests for the parallel polish worker function."""

    def test_returns_option_path_fitness_value(self):
        from src.optimizer import _polish_single_option_task  # pyright: ignore[reportPrivateUsage]

        lookups = _make_lookups(
            json_options={"UseTab": {"possible_values": ["Never", "Always"]}}
        )

        with (
            patch("src.optimizer.optimize_option_with_values") as mock_opt,
            patch("src.optimizer.IncrementalConfigBuilder") as mock_builder_cls,
        ):
            mock_opt.return_value = 42
            mock_builder = MagicMock()
            mock_builder_cls.return_value = mock_builder
            # Simulate optimize_option_with_values mutating the config
            original_config = {"UseTab": {"type": "str", "value": "Never"}}

            def side_effect(cfg, *_):
                cfg["UseTab"]["value"] = "Always"
                return 42

            mock_opt.side_effect = side_effect

            result = _polish_single_option_task(
                (
                    original_config,
                    "UseTab",
                    ["Never", "Always"],
                    lookups,
                    False,
                    "/tmp/repo",
                    100.0,
                    42,
                )
            )
            assert result[0] == "UseTab"
            assert result[1] == 42
            assert result[2] == "Always"

    def test_creates_builder_from_config_copy(self):
        from src.optimizer import _polish_single_option_task  # pyright: ignore[reportPrivateUsage]

        lookups = _make_lookups()

        with (
            patch("src.optimizer.optimize_option_with_values") as mock_opt,
            patch("src.optimizer.IncrementalConfigBuilder") as mock_builder_cls,
        ):
            mock_opt.return_value = 10
            mock_builder_cls.return_value = MagicMock()

            original = {"UseTab": {"type": "str", "value": "Never"}}
            _ = _polish_single_option_task(
                (
                    original,
                    "UseTab",
                    ["Never"],
                    lookups,
                    False,
                    "/tmp/repo",
                    100.0,
                    42,
                )
            )
            mock_builder_cls.assert_called_once_with(original)


# ---------------------------------------------------------------------------
# _polish_coordinate_descent — parallel path
# ---------------------------------------------------------------------------


class TestPolishCoordinateDescentParallel:
    """Tests for the parallel dispatch path of _polish_coordinate_descent."""

    def _make_config(self):
        return {
            "UseTab": {"type": "str", "value": "Never"},
            "IndentWidth": {"type": "int", "value": 4},
        }

    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer.IncrementalConfigBuilder")
    def test_parallel_path_dispatches_via_pool(self, mock_builder_cls, mock_run):
        mock_builder = MagicMock()
        mock_builder.build.return_value = "config\n"
        mock_builder_cls.return_value = mock_builder
        mock_run.return_value = 100

        mock_pool = MagicMock()
        mock_pool.map.return_value = [
            ("UseTab", 90, "Always"),
            ("IndentWidth", 85, 2),
        ]

        lookups = _make_lookups(
            json_options={
                "UseTab": {"possible_values": ["Never", "Always"]},
                "IndentWidth": {"possible_values": ["2", "4"]},
            }
        )
        config = self._make_config()

        _ = _polish_coordinate_descent(
            config, "/tmp/repo", lookups, False, 100.0, 42, max_passes=1, pool=mock_pool
        )

        mock_pool.map.assert_called_once()
        # Verify the task function was passed as the first argument
        call_args = mock_pool.map.call_args
        assert call_args[0][0] == _polish_single_option_task

    @patch("src.optimizer._polish_single_option_task")
    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer.IncrementalConfigBuilder")
    def test_parallel_path_applies_improvements(
        self, mock_builder_cls, mock_run, _mock_task
    ):
        mock_builder = MagicMock()
        mock_builder.build.return_value = "config\n"
        mock_builder_cls.return_value = mock_builder
        mock_run.return_value = 100

        mock_pool = MagicMock()
        mock_pool.map.return_value = [
            ("UseTab", 50, "Always"),  # improvement
            ("IndentWidth", 100, 4),  # no improvement
        ]

        lookups = _make_lookups(
            json_options={
                "UseTab": {"possible_values": ["Never", "Always"]},
                "IndentWidth": {"possible_values": ["2", "4"]},
            }
        )
        config = self._make_config()

        result_config, result_fitness = _polish_coordinate_descent(
            config, "/tmp/repo", lookups, False, 100.0, 42, max_passes=1, pool=mock_pool
        )

        assert result_config["UseTab"]["value"] == "Always"
        assert result_fitness == 50

    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer.IncrementalConfigBuilder")
    def test_parallel_path_converges_no_improvements(
        self, mock_builder_cls, mock_run, capsys
    ):
        mock_builder = MagicMock()
        mock_builder.build.return_value = "config\n"
        mock_builder_cls.return_value = mock_builder
        mock_run.return_value = 100

        mock_pool = MagicMock()
        mock_pool.map.return_value = [
            ("UseTab", 100, "Never"),  # no improvement
        ]

        lookups = _make_lookups(
            json_options={"UseTab": {"possible_values": ["Never", "Always"]}}
        )
        config = self._make_config()

        _ = _polish_coordinate_descent(
            config, "/tmp/repo", lookups, False, 100.0, 42, max_passes=3, pool=mock_pool
        )

        captured = capsys.readouterr()
        assert "converged after 1 pass" in captured.err
        # Only one pass should have run
        assert mock_pool.map.call_count == 1

    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer.IncrementalConfigBuilder")
    def test_parallel_path_bool_option(self, mock_builder_cls, mock_run, capsys):  # pyright: ignore[reportUnusedParameter]
        """Verify bool options are included in parallel dispatch."""
        mock_builder = MagicMock()
        mock_builder.build.return_value = "config\n"
        mock_builder_cls.return_value = mock_builder
        mock_run.return_value = 100

        mock_pool = MagicMock()
        mock_pool.map.return_value = [
            ("BreakBeforeBraces", 50, False),
        ]

        lookups = _make_lookups(json_options={})
        config = {
            "BreakBeforeBraces": {"type": "bool", "value": True},
        }

        _ = _polish_coordinate_descent(
            config, "/tmp/repo", lookups, False, 100.0, 42, max_passes=1, pool=mock_pool
        )

        # pool.map should have been called with a task for the bool option
        mock_pool.map.assert_called_once()
        tasks = mock_pool.map.call_args[0][1]
        assert len(tasks) == 1
        # Bool options get [True, False] as possible_values
        assert tasks[0][2] == [True, False]


class TestCheckpointWiring:
    """Tests for checkpoint save/resume wired into optimize()."""

    def _make_config(
        self,
        num_iterations: int = 2,
        total_population_size: int = 10,
        num_islands: int = 1,
        checkpoint_interval: int = 0,
    ) -> GeneticOptimizationConfig:
        return GeneticOptimizationConfig(
            num_iterations=num_iterations,
            total_population_size=total_population_size,
            num_islands=num_islands,
            debug=False,
            plot_fitness=False,
            polish_passes=0,
            checkpoint_interval=checkpoint_interval,
        )

    def _make_base_config(self):
        return {
            "UseTab": {"type": "str", "value": "Never"},
            "IndentWidth": {"type": "int", "value": 4},
        }

    @patch("src.optimizer.multiprocessing.Pool")
    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer._island_evolution_task_wrapper")
    def test_checkpoint_saved_at_interval(
        self,
        mock_wrapper,  # pyright: ignore[reportUnusedParameter]
        mock_run,
        mock_pool_cls,
        tmp_path,
        capsys,  # pyright: ignore[reportUnusedParameter]
    ):
        mock_run.return_value = 100
        mock_pool = MagicMock()
        # 4 iterations, checkpoint_interval=2 -> save at iteration 2
        mock_pool.imap_unordered.return_value = iter(
            [
                (0, [{"config": {}, "fitness": 90}], {"config": {}, "fitness": 90}),
            ]
            * 4
        )
        mock_pool_cls.return_value = mock_pool

        checkpoint_path = str(tmp_path / "ckpt.json")
        with patch("src.optimizer.save_checkpoint") as mock_save:
            lookups = _make_lookups()
            base = self._make_base_config()
            opt = GeneticAlgorithmOptimizer(
                self._make_config(num_iterations=4, checkpoint_interval=2)
            )
            _ = opt.optimize(base, ["/tmp/repo"], lookups, 100.0, 42, checkpoint_path)
            # Should save at iteration 2 (index 1), not at iteration 4 (last)
            assert mock_save.call_count == 1
            call_args = mock_save.call_args[0]
            assert call_args[3] == 1  # iteration index

    @patch("src.optimizer.multiprocessing.Pool")
    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer._island_evolution_task_wrapper")
    def test_checkpoint_not_saved_last_iteration(
        self,
        mock_wrapper,  # pyright: ignore[reportUnusedParameter]
        mock_run,
        mock_pool_cls,
        tmp_path,
    ):
        mock_run.return_value = 100
        mock_pool = MagicMock()
        mock_pool.imap_unordered.return_value = iter(
            [
                (0, [{"config": {}, "fitness": 90}], {"config": {}, "fitness": 90}),
            ]
            * 2
        )
        mock_pool_cls.return_value = mock_pool

        checkpoint_path = str(tmp_path / "ckpt.json")
        with patch("src.optimizer.save_checkpoint") as mock_save:
            lookups = _make_lookups()
            base = self._make_base_config()
            opt = GeneticAlgorithmOptimizer(
                self._make_config(num_iterations=2, checkpoint_interval=1)
            )
            _ = opt.optimize(base, ["/tmp/repo"], lookups, 100.0, 42, checkpoint_path)
            # interval=1, but skip last iteration -> save at iteration 1 only
            assert mock_save.call_count == 1

    @patch("src.optimizer.multiprocessing.Pool")
    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer._island_evolution_task_wrapper")
    def test_checkpoint_disabled_by_default(
        self,
        mock_wrapper,  # pyright: ignore[reportUnusedParameter]
        mock_run,
        mock_pool_cls,
        tmp_path,  # pyright: ignore[reportUnusedParameter]
    ):
        mock_run.return_value = 100
        mock_pool = MagicMock()
        mock_pool.imap_unordered.return_value = iter(
            [
                (0, [{"config": {}, "fitness": 90}], {"config": {}, "fitness": 90}),
            ]
        )
        mock_pool_cls.return_value = mock_pool

        # checkpoint_interval=0 (default), no checkpoint_path
        with patch("src.optimizer.save_checkpoint") as mock_save:
            lookups = _make_lookups()
            base = self._make_base_config()
            opt = GeneticAlgorithmOptimizer(self._make_config())
            _ = opt.optimize(base, ["/tmp/repo"], lookups, 100.0, 42)
            mock_save.assert_not_called()

    @patch("src.optimizer.multiprocessing.Pool")
    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer._island_evolution_task_wrapper")
    def test_resume_from_checkpoint(
        self,
        mock_wrapper,  # pyright: ignore[reportUnusedParameter]
        mock_run,
        mock_pool_cls,
        tmp_path,
        capsys,
    ):
        mock_run.return_value = 100
        mock_pool = MagicMock()
        mock_pool.imap_unordered.return_value = iter(
            [
                (0, [{"config": {}, "fitness": 80}], {"config": {}, "fitness": 80}),
            ]
        )
        mock_pool_cls.return_value = mock_pool

        # Create a checkpoint file
        checkpoint_path = str(tmp_path / "ckpt.json")
        save_checkpoint(
            checkpoint_path,
            {"IndentWidth": {"type": "int", "value": 2}},
            85.0,
            2,  # saved at iteration index 2 (3rd iteration)
            [[{"config": {}, "fitness": 85}]],
            [[100.0, 95.0, 85.0]],
        )

        with patch("src.optimizer.load_checkpoint") as mock_load:
            mock_load.return_value = {
                "best_config": {"IndentWidth": {"type": "int", "value": 2}},
                "best_fitness": 85.0,
                "iteration": 2,
                "populations": [[{"config": {}, "fitness": 85}]],
                "fitness_history_per_island": [[100.0, 95.0, 85.0]],
            }
            lookups = _make_lookups()
            base = self._make_base_config()
            opt = GeneticAlgorithmOptimizer(
                self._make_config(num_iterations=10, num_islands=1)
            )
            _ = opt.optimize(base, ["/tmp/repo"], lookups, 100.0, 42, checkpoint_path)
            mock_load.assert_called_once_with(checkpoint_path)
            captured = capsys.readouterr()
            assert "Resuming from checkpoint" in captured.err
            assert "Resumed at iteration 3" in captured.err

    @patch("src.optimizer.multiprocessing.Pool")
    @patch("src.optimizer.run_clang_format_and_count_changes")
    @patch("src.optimizer._island_evolution_task_wrapper")
    def test_resume_missing_checkpoint_starts_fresh(
        self,
        mock_wrapper,  # pyright: ignore[reportUnusedParameter]
        mock_run,
        mock_pool_cls,
        tmp_path,
        capsys,
    ):
        mock_run.return_value = 100
        mock_pool = MagicMock()
        mock_pool.imap_unordered.return_value = iter(
            [
                (0, [{"config": {}, "fitness": 90}], {"config": {}, "fitness": 90}),
            ]
        )
        mock_pool_cls.return_value = mock_pool

        checkpoint_path = str(tmp_path / "nonexistent.json")
        with patch("src.optimizer.load_checkpoint") as mock_load:
            mock_load.return_value = None
            lookups = _make_lookups()
            base = self._make_base_config()
            opt = GeneticAlgorithmOptimizer(self._make_config())
            _ = opt.optimize(base, ["/tmp/repo"], lookups, 100.0, 42, checkpoint_path)
            captured = capsys.readouterr()
            assert "Starting fresh" in captured.err
