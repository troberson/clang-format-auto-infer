"""Tests for optimization engine types."""

from src.optimization_engine.types import (
    Individual,
    OptimizationResult,
    ParameterDef,
    SearchSpace,
)


class TestParameterDef:
    def test_default_tier_is_polish(self):
        p = ParameterDef("IndentWidth", "int", [4, 8])
        assert p.tier == "polish"

    def test_explicit_tier(self):
        p = ParameterDef("IndentWidth", "int", [4, 8], tier="structure")
        assert p.tier == "structure"

    def test_mutable_when_has_values(self):
        p = ParameterDef("IndentWidth", "int", [4, 8])
        assert p.mutable is True

    def test_not_mutable_when_fixed(self):
        p = ParameterDef("BreakBeforeBraces", "str", ["Custom"], fixed=True)
        assert p.mutable is False

    def test_not_mutable_when_no_values(self):
        p = ParameterDef("Language", "str", [])
        assert p.mutable is False


class TestSearchSpace:
    def _make_space(self) -> SearchSpace:
        return SearchSpace(
            parameters={
                "IndentWidth": ParameterDef(
                    "IndentWidth", "int", [4, 8], tier="resolve"
                ),
                "ColumnLimit": ParameterDef(
                    "ColumnLimit", "int", [80, 100, 120], tier="structure"
                ),
                "AllowShortFunctions": ParameterDef(
                    "AllowShortFunctions", "str", ["Yes", "No"], tier="structure"
                ),
                "SpaceAfterAssignment": ParameterDef(
                    "SpaceAfterAssignment", "bool", [True, False], tier="polish"
                ),
                "BreakBeforeBraces": ParameterDef(
                    "BreakBeforeBraces", "str", ["Custom"], fixed=True
                ),
            }
        )

    def test_mutable_parameters_excludes_fixed(self):
        space = self._make_space()
        names = [p.name for p in space.mutable_parameters]
        assert "BreakBeforeBraces" not in names
        assert "IndentWidth" in names

    def test_mutable_names(self):
        space = self._make_space()
        assert space.mutable_names == [
            "IndentWidth",
            "ColumnLimit",
            "AllowShortFunctions",
            "SpaceAfterAssignment",
        ]

    def test_mutable_by_tier_resolve(self):
        space = self._make_space()
        params = space.mutable_by_tier("resolve")
        assert len(params) == 1
        assert params[0].name == "IndentWidth"

    def test_mutable_by_tier_structure(self):
        space = self._make_space()
        params = space.mutable_by_tier("structure")
        assert len(params) == 2
        names = [p.name for p in params]
        assert "ColumnLimit" in names
        assert "AllowShortFunctions" in names

    def test_mutable_by_tier_polish(self):
        space = self._make_space()
        params = space.mutable_by_tier("polish")
        assert len(params) == 1
        assert params[0].name == "SpaceAfterAssignment"

    def test_mutable_by_tier_unknown_returns_empty(self):
        space = self._make_space()
        params = space.mutable_by_tier("nonexistent")
        assert params == []

    def test_mutable_by_tier_excludes_fixed(self):
        space = self._make_space()
        params = space.mutable_by_tier("polish")
        names = [p.name for p in params]
        assert "BreakBeforeBraces" not in names


class TestIndividual:
    def test_default_fitness_is_inf(self):
        ind = Individual(config={"IndentWidth": 4})
        assert ind.fitness == float("inf")

    def test_set_fitness(self):
        ind = Individual(config={"IndentWidth": 4}, fitness=10.0)
        assert ind.fitness == 10.0


class TestOptimizationResult:
    def test_attributes(self):
        result = OptimizationResult(best_config={"IndentWidth": 4}, best_fitness=5.0)
        assert result.best_config == {"IndentWidth": 4}
        assert result.best_fitness == 5.0
