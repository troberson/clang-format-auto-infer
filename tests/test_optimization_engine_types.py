"""Tests for the generic optimization engine types."""

import copy
from src.optimization_engine.types import (
    Individual,
    OptimizationResult,
    ParameterDef,
    SearchSpace,
)


class TestParameterDef:
    def test_basic_construction(self):
        p = ParameterDef(name="IndentWidth", param_type="int", possible_values=[2, 4])
        assert p.name == "IndentWidth"
        assert p.param_type == "int"
        assert p.possible_values == [2, 4]
        assert p.fixed is False
        assert p.mutable is True

    def test_fixed_parameter_not_mutable(self):
        p = ParameterDef(name="BasedOnStyle", param_type="str", fixed=True)
        assert p.mutable is False

    def test_no_values_not_mutable(self):
        p = ParameterDef(name="SomeOption", param_type="str", possible_values=[])
        assert p.mutable is False


class TestSearchSpace:
    def test_construction(self):
        params = {
            "a": ParameterDef(name="a", param_type="int", possible_values=[1, 2]),
            "b": ParameterDef(name="b", param_type="bool", fixed=True),
        }
        ss = SearchSpace(parameters=params)
        assert len(ss.parameters) == 2

    def test_mutable_parameters(self):
        params = {
            "a": ParameterDef(name="a", param_type="int", possible_values=[1, 2]),
            "b": ParameterDef(name="b", param_type="bool", fixed=True),
            "c": ParameterDef(name="c", param_type="str", possible_values=[]),
        }
        ss = SearchSpace(parameters=params)
        assert len(ss.mutable_parameters) == 1
        assert ss.mutable_parameters[0].name == "a"

    def test_mutable_names(self):
        params = {
            "x": ParameterDef(name="x", param_type="int", possible_values=[1]),
            "y": ParameterDef(name="y", param_type="bool", fixed=True),
        }
        ss = SearchSpace(parameters=params)
        assert ss.mutable_names == ["x"]

    def test_empty_search_space(self):
        ss = SearchSpace(parameters={})
        assert ss.mutable_parameters == []
        assert ss.mutable_names == []


class TestIndividual:
    def test_construction(self):
        ind = Individual(config={"a": 1}, fitness=10.0)
        assert ind.config == {"a": 1}
        assert ind.fitness == 10.0

    def test_default_fitness_is_inf(self):
        ind = Individual(config={"a": 1})
        assert ind.fitness == float("inf")

    def test_config_is_mutable(self):
        ind = Individual(config={"a": 1})
        ind.config["a"] = 2
        assert ind.config["a"] == 2

    def test_deep_copy_independence(self):
        ind1 = Individual(config={"a": 1, "b": [1, 2]}, fitness=5.0)
        ind2 = copy.deepcopy(ind1)
        ind2.config["a"] = 99
        ind2.config["b"].append(3)
        assert ind1.config["a"] == 1
        assert ind1.config["b"] == [1, 2]


class TestOptimizationResult:
    def test_construction(self):
        r = OptimizationResult(best_config={"a": 1}, best_fitness=3.0)
        assert r.best_config == {"a": 1}
        assert r.best_fitness == 3.0
