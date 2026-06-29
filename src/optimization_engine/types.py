"""Generic optimization types with no domain-specific dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParameterDef:
    """Definition of a single tunable parameter in the search space.

    Attributes:
        name: Unique identifier for this parameter.
        param_type: Type hint (e.g. 'int', 'bool', 'str').
        possible_values: Allowed values. Empty means the parameter is fixed.
        fixed: If True, this parameter must not be mutated during optimization.
        tier: Optimization phase this parameter belongs to.
            - 'resolve': Guessed values needing empirical disambiguation.
            - 'structure': High-impact structural options.
            - 'polish': Lower-impact options for final tuning.
        confidence: How certain the analyzer is about this value.
            - 'forced': Analyzer is certain, must not be changed.
            - 'detected': Analyzer detected from source, fixed in search space.
            - 'guessed': Analyzer guessed, needs empirical validation (mutable).
            - '' (empty): Not detected by analyzer, candidate for impact scan.
    """

    name: str
    param_type: str
    possible_values: list[Any] = field(default_factory=list)
    fixed: bool = False
    tier: str = "polish"
    confidence: str = ""

    @property
    def mutable(self) -> bool:
        """Whether this parameter can be mutated during optimization."""
        return not self.fixed and bool(self.possible_values)


@dataclass(frozen=True)
class SearchSpace:
    """Container for all tunable parameters.

    Attributes:
        parameters: Mapping from parameter name to its definition.
    """

    parameters: dict[str, ParameterDef]

    @property
    def mutable_parameters(self) -> list[ParameterDef]:
        """Return only parameters that can be mutated."""
        return [p for p in self.parameters.values() if p.mutable]

    @property
    def mutable_names(self) -> list[str]:
        """Return names of mutable parameters."""
        return [p.name for p in self.mutable_parameters]

    def mutable_by_tier(self, tier: str) -> list[ParameterDef]:
        """Return mutable parameters belonging to the given tier."""
        return [p for p in self.mutable_parameters if p.tier == tier]

    def remaining_fixed(self) -> list[ParameterDef]:
        """Return parameters that are still fixed (not yet unlocked)."""
        return [p for p in self.parameters.values() if p.fixed]

    def fix(self, names: list[str]) -> SearchSpace:
        """Return a new SearchSpace with the given parameters fixed.

        Fixed parameters become immutable and are excluded from future
        optimization rounds. They retain their current values.

        Args:
            names: Parameter names to fix.

        Returns:
            A new SearchSpace with the specified parameters fixed.
        """
        new_parameters: dict[str, ParameterDef] = {}
        name_set = set(names)
        for name, param in self.parameters.items():
            if name in name_set and not param.fixed:
                new_parameters[name] = ParameterDef(
                    name=param.name,
                    param_type=param.param_type,
                    possible_values=param.possible_values,
                    fixed=True,
                    tier=param.tier,
                    confidence=param.confidence,
                )
            else:
                new_parameters[name] = param
        return SearchSpace(parameters=new_parameters)

    def unlock(self, names: list[str]) -> SearchSpace:
        """Return a new SearchSpace with the given parameters unlocked.

        Unlocked parameters become mutable with tier 'polish' and inherit
        their possible_values from the existing definition. If a parameter
        has no possible_values, it is not unlocked.

        Args:
            names: Parameter names to unlock.

        Returns:
            A new SearchSpace with the specified parameters mutable.
        """
        new_parameters: dict[str, ParameterDef] = {}
        name_set = set(names)
        for name, param in self.parameters.items():
            if name in name_set and param.fixed and param.possible_values:
                new_parameters[name] = ParameterDef(
                    name=param.name,
                    param_type=param.param_type,
                    possible_values=param.possible_values,
                    fixed=False,
                    tier=param.tier,
                )
            else:
                new_parameters[name] = param
        return SearchSpace(parameters=new_parameters)


@dataclass
class Individual:
    """One candidate solution in the search space.

    Attributes:
        config: Mapping from parameter name to its current value.
        fitness: Lower is better. float('inf') means unevaluated or invalid.
    """

    config: dict[str, Any]
    fitness: float = float("inf")


@dataclass
class OptimizationResult:
    """Final output of an optimization run.

    Attributes:
        best_config: The best configuration found.
        best_fitness: Fitness of the best configuration.
        evaluations_used: Number of fitness evaluations actually performed.
            Allows callers to deduct only consumed budget.
    """

    best_config: dict[str, Any]
    best_fitness: float
    evaluations_used: int = 0
