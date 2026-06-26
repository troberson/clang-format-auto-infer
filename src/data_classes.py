from dataclasses import dataclass
from typing import Any


@dataclass
class GeneticAlgorithmLookups:
    """
    Lookup data (JSON option values, forced options) needed by the optimizer.
    """

    json_options_lookup: dict[str, Any]
    forced_options_lookup: dict[str, Any]
