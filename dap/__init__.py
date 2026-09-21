"""Density-aware partitioning primitives and forest estimators."""

from .forest import DensityAwareForest
from .partition import (
    counting_map,
    counting_map_range,
    density_aware_direction,
    density_aware_split,
    density_measure,
    epsilon,
)

__version__ = "0.2.0"
__all__ = [
    "counting_map",
    "counting_map_range",
    "density_measure",
    "epsilon",
    "density_aware_split",
    "density_aware_direction",
    "DensityAwareForest",
]
