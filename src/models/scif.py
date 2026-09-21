"""Legacy figure adapter; use dap.DensityAwareForest for new applications."""

from .base import BaseIForest


class SCiForest(BaseIForest):
    family = "scif"
    default_mode = "SCIF"
