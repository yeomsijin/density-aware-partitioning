"""Legacy figure adapter; use dap.DensityAwareForest for new applications."""

from .base import BaseIForest


class ObliqueIForest(BaseIForest):
    family = "eif"
    default_mode = "EIF"
