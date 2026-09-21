"""Legacy figure adapter; use dap.DensityAwareForest for new applications."""

from .base import BaseIForest


class IsolationForestAD(BaseIForest):
    family = "if"
    default_mode = "IF"
