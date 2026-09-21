"""Weighted isolation forest, Yeom & Jung (2026)."""

from numbers import Integral

from dap import DensityAwareForest

__version__ = "0.2.0"
__all__ = ["WeightedIsolationForest"]


class WeightedIsolationForest(DensityAwareForest):
    """WIF: uniform feature choice and count < alpha threshold acceptance.

    alpha is an integer >=2, following weighted-paper section 5.1. Thus alpha=2
    is identical to Drop-in DAS alpha=1. The shared implementation lives in dap.
    Scores and sklearn-style methods are inherited from DensityAwareForest.
    """

    def __init__(
        self,
        *,
        n_estimators=100,
        max_samples=256,
        max_depth=None,
        alpha=2,
        contamination="auto",
        random_state=None,
    ):
        super().__init__(
            model="if",
            das=True,
            dad=False,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_depth=max_depth,
            alpha=alpha,
            contamination=contamination,
            random_state=random_state,
        )

    def _split_alpha(self):
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, Integral) or self.alpha < 2:
            raise ValueError("WIF alpha must be an integer >= 2 (accept count < alpha)")
        return self.alpha - 1
