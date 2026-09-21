"""Fixed-length rolling windows for scalar or multivariate streams."""

from collections import deque

import numpy as np

from dap.partition import positive_int


def shingle(sequence, size):
    """Yield flattened oldest-to-newest windows; first output is at index size-1."""
    size = positive_int(size, "size")
    window = deque(maxlen=size)
    shape = None
    for value in sequence:
        value = np.asarray(value, dtype=float)
        if value.ndim > 1 or not value.size or not np.isfinite(value).all():
            raise ValueError("stream values must be finite scalars or 1D arrays")
        if shape is not None and value.shape != shape:
            raise ValueError("all stream values must have the same shape")
        shape = value.shape
        window.append(value.copy())
        if len(window) == size:
            yield np.asarray(window).reshape(-1).copy()
