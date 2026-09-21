"""Bounded-window weighted rebuilding; intentionally a small runnable example."""

import numpy as np

from density_aware_forests import RollingWRCF, shingle

signal = np.sin(np.arange(80) / 5)
signal[45:48] += 4
model = RollingWRCF(n_estimators=5, window_size=24, random_state=42)
scores = [(t, model.update(point)) for t, point in enumerate(shingle(signal, 4), start=3)]
print("Highest scoring time indices:", sorted(scores, key=lambda row: -row[1])[:5])
