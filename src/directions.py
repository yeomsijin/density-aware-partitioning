"""Compatibility helpers for the original figure scripts."""

from dap.partition import random_directions, density_aware_direction


def random_unit_vectors(rng, d, k):
    return random_directions(rng, d, k)


def dad_direction_uniform(X, rng, n_dir_candidates=256):
    return density_aware_direction(X, random_state=rng, n_candidates=n_dir_candidates)
