"""Synthetic fingerprints and scenes for the identity tests: people are unit vectors."""
from __future__ import annotations

import numpy as np


def unit(rng, dim: int = 32) -> np.ndarray:
    v = rng.normal(size=dim)
    return v / np.linalg.norm(v)


def near(v: np.ndarray, cos: float, rng) -> np.ndarray:
    """A unit vector at exactly `cos` to `v`, in a random direction."""
    w = rng.normal(size=v.shape)
    w -= (w @ v) * v
    w /= np.linalg.norm(w)
    return cos * v + np.sqrt(1 - cos ** 2) * w
