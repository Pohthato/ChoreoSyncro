"""Synthetic fingerprints and scenes for the identity tests: people are unit vectors."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from syncscore.pose.tracks import Detections


def unit(rng, dim: int = 32) -> np.ndarray:
    v = rng.normal(size=dim)
    return v / np.linalg.norm(v)


def near(v: np.ndarray, cos: float, rng) -> np.ndarray:
    """A unit vector at exactly `cos` to `v`, in a random direction."""
    w = rng.normal(size=v.shape)
    w -= (w @ v) * v
    w /= np.linalg.norm(w)
    return cos * v + np.sqrt(1 - cos ** 2) * w


@dataclass
class Person:
    look: np.ndarray                     # her fingerprint centre
    x: Callable[[int], float]            # where she stands in frame f
    track: Callable[[int], int | None]   # the tracker ID she has in frame f; None when not detected


def scene(people: list[Person], n_frames: int = 300, fps: float = 30.0, noise: float = 0.05, seed: int = 0) -> tuple[Detections, np.ndarray]:
    """People walking along a line: boxes [x-40, 200, x+40, 520], hips at y 400, every keypoint at 0.9.
    Returns the detections (with fingerprints) and the true person of each row."""
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_frames):
        for i, person in enumerate(people):
            track = person.track(f)
            if track is not None:
                rows.append((f, track, person.x(f), i))
    n = len(rows)
    xs = np.array([r[2] for r in rows], float)
    xy = np.zeros((n, 17, 2))
    xy[:, :, 0] = xs[:, None]
    xy[:, :, 1] = np.linspace(210, 510, 17)[None]
    xy[:, 11] = np.stack([xs - 10, np.full(n, 400.0)], 1)
    xy[:, 12] = np.stack([xs + 10, np.full(n, 400.0)], 1)
    looks = np.stack([people[r[3]].look for r in rows]) if n else np.zeros((0, 32))
    vectors = looks + noise * rng.normal(size=looks.shape)
    det = Detections(frame=np.array([r[0] for r in rows], int), track_id=np.array([r[1] for r in rows], int),
                     box=np.stack([xs - 40, np.full(n, 200.0), xs + 40, np.full(n, 520.0)], 1), box_conf=np.full(n, 0.9),
                     xy=xy, kconf=np.full((n, 17), 0.9), hist=np.zeros((n, 4)), fps=fps, n_frames=n_frames,
                     image_size=(1920, 1080), vectors=vectors / np.linalg.norm(vectors, axis=1, keepdims=True))
    return det, np.array([r[3] for r in rows], int)
