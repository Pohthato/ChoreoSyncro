"""Keypoint smoothing: One Euro filter, run forward and backward so it adds no lag.

Pose models jitter by a few pixels from frame to frame even when a dancer stands still. A fixed
low-pass filter that removes the jitter also blunts the fast hits we want to time. The One Euro
filter (Casiez et al., CHI 2012) changes its cut-off with speed: heavy smoothing when a joint is
nearly still (where jitter is all there is), light smoothing when it moves fast (where lag
would hurt).

Any causal filter delays its output, and this filter's delay depends on how fast the member
moves. That is exactly what this tool measures, so a causal filter would bias the results. The
analysis is offline, so we run the filter forward and backward in time and average the two.
Their delays are equal and opposite and cancel (tested in tests/test_smooth.py).

Speeds are measured in torso lengths per second, so `beta` means the same for a dancer near the
camera and one far away.
"""
from __future__ import annotations

import numpy as np

from ..config import SmoothingConfig


def _alpha(cutoff: np.ndarray | float, fps: float) -> np.ndarray:
    r = 2 * np.pi * np.asarray(cutoff) / fps
    return r / (r + 1)


def one_euro(x: np.ndarray, fps: float, min_cutoff: float, beta: float, d_cutoff: float) -> np.ndarray:
    """Causal One Euro filter over axis 0. x (T, K, D); NaN samples restart the filter."""
    out = np.full_like(x, np.nan)
    prev = np.full(x.shape[1:], np.nan)
    dprev = np.zeros(x.shape[1:])
    a_d = _alpha(d_cutoff, fps)
    for t in range(len(x)):
        cur = x[t]
        known = np.isfinite(cur).all(axis=-1)
        fresh = known & ~np.isfinite(prev).all(axis=-1)
        with np.errstate(invalid="ignore"):
            dhat = dprev + a_d * ((cur - prev) * fps - dprev)
            a = _alpha(min_cutoff + beta * np.linalg.norm(dhat, axis=-1), fps)[..., None]
            xhat = prev + a * (cur - prev)
        xhat = np.where(fresh[..., None], cur, xhat)
        dhat = np.where(fresh[..., None], 0.0, dhat)
        out[t] = np.where(known[..., None], xhat, np.nan)
        prev = out[t]
        dprev = np.where(known[..., None], dhat, 0.0)
    return out


def fill_short_gaps(x: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate runs of missing samples no longer than `max_gap` frames.

    x (T, K, D). Returns the filled array and a (T, K) mask of what was filled. Longer gaps stay
    missing: guessing a whole second of a dancer's arm would be inventing data.
    """
    out = x.copy()
    filled = np.zeros(x.shape[:2], dtype=bool)
    frames = len(x)
    for k in range(x.shape[1]):
        missing = ~np.isfinite(x[:, k]).all(axis=-1)
        t = 0
        while t < frames:
            if not missing[t]:
                t += 1
                continue
            end = t
            while end < frames and missing[end]:
                end += 1
            if t > 0 and end < frames and end - t <= max_gap:
                w = (np.arange(t, end) - (t - 1)) / (end - (t - 1))
                out[t:end, k] = (1 - w)[:, None] * x[t - 1, k] + w[:, None] * x[end, k]
                filled[t:end, k] = True
            t = end
    return out, filled


def body_scale(xy: np.ndarray) -> float:
    """Median torso length (mid-shoulder to mid-hip) in pixels, used as the unit of speed."""
    torso = np.linalg.norm((xy[:, 5] + xy[:, 6]) / 2 - (xy[:, 11] + xy[:, 12]) / 2, axis=-1)
    torso = torso[np.isfinite(torso) & (torso > 1)]
    return float(np.median(torso)) if torso.size else 100.0


def smooth_keypoints(xy: np.ndarray, conf: np.ndarray, fps: float, cfg: SmoothingConfig, min_conf: float, scale: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """One member's keypoints (T, 17, 2) and confidences (T, 17) -> smoothed copies.

    Low-confidence points become missing, short gaps are filled (they get the bare
    threshold confidence), and the forward and backward One Euro passes are averaged.
    """
    xy = np.where((conf >= min_conf)[..., None], xy, np.nan)
    xy, filled = fill_short_gaps(xy, cfg.max_gap_frames)
    conf = conf.copy()
    conf[filled] = min_conf  # filled points are usable, but trusted least
    conf[~np.isfinite(xy).all(axis=-1)] = 0.0
    unit = scale or body_scale(xy)
    norm = xy / unit
    forward = one_euro(norm, fps, cfg.min_cutoff, cfg.beta, cfg.d_cutoff)
    backward = one_euro(norm[::-1], fps, cfg.min_cutoff, cfg.beta, cfg.d_cutoff)[::-1]
    return (forward + backward) / 2 * unit, conf
