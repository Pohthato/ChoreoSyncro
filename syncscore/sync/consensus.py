"""The group's reference motion, and moving a member along time.

The consensus is what "the group" is doing at each moment: for each bone, the median direction
of the other members. Two choices matter:

- **Leave-one-out.** When member i is scored, i is not part of the reference. Otherwise i pulls
  the reference toward themselves, and with only 4-7 members that hides a real share of their
  error.
- **Median, not mean.** If one of the others holds a limb wrong, the mean is dragged toward
  them and everybody else then looks a little wrong. The median ignores a single outlier.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-9


def warp(u: np.ndarray, w: np.ndarray, shift: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample a direction series at frame t + shift[t] (fractional, linear interpolation).

    u (T, B, 2) unit vectors, w (T, B) weights, shift (T,) in frames. Interpolated vectors are
    renormalised. Positions outside the clip, or next to an unknown sample, come back unknown.
    """
    frames = len(u)
    pos = np.arange(frames) + np.asarray(shift, dtype=float)
    inside = (pos >= 0) & (pos <= frames - 1)
    pos = np.clip(pos, 0, frames - 1)
    i0 = np.minimum(np.floor(pos).astype(int), frames - 2)
    f = (pos - i0)[:, None]
    u0, u1, w0, w1 = u[i0], u[i0 + 1], w[i0], w[i0 + 1]
    on0, on1 = f < _EPS, f > 1 - _EPS
    mixed = (1 - f)[..., None] * u0 + f[..., None] * u1
    out = np.where(on0[..., None], u0, np.where(on1[..., None], u1, mixed))
    weight = np.where(on0, w0, np.where(on1, w1, np.minimum(w0, w1)))
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    out = out / np.where(norm > _EPS, norm, np.nan)
    known = inside[:, None] & np.isfinite(out[..., 0]) & (weight > 0)
    return np.where(known[..., None], out, np.nan), np.where(known, weight, 0.0)


def nanmedian_first_axis(x: np.ndarray) -> np.ndarray:
    """Median over axis 0 ignoring NaN (NaN where all are NaN).

    Same result as np.nanmedian(x, axis=0), but sorting is much faster than NumPy's
    masked-array path when axis 0 is short (a handful of members).
    """
    ordered = np.sort(x, axis=0)  # NaNs sort last
    count = np.isfinite(x).sum(axis=0)
    lo = np.take_along_axis(ordered, np.maximum((count - 1) // 2, 0)[None], axis=0)[0]
    hi = np.take_along_axis(ordered, np.maximum(count // 2, 0)[None], axis=0)[0]
    return np.where(count > 0, (lo + hi) / 2, np.nan)


def leave_one_out_median(u: np.ndarray, w: np.ndarray, min_weight: float = 0.05, min_members: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """For every member i, the median direction of all other members.

    u (M, T, B, 2), w (M, T, B) -> consensus (M, T, B, 2), consensus weight (M, T, B).
    The weight is the median trust of the members that were available; it is 0 where fewer
    than `min_members` others were seen.
    """
    members = len(u)
    valid = w > min_weight
    values = np.where(valid[..., None], u, np.nan)
    weights = np.where(valid, w, np.nan)
    out = np.full_like(u, np.nan)
    out_w = np.zeros_like(w)
    for i in range(members):
        others = [j for j in range(members) if j != i]
        med = nanmedian_first_axis(values[others])
        norm = np.linalg.norm(med, axis=-1, keepdims=True)
        with np.errstate(invalid="ignore"):
            med = med / np.where(norm > _EPS, norm, np.nan)
        count = valid[others].sum(axis=0)
        ok = (count >= min_members) & np.isfinite(med[..., 0])
        out[i] = np.where(ok[..., None], med, np.nan)
        out_w[i] = np.where(ok, nanmedian_first_axis(weights[others]), 0.0)
    return out, out_w
