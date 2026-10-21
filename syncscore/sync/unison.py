"""When is the group dancing together? Only then is "the group" a reference for one member.

Timing and the ghost compare a member with the other members. That only means something while
they are all dancing the same thing. During a solo the others hold still while one member
dances, and in split parts the others do different moves; comparing anyone with "the group"
then measures nothing (on real footage these stretches were a large part of why sections came
out with no reading).

Member i is together with the group at frame t when, over a short window:
- the **others agree** on their pose: the median, over every pair of them, of the angle between
  the pair's limbs is at most `max_spread_deg` (over `smooth_s`). Split parts fail here. (Pairs,
  not "angle to the median": in a 3/3 split the median falls between the two parts and everyone
  looks only half as far from it as the parts are from each other);
- **i dances what they dance**: i's limbs are within `max_own_deg` of the others' median (over the
  longer `own_smooth_s`, so a brief mistake doesn't count, a different part does);
- **i is in the same state**: moving while most of them move, still while they are still (moving =
  average speed over `smooth_s` above `moving_speed` times the member's usual moving speed). A solo
  fails here, and a group holding a pose together passes;
- i is seen, and at least `min_others` others are.

Limb bones only (upper arms, forearms, thighs, shins): the torso and head barely move in most
choreography and would dilute the comparison.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.ndimage import uniform_filter1d

from ..features.bones import BONE_NAMES
from .consensus import leave_one_out_median

_LIMBS = [BONE_NAMES.index(n) for n in ("left_upper_arm", "left_forearm", "right_upper_arm", "right_forearm",
                                        "left_thigh", "left_shin", "right_thigh", "right_shin")]


def _smooth(x: np.ndarray, frames: int) -> np.ndarray:
    """Moving average along the last axis over known samples (NaN where none were known)."""
    known = np.isfinite(x)
    size = max(1, int(frames))
    num = uniform_filter1d(np.where(known, x, 0.0), size, axis=-1, mode="nearest")
    den = uniform_filter1d(known.astype(float), size, axis=-1, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0.25, num / den, np.nan)


def _limb_angle(u: np.ndarray, w: np.ndarray, ref: np.ndarray, ref_w: np.ndarray) -> np.ndarray:
    """(..., T) weighted mean angle in degrees between u and ref over the limb bones; NaN if too few known."""
    a, b = u[..., _LIMBS, :], ref[..., _LIMBS, :]
    weight = np.where(np.isfinite(a[..., 0]) & np.isfinite(b[..., 0]), w[..., _LIMBS] * ref_w[..., _LIMBS], 0.0)
    cos = np.clip(np.sum(np.nan_to_num(a) * np.nan_to_num(b), axis=-1), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos))
    total = weight.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 1.0, (weight * angle).sum(axis=-1) / total, np.nan)


def member_speed(v: np.ndarray, vw: np.ndarray) -> np.ndarray:
    """(T,) bone angular speed (RMS over bones) in units of the member's usual *moving* speed (the
    median over frames where they move at all). A median over all frames would be 0 for a dancer
    who holds still more than half the time, and every speed would become unknown."""
    known = np.isfinite(v[..., 0]) & (vw > 0)
    w = np.where(known, vw, 0.0)
    total = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.where(total > 0.5, np.sqrt((w * np.sum(np.nan_to_num(v) ** 2, axis=-1)).sum(axis=1) / total), np.nan)
    moving = speed[np.isfinite(speed) & (speed > 1e-3)]
    return speed / np.median(moving) if moving.size else np.full(len(v), np.nan)


def together(u: np.ndarray, w: np.ndarray, speed: np.ndarray, fps: float, cfg) -> np.ndarray:
    """(M, T) bool: member i dances together with the others at frame t. u (M, T, B, 2), w (M, T, B),
    speed (M, T) from `member_speed`, NaN where unseen. `cfg` is config.unison."""
    members, frames = speed.shape
    ref, ref_w = leave_one_out_median(u, w)
    seen = np.isfinite(speed)
    window, own_window = int(round(cfg.smooth_s * fps)), int(round(cfg.own_smooth_s * fps))
    with np.errstate(invalid="ignore"):
        moving = np.where(seen, (_smooth(speed, window) > cfg.moving_speed).astype(float), np.nan)
    state = np.nan_to_num(moving) > 0.5
    pair_angle = {(a, b): _limb_angle(u[a], w[a], u[b], w[b]) for a in range(members) for b in range(a + 1, members)}
    out = np.zeros((members, frames), dtype=bool)
    for i in range(members):
        others = [j for j in range(members) if j != i]
        pairs = [angle for (a, b), angle in pair_angle.items() if i not in (a, b)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            spread = np.nanmedian(np.stack(pairs), axis=0) if pairs else np.full(frames, np.nan)
            others_moving = np.nanmean(moving[others], axis=0)
        agree = np.nan_to_num(_smooth(spread, window), nan=np.inf) <= cfg.max_spread_deg
        own = np.nan_to_num(_smooth(_limb_angle(u[i], w[i], ref[i], ref_w[i]), own_window), nan=np.inf) <= cfg.max_own_deg
        their_state = np.nan_to_num(others_moving, nan=0.0) >= cfg.min_moving_share  # already smoothed, like state[i]
        enough = seen[others].sum(axis=0) >= cfg.min_others
        out[i] = seen[i] & enough & agree & own & (state[i] == their_state)
    return out


def apart_intervals(flags: np.ndarray, fps: float, min_s: float = 1.0, seen: np.ndarray | None = None) -> list:
    """Per member, [(start_s, end_s), ...] stretches of at least `min_s` where they were seen (if
    `seen` is given) but not together with the group."""
    out = []
    for i, row in enumerate(flags):
        apart = ~row if seen is None else (~row & seen[i])
        edges = np.diff(np.r_[0, apart.astype(int), 0])
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        out.append([(float(a / fps), float(b / fps)) for a, b in zip(starts, ends) if (b - a) / fps >= min_s])
    return out
