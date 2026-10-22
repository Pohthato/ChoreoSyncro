"""When is the group dancing together? Only then is "the group" a reference for one member.

Timing and the ghost compare a member with the other members. That only means something while
they are all dancing the same thing. During a solo the others hold still while one member
dances, and in split parts the others do different moves; comparing anyone with "the group"
then measures nothing (on real footage these stretches were a large part of why sections came
out with no reading).

Member i is together with the group at frame t when, over a short window:
- **most of the others dance the same thing**: the largest set of them whose limbs are within
  `max_spread_deg` of each other (pairwise, over `smooth_s`) is more than half of those seen. One other
  member soloing doesn't break it; split parts (2/2, 3/3) do. (Not "angle to their median": in a
  3/3 split the median falls between the parts and everyone looks only half as far from it as
  the parts are from each other.);
- **i dances what they dance**: i's limbs are within `max_own_deg` of the others' median (over the
  longer `own_smooth_s`, so a brief mistake doesn't count, a different part does);
- **i is in the same state**: moving while most of them move, still while they are still (moving =
  average speed over `smooth_s` above `moving_speed` times the member's usual moving speed). A solo
  fails here, and a group holding a pose together passes;
- i is seen, and at least `min_others` others are.

Limb bones only (upper arms, forearms, thighs, shins): the torso and head barely move in most
choreography and would dilute the comparison.

**The limits scale with the footage.** In 2D, camera angle and personal styling put members' limbs
apart even when they dance the same move: on the BTS clip pairs of members sat at a median of 17
degrees, a quarter of them above 24. Fixed 25/30-degree limits then called members of the whole
group dancing the chorus together only 52-77% of the time (100% with these). So both limits are at
least twice this clip's typical difference
(the median over pairs, or over members, while most members are seen), with `max_spread_deg` and
`max_own_deg` as floors. Clean synthetic groups keep the floors, so their splits are still caught.
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
    crowd = seen.sum(axis=0) >= max(2, int(np.ceil(2 * members / 3)))  # most members seen: what "typical" is judged on
    pair = {(a, b): _smooth(_limb_angle(u[a], w[a], u[b], w[b]), window) for a in range(members) for b in range(a + 1, members)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        typical_pair = np.nanmedian(np.concatenate([v[crowd] for v in pair.values()])) if pair and crowd.any() else 0.0
    max_spread = max(cfg.max_spread_deg, 2.0 * np.nan_to_num(typical_pair))
    close = np.eye(members, dtype=bool)[:, :, None].repeat(frames, axis=2)  # (M, M, T) within max_spread of each other
    for (a, b), angle in pair.items():
        close[a, b] = close[b, a] = np.nan_to_num(angle, nan=np.inf) <= max_spread
    own_angle = np.stack([_smooth(_limb_angle(u[i], w[i], ref[i], ref_w[i]), own_window) for i in range(members)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        typical_own = np.nanmedian(own_angle[:, crowd]) if crowd.any() else 0.0
    max_own = max(cfg.max_own_deg, 2.0 * np.nan_to_num(typical_own))
    out = np.zeros((members, frames), dtype=bool)
    for i in range(members):
        others = [j for j in range(members) if j != i]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            others_moving = np.nanmean(moving[others], axis=0)
        # Largest set of seen others close to one of them (a member and everyone near them), against
        # the number seen: members off screen neither agree nor disagree.
        seen_others = seen[others]
        near = close[np.ix_(others, others)] & seen_others[:, None, :] & seen_others[None, :, :]
        agree = near.sum(axis=1).max(axis=0) > seen_others.sum(axis=0) / 2
        own = np.nan_to_num(own_angle[i], nan=np.inf) <= max_own
        their_state = np.nan_to_num(others_moving, nan=0.0) >= cfg.min_moving_share  # already smoothed, like state[i]
        enough = seen[others].sum(axis=0) >= min(cfg.min_others, len(others))
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
