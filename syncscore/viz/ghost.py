"""The ghost: what the other members are doing at this moment, drawn on each member's own body.

For each member and each video frame, the ghost is the median direction of every bone over the
*other* members at that same frame (leave-one-out, as in consensus.py: a member never drags their
own reference toward themselves). It is drawn from the member's own pelvis with the member's own
bone lengths, so it lines up with them and differs only where the pose or the timing does.

"At that same frame" matters: no timing estimate sits in between. A member who is late is still
doing the move the others already did, so their skeleton visibly trails the ghost; a member with
a wrong arm shows the gap between their arm and the ghost's.

A member dancing the mirrored version (config `features.mirror`) is compared mirrored, exactly as
the measurement does: the ghost is built on their mirrored body and mirrored back.
"""
from __future__ import annotations

import warnings

import numpy as np

from ..features.bones import BONE_NAMES, bone_directions, mirror_keypoints
from ..sync.consensus import leave_one_out_median

_B = {name: i for i, name in enumerate(BONE_NAMES)}
# (bone, from joint, to joint): limbs chained outward once the torso is placed.
_CHAINS = (
    ("left_upper_arm", 5, 7), ("left_forearm", 7, 9), ("right_upper_arm", 6, 8), ("right_forearm", 8, 10),
    ("left_thigh", 11, 13), ("left_shin", 13, 15), ("right_thigh", 12, 14), ("right_shin", 14, 16),
)


def _lengths(xy: np.ndarray, conf: np.ndarray, min_conf: float) -> np.ndarray:
    """(T, B) the member's bone lengths: this frame's where measured, else their median over the clip."""
    length = bone_directions(xy, conf, min_conf, foreshortening=False).length
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        typical = np.nanmedian(length, axis=0)
    return np.where(np.isfinite(length), length, typical[None])


def pose_from_directions(u: np.ndarray, w: np.ndarray, xy: np.ndarray, conf: np.ndarray, min_conf: float) -> np.ndarray:
    """Keypoints (T, 17, 2) of the pose given by bone directions u (T, B, 2), drawn on this body.

    Joints 0 and 5-16 are placed by chaining bones out from the member's pelvis; eyes and ears
    (1-4) are not drawn. A joint is NaN where its chain is unknown or the member's pelvis is."""
    frames = len(u)
    d = np.where(((w > 0) & np.isfinite(u[..., 0]))[..., None], u, np.nan)
    length = _lengths(xy, conf, min_conf)
    hips_seen = (conf[:, 11] >= min_conf) & (conf[:, 12] >= min_conf)
    pelvis = np.where(hips_seen[:, None], (xy[:, 11] + xy[:, 12]) / 2, np.nan)

    def step(bone: str) -> np.ndarray:
        b = _B[bone]
        return d[:, b] * length[:, b, None]

    out = np.full((frames, 17, 2), np.nan)
    out[:, 11] = pelvis + 0.5 * step("hips")  # the hips bone runs right hip -> left hip
    out[:, 12] = pelvis - 0.5 * step("hips")
    neck = pelvis + step("spine")
    out[:, 5] = neck + 0.5 * step("shoulders")  # right shoulder -> left shoulder
    out[:, 6] = neck - 0.5 * step("shoulders")
    out[:, 0] = neck + step("head")
    for bone, start, end in _CHAINS:
        out[:, end] = out[:, start] + step(bone)
    return out


def ghost_keypoints(xy: np.ndarray, conf: np.ndarray, mirrored: np.ndarray, min_conf: float) -> np.ndarray:
    """(M, T, 17, 2) every member's ghost. xy (M, T, 17, 2), conf (M, T, 17), mirrored (M, T) bool."""
    members = len(xy)
    u, w, flipped = [], [], []
    for i in range(members):
        m_xy, m_conf = mirror_keypoints(xy[i], conf[i])
        plain, mirror = bone_directions(xy[i], conf[i], min_conf), bone_directions(m_xy, m_conf, min_conf)
        flip = mirrored[i][:, None]
        u.append(np.where(flip[..., None], mirror.u, plain.u))  # every member in the group's convention
        w.append(np.where(flip, mirror.weight, plain.weight))
        flipped.append((m_xy, m_conf))
    ref_u, ref_w = leave_one_out_median(np.stack(u), np.stack(w), min_members=2)
    out = np.full(xy.shape, np.nan)
    for i in range(members):
        plain = pose_from_directions(ref_u[i], ref_w[i], xy[i], conf[i], min_conf)
        if mirrored[i].any():
            m_xy, m_conf = flipped[i]
            back = mirror_keypoints(pose_from_directions(ref_u[i], ref_w[i], m_xy, m_conf, min_conf))
            plain = np.where(mirrored[i][:, None, None], back, plain)
        out[i] = plain
    return out


def best_lag(member_u: np.ndarray, ghost_u: np.ndarray, max_lag: int) -> int:
    """Frames by which the member trails the ghost (+ = behind): the shift at which the member's
    bone directions best match the ghost's (mean cosine over bones and frames known in both)."""
    frames = len(member_u)
    best, best_score = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        a = member_u[max(lag, 0):frames + min(lag, 0)]
        b = ghost_u[max(-lag, 0):frames - max(lag, 0)]
        cos = np.sum(a * b, axis=-1)
        known = np.isfinite(cos)
        if known.any() and cos[known].mean() > best_score:
            best, best_score = lag, float(cos[known].mean())
    return best
