"""Shape errors: what is still wrong after timing has been taken out.

Once a member has been shifted by their own offset, any remaining difference from the consensus
is a pose difference: the right moment, the wrong shape. It is measured as the angle between the
member's bone and the consensus bone, in degrees, and averaged per body part (weighted by trust).
"""
from __future__ import annotations

import numpy as np

from ..features.bones import PART_OF_BONE, PARTS, angle_between


def shape_errors(member_u: np.ndarray, member_w: np.ndarray, ref_u: np.ndarray, ref_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-bone (T, B) and per-part (T, P) angular error in degrees, NaN where unknown."""
    weight = np.minimum(member_w, ref_w)
    bone = np.where(weight > 0, angle_between(member_u, ref_u), np.nan)
    part = np.full((len(bone), len(PARTS)), np.nan)
    for p in range(len(PARTS)):
        cols = PART_OF_BONE == p
        w = np.where(np.isfinite(bone[:, cols]), weight[:, cols], 0.0)
        total = w.sum(axis=1)
        with np.errstate(invalid="ignore"):
            part[:, p] = np.where(total > 0, np.nansum(bone[:, cols] * w, axis=1) / total, np.nan)
    return bone, part
