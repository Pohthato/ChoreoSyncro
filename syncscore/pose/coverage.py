"""How much of the clip has every member tracked, and why a member goes missing.

A member can be missing in a frame for two different reasons, with different fixes:
- **the detector didn't find them** (usually hidden behind another dancer): fewer clear people
  were detected than there are members;
- **they were detected but not assigned**: as many clear people as members were detected, but
  the member assignment left one out (a piece too short to be a tracklet, or one that fit no
  member).
"""
from __future__ import annotations

import numpy as np

from .tracks import Detections, MemberTracks

GAP_BINS_S = ((0.0, 0.5, "<0.5s"), (0.5, 2.0, "0.5-2s"), (2.0, np.inf, ">=2s"))


def _clear_per_frame(det: Detections, min_keypoint_conf: float) -> np.ndarray:
    """Clearly visible people per frame, by the same rule as count_members."""
    clear = (det.box_conf >= 0.5) & ((det.kconf >= min_keypoint_conf).sum(axis=1) >= 8)
    return np.bincount(det.frame[clear], minlength=det.n_frames)


def _gap_lengths(seen: np.ndarray) -> np.ndarray:
    """Lengths in frames of each run of unseen frames between or around sightings."""
    edges = np.diff(np.r_[1, seen.astype(int), 1])
    return np.flatnonzero(edges == 1) - np.flatnonzero(edges == -1)


def coverage(det: Detections, tracks: MemberTracks, min_keypoint_conf: float) -> dict:
    seen = tracks.rows >= 0
    members = len(seen)
    everyone = seen.all(axis=0)
    gaps = {label: 0 for _, _, label in GAP_BINS_S}
    for member_seen in seen:
        lengths = _gap_lengths(member_seen) / det.fps
        for lo, hi, label in GAP_BINS_S:
            gaps[label] += int(((lengths >= lo) & (lengths < hi)).sum())
    missing = ~everyone
    detected = _clear_per_frame(det, min_keypoint_conf)
    unassigned = float((detected[missing] >= members).mean()) if missing.any() else 0.0
    return {
        "all_members_share": float(everyone.mean()),
        "members_seen_share": [float(s.mean()) for s in seen],
        "gaps": gaps,
        "missing_detector_share": (1.0 - unassigned) if missing.any() else 0.0,
        "missing_unassigned_share": unassigned,
    }
