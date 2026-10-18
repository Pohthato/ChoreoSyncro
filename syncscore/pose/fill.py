"""Filling a member's short gap with a real detection, and only a real one.

Most member gaps on real practice footage are short (on the BTS clip 266 of 339 were under
0.5 s): one dancer passes behind another, or a detection was too brief to become a tracklet.
A gap is filled frame by frame, but only with a detection the pose model actually produced.
Nothing is interpolated into the keypoints: an invented pose would be scored as if it were
measured, which is exactly what the tool must never do.

A candidate is accepted for a member's gap only if it is
- near where the member should be: the box interpolated between the last sighting before the
  gap and the first after it (gaps touching the clip's start or end have no such box, and are
  never filled),
- about the member's size (a reflection or a crew member at another depth is not),
- not another member who is already present in that frame,
- not a copy of another member's skeleton: a box drawn around a half-hidden dancer often
  carries the front dancer's keypoints.
"""
from __future__ import annotations

import numpy as np

MAX_JUMP = 0.5        # centre distance / expected box height
SIZE_RATIO = 1.3      # box height within x/÷ this of the expected height
MAX_IOU_OTHER = 0.7   # more overlap than this with a present member's box = that member
DUPLICATE_PX = 0.05   # median distance of shared confident keypoints / expected height
MIN_SHARED = 5        # confident keypoints both skeletons need before they can be compared


def _height(box: np.ndarray) -> float:
    return float(box[3] - box[1])


def _center(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(w, 0.0) * max(h, 0.0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def expected_box(box: np.ndarray, member: int, frame: int, max_gap: int) -> np.ndarray | None:
    """Where `member` should be at `frame`, if it lies inside a gap of at most `max_gap` unseen
    frames with a sighting on both sides. box is (M, T, 4), NaN where unseen."""
    seen = np.isfinite(box[member, :, 0])
    if seen[frame]:
        return None
    before = np.flatnonzero(seen[:frame])
    after = np.flatnonzero(seen[frame + 1:])
    if not before.size or not after.size:
        return None
    a, b = int(before[-1]), int(frame + 1 + after[0])
    if b - a - 1 > max_gap:
        return None
    w = (frame - a) / (b - a)
    return (1 - w) * box[member, a] + w * box[member, b]


def accept(candidate_box: np.ndarray, candidate_xy: np.ndarray, candidate_conf: np.ndarray, expected: np.ndarray,
           others_box: np.ndarray, others_xy: np.ndarray, others_conf: np.ndarray, min_keypoint_conf: float) -> float:
    """Cost of filling the gap with this candidate (lower is better), np.inf if it is rejected."""
    height = _height(expected)
    if height <= 0:
        return np.inf
    jump = np.linalg.norm(_center(candidate_box) - _center(expected)) / height
    ratio = _height(candidate_box) / height
    if jump > MAX_JUMP or not (1 / SIZE_RATIO <= ratio <= SIZE_RATIO):
        return np.inf
    for box, xy, conf in zip(others_box, others_xy, others_conf):
        if _iou(candidate_box, box) > MAX_IOU_OTHER:
            return np.inf
        shared = (candidate_conf >= min_keypoint_conf) & (conf >= min_keypoint_conf)
        if shared.sum() >= MIN_SHARED:
            distance = np.median(np.linalg.norm(candidate_xy[shared] - xy[shared], axis=1))
            if distance / height < DUPLICATE_PX:
                return np.inf
    return float(jump + 0.5 * abs(np.log(ratio)))
