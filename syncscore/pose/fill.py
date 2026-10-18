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

And of an accepted candidate, only the member's *own* joints are kept. Looking at recovered
frames on the real clip showed the box usually on the right, half-hidden dancer, but with some
joints placed on the dancer in front (their knees, their arm). Those would make the hidden
member's timing partly the front dancer's. So a joint lying on another present member's joint
is dropped (confidence 0), and a candidate left with fewer than `MIN_OWN` joints is rejected.
"""
from __future__ import annotations

import numpy as np

MAX_JUMP = 0.5        # centre distance / expected box height
SIZE_RATIO = 1.3      # box height within x/÷ this of the expected height
MAX_IOU_OTHER = 0.7   # more overlap than this with a present member's box = that member
DUPLICATE_PX = 0.05   # median distance of shared confident keypoints / expected height
MIN_SHARED = 5        # confident keypoints both skeletons need before they can be compared
MIN_OWN = 6           # confident joints a candidate must keep after dropping borrowed ones


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


def own_keypoints(xy: np.ndarray, conf: np.ndarray, others_xy: np.ndarray, others_conf: np.ndarray,
                  height: float, min_keypoint_conf: float) -> np.ndarray:
    """conf with 0 for every joint lying within DUPLICATE_PX * height of a confident joint of
    another present member (any joint: a borrowed left knee can land on their right knee)."""
    kept = conf.copy()
    for o_xy, o_conf in zip(others_xy, others_conf):
        theirs = o_xy[o_conf >= min_keypoint_conf]
        if not len(theirs):
            continue
        nearest = np.min(np.linalg.norm(xy[:, None, :] - theirs[None], axis=2), axis=1)
        kept[np.nan_to_num(nearest, nan=np.inf) < DUPLICATE_PX * height] = 0.0
    return kept


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
    own = own_keypoints(candidate_xy, candidate_conf, others_xy, others_conf, height, min_keypoint_conf)
    if (own >= min_keypoint_conf).sum() < MIN_OWN:
        return np.inf
    return float(jump + 0.5 * abs(np.log(ratio)))


def fill_gaps(det, tracks, max_gap_s: float, min_keypoint_conf: float):
    """Fill members' short gaps with detections no member uses. Returns (new tracks, frames filled).

    Per frame, every (missing member, unused detection) pair is scored with `accept`, and pairs
    are taken cheapest first, so one detection fills at most one member and each member gets
    the detection nearest to where they should be.
    """
    from .tracks import MemberTracks

    max_gap = int(round(max_gap_s * det.fps))
    out = MemberTracks(tracks.xy.copy(), tracks.conf.copy(), tracks.box.copy(), tracks.hist.copy(),
                       tracks.fragments, tracks.rows.copy())
    if max_gap <= 0:
        return out, 0
    by_frame = np.split(np.argsort(det.frame, kind="stable"), np.cumsum(np.bincount(det.frame, minlength=det.n_frames))[:-1])
    filled = 0
    members = len(out.rows)
    for f in np.flatnonzero((out.rows < 0).any(axis=0)):
        missing = [i for i in range(members) if out.rows[i, f] < 0]
        expected = {i: expected_box(tracks.box, i, f, max_gap) for i in missing}
        missing = [i for i in missing if expected[i] is not None]
        candidates = [r for r in by_frame[f] if r not in set(out.rows[:, f])]
        if not missing or not candidates:
            continue
        present = [i for i in range(members) if out.rows[i, f] >= 0]
        o_box, o_xy, o_conf = out.box[present, f], out.xy[present, f], out.conf[present, f]
        pairs = sorted((accept(det.box[r], det.xy[r], det.kconf[r], expected[i], o_box, o_xy, o_conf, min_keypoint_conf), i, r)
                       for i in missing for r in candidates)
        used_members, used_rows = set(), set()
        for cost, i, r in pairs:
            if not np.isfinite(cost):
                break
            if i in used_members or r in used_rows:
                continue
            used_members.add(i)
            used_rows.add(r)
            own = own_keypoints(det.xy[r], det.kconf[r], o_xy, o_conf, float(expected[i][3] - expected[i][1]), min_keypoint_conf)
            # No clothing colour: in a half-hidden box the torso is mostly the front dancer's.
            out.xy[i, f], out.conf[i, f], out.box[i, f], out.hist[i, f] = det.xy[r], own, det.box[r], 0.0
            out.rows[i, f] = r
            filled += 1
    return out, filled
