"""Where each person is heading, and splitting tracks that the tracker handed to someone else.

Positions are hip centres on the stage (see camera.py), measured in body heights where it
matters, so the same rules work for a wide shot and a close one.

- `Track` is a constant-velocity Kalman filter: position and velocity, so heading is part of
  the state. While a member is unseen it predicts where she is going and its doubt grows.
- `split_at_crossings` looks only where two people's boxes overlap. ByteTrack sometimes hands
  an ID from one dancer to the other there. The track is split when its fingerprint before and
  after differ, or when after the crossing it carries on the *other* dancer's heading. A dancer
  turning around on her own is never split, because nobody overlaps her.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from ..config import Config
from .camera import CameraPath, scale_at, to_stage
from .tracks import Detections, Tracklet, _iou

ACCEL_STD = 2.0        # body heights per s²: how fast a dancer can change speed
MEASURE_STD = 0.05     # body heights: how far a hip centre measurement can be off
WINDOW_S = 0.5         # seconds before and after a crossing that are compared
HEADING_MARGIN = 0.5   # after a crossing, the track must follow the other's heading by this much more than its own
MIN_SPEED = 0.3        # body heights per s: slower than this, heading says nothing
START_SPEED_STD = 2.0  # body heights per s: what the filter assumes about a piece's speed before it has seen it move


def positions(det: Detections, rows: np.ndarray, camera: CameraPath, min_conf: float) -> tuple[np.ndarray, np.ndarray]:
    """Stage hip centres (n, 2), the box centre where a hip is unsure, and body heights (n,) in stage pixels."""
    rows = np.asarray(rows, int)
    hips = det.xy[rows][:, [11, 12]].mean(axis=1)
    sure = (det.kconf[rows][:, [11, 12]] >= min_conf).all(axis=1)
    box = det.box[rows]
    centre = np.stack([(box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2], 1)
    points = np.where(sure[:, None], hips, centre)
    frames = det.frame[rows]
    return to_stage(camera, frames, points), (box[:, 3] - box[:, 1]) * scale_at(camera, frames)


class Track:
    """Constant-velocity Kalman filter over one person's stage position, in stage pixels."""

    def __init__(self, point: np.ndarray, frame: int, height: float, fps: float):
        self.fps = fps
        self.q = (ACCEL_STD * height) ** 2
        self.r = (MEASURE_STD * height) ** 2
        self.x = np.array([point[0], point[1], 0.0, 0.0])
        self.P = np.diag([self.r, self.r, (START_SPEED_STD * height) ** 2, (START_SPEED_STD * height) ** 2])
        self.frame = int(frame)

    @classmethod
    def follow(cls, points: np.ndarray, frames: np.ndarray, height: float, fps: float) -> "Track":
        track = cls(points[0], frames[0], height, fps)
        for point, frame in zip(points[1:], frames[1:]):
            track.predict(int(frame) - track.frame)
            track.update(point)
        return track

    def predict(self, frames: int) -> "Track":
        dt = frames / self.fps
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        Q = self.q * np.array([[dt ** 3 / 3, 0, dt ** 2 / 2, 0], [0, dt ** 3 / 3, 0, dt ** 2 / 2],
                               [dt ** 2 / 2, 0, dt, 0], [0, dt ** 2 / 2, 0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.frame += frames
        return self

    def update(self, point: np.ndarray) -> "Track":
        S = self.P[:2, :2] + self.r * np.eye(2)
        K = self.P[:, :2] @ np.linalg.inv(S)
        self.x = self.x + K @ (np.asarray(point, float) - self.x[:2])
        self.P = self.P - K @ self.P[:2, :]
        return self

    def distance(self, point: np.ndarray) -> float:
        """Mahalanobis distance from where the track expects to be measured."""
        d = np.asarray(point, float) - self.x[:2]
        S = self.P[:2, :2] + self.r * np.eye(2)
        return float(np.sqrt(d @ np.linalg.solve(S, d)))

    @property
    def position(self) -> np.ndarray:
        return self.x[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:].copy()


@dataclass
class Split:
    frame: int
    track_id: int
    reason: str  # "fingerprint" or "heading"


def _velocity(points: np.ndarray, frames: np.ndarray, fps: float) -> np.ndarray:
    """Least-squares velocity (stage pixels per s)."""
    t = (frames - frames.mean()) / fps
    return (t[:, None] * (points - points.mean(axis=0))).sum(axis=0) / max(float((t ** 2).sum()), 1e-9)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def _mean_look(vectors: np.ndarray) -> np.ndarray | None:
    seen = vectors[np.any(vectors != 0, axis=1)]
    return seen.mean(axis=0) if len(seen) else None


def _events(frames: np.ndarray, crossing: np.ndarray) -> list[tuple[int, int]]:
    """Runs of crossing frames, merging runs at most 2 frames apart."""
    out: list[list[int]] = []
    for f in frames[crossing]:
        if out and f - out[-1][1] <= 3:
            out[-1][1] = int(f)
        else:
            out.append([int(f), int(f)])
    return [(a, b) for a, b in out]


def split_at_crossings(det: Detections, tracklets: list[Tracklet], camera: CameraPath,
                       config: Config) -> tuple[list[Tracklet], list[Split]]:
    min_frames = max(3, int(0.3 * det.fps))
    window = round(WINDOW_S * det.fps)
    order = np.argsort(det.frame, kind="stable")
    starts = np.searchsorted(det.frame[order], np.arange(det.n_frames + 1))
    points, heights = positions(det, np.arange(len(det.frame)), camera, config.pose.min_keypoint_conf)

    def overlap(row: int) -> tuple[float, int | None]:
        f = det.frame[row]
        others = order[starts[f]:starts[f + 1]]
        others = others[others != row]
        if not len(others):
            return 0.0, None
        iou = np.nan_to_num(_iou(det.box[row][None], det.box[others]))
        best = int(np.argmax(iou))
        return float(iou[best]), int(others[best])

    pieces, splits = [], []
    for tracklet in tracklets:
        touching = [overlap(r) for r in tracklet.rows]
        crossing = np.array([iou > config.tracking.crossing_iou for iou, _ in touching])
        partner = {int(f): other for f, (_, other) in zip(tracklet.frames, touching)}
        current = tracklet.rows
        for a, b in _events(tracklet.frames, crossing):
            frames = det.frame[current]
            before = current[(frames >= a - window) & (frames < a)]
            after = current[(frames > b) & (frames <= b + window)]
            mid = (a + b) // 2
            if len(before) < 3 or len(after) < 3:
                continue
            if (frames <= mid).sum() < min_frames or (frames > mid).sum() < min_frames:
                continue
            reason = None
            if det.vectors is not None:
                look_before, look_after = _mean_look(det.vectors[before]), _mean_look(det.vectors[after])
                if look_before is not None and look_after is not None and \
                        1 - _cos(look_before, look_after) > config.identity.split_distance:
                    reason = "fingerprint"
            if reason is None:
                ids = Counter(int(det.track_id[partner[f]]) for f in range(a, b + 1)
                              if partner.get(f) is not None and det.track_id[partner[f]] >= 0)
                if ids:
                    other_id = ids.most_common(1)[0][0]
                    theirs = np.flatnonzero((det.track_id == other_id) & (det.frame >= a - window) & (det.frame < a))
                    if len(theirs) >= 3:
                        own_v = _velocity(points[before], det.frame[before], det.fps)
                        other_v = _velocity(points[theirs], det.frame[theirs], det.fps)
                        after_v = _velocity(points[after], det.frame[after], det.fps)
                        slowest = MIN_SPEED * float(np.median(heights[current]))
                        if min(np.linalg.norm(own_v), np.linalg.norm(other_v)) >= slowest and _cos(own_v, other_v) < 0 \
                                and _cos(after_v, other_v) - _cos(after_v, own_v) > HEADING_MARGIN:
                            reason = "heading"
            if reason is not None:
                splits.append(Split(int(mid), tracklet.track_id, reason))
                pieces.append(current[frames <= mid])
                current = current[frames > mid]
        pieces.append(current)
    out = [Tracklet(int(det.track_id[rows[0]]), rows, det.frame[rows], det.hist[rows].mean(axis=0)) for rows in pieces]
    return out, splits
