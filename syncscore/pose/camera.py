"""Follow the camera, so positions can be compared across a pan or a zoom.

A dancer's position on screen moves when she moves and when the camera does. To predict where a
hidden member will come back, the motion model needs where she is on the *stage*. So for every
frame we estimate a similarity transform (shift, uniform scale, small rotation) from that image
to the first frame of its segment, from background points tracked with optical flow.

- People are masked out (their boxes, padded), so dancers moving don't pull the estimate.
- A hard cut, or a frame whose background can't be matched, starts a new segment: positions in
  different segments are not comparable. Only when at least a tenth of the frame is background:
  when dancers fill the frame, the few pixels left say little.
- A plain backdrop with too few background points is taken as a still camera, not a cut. On
  aespa's plain studio the corners are weak and few, so the thresholds are low (see the walkthrough).
"""
from __future__ import annotations

import sys
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .tracks import Detections
from .video import read_frames

WORK_WIDTH = 640      # frames are scaled down to this width to estimate
MAX_CORNERS = 400
QUALITY = 0.001       # corners at least this share as strong as the strongest (a plain studio wall has weak ones)
MIN_POINTS = 12       # fewer tracked background points: assume the camera didn't move
MIN_INLIERS = 8       # fewer points agreeing on one transform: no fit...
LOST_POINTS = 40      # ...which means the background was lost (new segment) only if this many points were tracked
BOX_PAD = 0.1         # people's boxes are padded by this share of their size before masking
CUT_FACTOR = 4.0      # a background change this many times the recent median...
CUT_MIN = 20.0        # ...and at least this many grey levels is a hard cut
MIN_VISIBLE = 0.1     # a new segment needs at least this share of the frame to be background (a few pixels next to dancers mislead)
HISTORY = 30          # frames of background change the median is taken over

IDENTITY = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)


@dataclass
class CameraPath:
    transform: np.ndarray  # (T, 2, 3) image of frame f -> stage (the first frame of its segment)
    segment: np.ndarray    # (T,) segment number
    inliers: np.ndarray    # (T,) background points agreeing with the step from the previous frame

    def save(self, path, key: str) -> None:
        np.savez(path, transform=self.transform, segment=self.segment, inliers=self.inliers, key=key)

    @classmethod
    def load(cls, path, key: str) -> "CameraPath | None":
        if not Path(path).exists():
            return None
        with np.load(path) as data:
            if str(data["key"]) != key:
                return None
            return cls(data["transform"], data["segment"], data["inliers"])


def still_camera(n_frames: int) -> CameraPath:
    return CameraPath(np.tile(IDENTITY, (n_frames, 1, 1)), np.zeros(n_frames, np.int32), np.zeros(n_frames, np.int32))


def _background_mask(shape: tuple, boxes: np.ndarray, scale: float) -> np.ndarray:
    mask = np.full(shape, 255, np.uint8)
    for x1, y1, x2, y2 in np.asarray(boxes, float).reshape(-1, 4) * scale:
        px, py = BOX_PAD * (x2 - x1), BOX_PAD * (y2 - y1)
        mask[max(0, int(y1 - py)):max(0, int(np.ceil(y2 + py))), max(0, int(x1 - px)):max(0, int(np.ceil(x2 + px)))] = 0
    return mask


def _grey(frame: np.ndarray, scale: float) -> np.ndarray:
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    if scale != 1.0:
        grey = cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return grey


def _step(prev: np.ndarray, grey: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray | None, int]:
    """The transform from this frame to the previous one (2x3, working pixels). (None, inliers)
    when plenty of background can't be matched; (identity, 0) when there is too little to tell."""
    corners = cv2.goodFeaturesToTrack(prev, MAX_CORNERS, QUALITY, 8, mask=mask)
    if corners is None or len(corners) < MIN_POINTS:
        return IDENTITY.copy(), 0
    moved, status, _ = cv2.calcOpticalFlowPyrLK(prev, grey, corners, None)
    tracked = status.ravel() == 1
    if tracked.sum() < MIN_POINTS:
        return IDENTITY.copy(), 0
    step, inliers = cv2.estimateAffinePartial2D(moved[tracked], corners[tracked], method=cv2.RANSAC,
                                                ransacReprojThreshold=2.0)
    count = int(inliers.sum()) if inliers is not None else 0
    if step is None or count < MIN_INLIERS:
        return (None, count) if tracked.sum() >= LOST_POINTS else (IDENTITY.copy(), 0)
    return step.astype(np.float32), count


def estimate_frames(frames: Iterable[np.ndarray], boxes: list[np.ndarray]) -> CameraPath:
    """The camera path over `frames`; `boxes[f]` are the people in frame f (full-size pixels)."""
    transforms, segments, inliers = [], [], []
    prev = prev_boxes = None
    scale = 1.0
    diffs: deque = deque(maxlen=HISTORY)
    for f, frame in enumerate(frames):
        if prev is None:
            scale = min(1.0, WORK_WIDTH / frame.shape[1])
            prev, prev_boxes = _grey(frame, scale), boxes[f]
            transforms.append(IDENTITY.copy())
            segments.append(0)
            inliers.append(0)
            continue
        grey = _grey(frame, scale)
        mask = _background_mask(grey.shape, np.vstack([np.reshape(prev_boxes, (-1, 4)), np.reshape(boxes[f], (-1, 4))]), scale)
        visible = mask > 0
        diff = float(np.abs(grey.astype(np.float32) - prev)[visible].mean()) if visible.any() else 0.0
        judged = visible.mean() >= MIN_VISIBLE
        cut = judged and bool(diffs) and diff > max(CUT_MIN, CUT_FACTOR * float(np.median(diffs)))
        if judged:
            diffs.append(diff)
        step, count = (None, 0) if cut else _step(prev, grey, mask)
        if step is None and not judged:  # too little background to call it a new shot
            step, count = IDENTITY.copy(), 0
        if step is None:
            transforms.append(IDENTITY.copy())
            segments.append(segments[-1] + 1)
        else:
            step[:, 2] /= scale
            last = np.vstack([transforms[-1], [0, 0, 1]])
            transforms.append((last @ np.vstack([step, [0, 0, 1]]))[:2].astype(np.float32))
            segments.append(segments[-1])
        inliers.append(count)
        prev, prev_boxes = grey, boxes[f]
    return CameraPath(np.array(transforms, np.float32).reshape(-1, 2, 3), np.array(segments, np.int32),
                      np.array(inliers, np.int32))


def estimate_camera(video, det: Detections, progress: bool = True) -> CameraPath:
    """The camera path for every frame of `video`, decoded in order (frame numbers match `det`)."""
    order = np.argsort(det.frame, kind="stable")
    starts = np.searchsorted(det.frame[order], np.arange(det.n_frames + 1))
    boxes = [det.box[order[starts[f]:starts[f + 1]]] for f in range(det.n_frames)]
    cap = cv2.VideoCapture(str(video))

    def frames():
        for f, image in read_frames(cap, np.arange(det.n_frames)):
            if progress and f % 50 == 0:
                print(f"\r  following the camera: frame {f}/{det.n_frames}", end="", file=sys.stderr, flush=True)
            yield image

    try:
        path = estimate_frames(frames(), boxes)
    finally:
        cap.release()
    if progress:
        print(file=sys.stderr)
    missing = det.n_frames - len(path.segment)
    if missing > 0:  # the video decoded fewer frames than were detected: hold the last transform
        path = CameraPath(np.concatenate([path.transform, np.repeat(path.transform[-1:], missing, 0)]),
                          np.concatenate([path.segment, np.repeat(path.segment[-1:], missing)]),
                          np.concatenate([path.inliers, np.zeros(missing, np.int32)]))
    return path


def to_stage(camera: CameraPath, frames: np.ndarray, points: np.ndarray) -> np.ndarray:
    """(n, 2) image points seen in `frames` -> (n, 2) stage points."""
    t = camera.transform[np.asarray(frames, int)]
    p = np.asarray(points, float)
    return np.einsum("nij,nj->ni", t[:, :, :2], p) + t[:, :, 2]


def scale_at(camera: CameraPath, frames: np.ndarray) -> np.ndarray:
    """How many stage pixels one image pixel is in each frame."""
    a = camera.transform[np.asarray(frames, int)][:, :, :2]
    return np.sqrt(np.abs(a[:, 0, 0] * a[:, 1, 1] - a[:, 0, 1] * a[:, 1, 0]))
