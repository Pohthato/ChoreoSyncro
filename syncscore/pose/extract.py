"""Video -> person detections with keypoints and tracker IDs (Ultralytics YOLO11-pose + ByteTrack).

Why YOLO11-pose: it is one network that finds every person *and* their 17 COCO keypoints in a
single pass (bottom-up speed with top-down accuracy for well-separated dancers), it runs on a
laptop CPU, and ByteTrack is built into the same `track()` call. It is also the model family
of Optiqen (the badminton project), so this really is the same pipeline in a different domain.
RTMPose (MMPose) is somewhat more accurate on crowded scenes, but needs a separate detector and
the MMCV stack, which is hard to install on a Windows CPU.

Nothing here stores pixels. Per detection we keep the box, the keypoints, and a 32-number
colour histogram of the torso (for re-identification), all derived data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from ..config import Config
from .tracks import Detections

HIST_BINS = (8, 4)  # hue x saturation


def torso_histogram(frame: np.ndarray, xy: np.ndarray, kconf: np.ndarray, min_conf: float) -> np.ndarray:
    """Hue/saturation histogram of the torso (shoulders to hips), normalised; zeros if unseen."""
    corners = [5, 6, 11, 12]
    if (kconf[corners] < min_conf).any():
        return np.zeros(HIST_BINS[0] * HIST_BINS[1])
    pts = xy[corners]
    x1, y1 = np.floor(pts.min(axis=0)).astype(int)
    x2, y2 = np.ceil(pts.max(axis=0)).astype(int)
    h, w = frame.shape[:2]
    x1, x2, y1, y2 = max(x1, 0), min(x2, w), max(y1, 0), min(y2, h)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return np.zeros(HIST_BINS[0] * HIST_BINS[1])
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(HIST_BINS), [0, 180, 0, 256]).ravel()
    total = hist.sum()
    return hist / total if total > 0 else hist


def extract(video: str | Path, config: Config, n_frames: int, fps: float, size: tuple, progress: bool = True) -> Detections:
    from ultralytics import YOLO

    model = YOLO(config.pose.model)
    stream = model.track(source=str(video), stream=True, tracker=config.tracking.tracker, persist=True,
                         conf=config.pose.min_box_conf, device=config.pose.device, verbose=False)
    frame_idx, track_id, box, box_conf, xy, kconf, hist = [], [], [], [], [], [], []
    count = 0
    for f, result in enumerate(stream):
        count = f + 1
        boxes, keypoints = result.boxes, result.keypoints
        if boxes is None or len(boxes) == 0 or keypoints is None:
            continue
        ids = boxes.id.int().cpu().numpy() if boxes.id is not None else np.full(len(boxes), -1)
        k_xy = keypoints.xy.cpu().numpy()
        k_conf = keypoints.conf.cpu().numpy() if keypoints.conf is not None else np.ones(k_xy.shape[:2])
        for d in range(len(boxes)):
            frame_idx.append(f)
            track_id.append(int(ids[d]))
            box.append(boxes.xyxy[d].cpu().numpy())
            box_conf.append(float(boxes.conf[d]))
            xy.append(k_xy[d])
            kconf.append(k_conf[d])
            hist.append(torso_histogram(result.orig_img, k_xy[d], k_conf[d], config.pose.min_keypoint_conf))
        if progress and f % 50 == 0:
            print(f"\r  pose + tracking: frame {f}/{n_frames}", end="", file=sys.stderr, flush=True)
    if progress:
        print(file=sys.stderr)
    empty = len(frame_idx) == 0
    return Detections(
        frame=np.array(frame_idx, dtype=int), track_id=np.array(track_id, dtype=int),
        box=np.zeros((0, 4)) if empty else np.array(box, dtype=float), box_conf=np.array(box_conf, dtype=float),
        xy=np.zeros((0, 17, 2)) if empty else np.array(xy, dtype=float), kconf=np.zeros((0, 17)) if empty else np.array(kconf, dtype=float),
        hist=np.zeros((0, HIST_BINS[0] * HIST_BINS[1])) if empty else np.array(hist, dtype=float),
        fps=fps, n_frames=max(n_frames, count), image_size=tuple(size),
    )
