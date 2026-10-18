"""A second, more sensitive look at only the frames where a member is missing.

On the BTS clip, when a member was missing the detector itself had found fewer people 73% of
the time: a dancer half hidden behind another scores below the normal box threshold. Running the
model again on *every* frame at a lower threshold would mostly add noise (reflections, crew,
duplicate boxes). So it runs only on frames inside a member's short gap, where we know roughly
where they should be, and its detections then go through the same acceptance rule as any other
gap-filler (fill.py). They get no tracker ID (-1).

Cached in `recovered.npz` with the frame list, model, threshold and input size, and reused only
when all four match.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ..config import Config
from .fill import expected_box
from .tracks import Detections, MemberTracks


def frames_to_recover(tracks: MemberTracks, fps: float, max_gap_s: float) -> np.ndarray:
    """Frames where some member is unseen inside a gap short enough to fill."""
    max_gap = int(round(max_gap_s * fps))
    if max_gap <= 0:
        return np.zeros(0, dtype=int)
    frames = set()
    for i in range(len(tracks.box)):
        for f in np.flatnonzero(~np.isfinite(tracks.box[i, :, 0])):
            if expected_box(tracks.box, i, int(f), max_gap) is not None:
                frames.add(int(f))
    return np.array(sorted(frames), dtype=int)


def merge(det: Detections, extra: Detections) -> Detections:
    """Extra detections appended after the existing rows, so existing row indices stay valid."""
    return Detections(
        frame=np.concatenate([det.frame, extra.frame]), track_id=np.concatenate([det.track_id, extra.track_id]),
        box=np.concatenate([det.box, extra.box]), box_conf=np.concatenate([det.box_conf, extra.box_conf]),
        xy=np.concatenate([det.xy, extra.xy]), kconf=np.concatenate([det.kconf, extra.kconf]),
        hist=np.concatenate([det.hist, extra.hist]), fps=det.fps, n_frames=det.n_frames, image_size=det.image_size,
    )


def _settings(config: Config) -> np.ndarray:
    t = config.tracking
    return np.array([config.pose.model, str(t.recover_conf), str(t.recover_imgsz)])


def _save_cache(path: Path, det: Detections, frames: np.ndarray, config: Config) -> None:
    det.save(path)
    with np.load(path) as data:
        arrays = dict(data)
    np.savez_compressed(path, **arrays, recover_frames=frames, recover_settings=_settings(config))


def _cache_matches(path: Path, frames: np.ndarray, config: Config) -> bool:
    if not Path(path).exists():
        return False
    with np.load(path) as data:
        if "recover_frames" not in data or "recover_settings" not in data:
            return False
        return np.array_equal(data["recover_frames"], frames) and np.array_equal(data["recover_settings"], _settings(config))


def redetect(video: str | Path, frames: np.ndarray, config: Config, cache: Path, size: tuple, fps: float, n_frames: int) -> Detections:
    """Run the pose model on exactly `frames` at config.tracking.recover_conf / recover_imgsz."""
    cache = Path(cache)
    if _cache_matches(cache, frames, config):
        return Detections.load(cache)
    import cv2
    from ultralytics import YOLO

    from .extract import HIST_BINS, torso_histogram

    model = YOLO(config.pose.model)
    cap = cv2.VideoCapture(str(video))
    rows = {"frame": [], "box": [], "box_conf": [], "xy": [], "kconf": [], "hist": []}
    for n, f in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, image = cap.read()
        if not ok:
            continue
        result = model.predict(image, conf=config.tracking.recover_conf, imgsz=config.tracking.recover_imgsz,
                               device=config.pose.device, verbose=False)[0]
        if result.boxes is None or result.keypoints is None:
            continue
        k_xy = result.keypoints.xy.cpu().numpy()
        k_conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else np.ones(k_xy.shape[:2])
        for d in range(len(result.boxes)):
            rows["frame"].append(int(f))
            rows["box"].append(result.boxes.xyxy[d].cpu().numpy())
            rows["box_conf"].append(float(result.boxes.conf[d]))
            rows["xy"].append(k_xy[d])
            rows["kconf"].append(k_conf[d])
            rows["hist"].append(torso_histogram(image, k_xy[d], k_conf[d], config.pose.min_keypoint_conf))
        if n % 50 == 0:
            print(f"\r  re-detecting missed frames: {n}/{len(frames)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    count = len(rows["frame"])
    bins = HIST_BINS[0] * HIST_BINS[1]
    det = Detections(
        frame=np.array(rows["frame"], dtype=int), track_id=np.full(count, -1, dtype=int),
        box=np.array(rows["box"], dtype=float).reshape(count, 4), box_conf=np.array(rows["box_conf"], dtype=float),
        xy=np.array(rows["xy"], dtype=float).reshape(count, 17, 2), kconf=np.array(rows["kconf"], dtype=float).reshape(count, 17),
        hist=np.array(rows["hist"], dtype=float).reshape(count, bins), fps=fps, n_frames=n_frames, image_size=tuple(size),
    )
    _save_cache(cache, det, frames, config)
    return det
