"""Video facts we must get right: frame rate, size and the real timestamp of every frame.

Phone recordings are often variable frame rate: frames are not evenly spaced in time. If we
assumed frame k sits at k / fps, timing offsets would pick up the recording's own jitter. So we
read each frame's presentation timestamp and, if spacing is uneven, resample the keypoints onto
an even clock before measuring anything.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoInfo:
    fps: float
    n_frames: int
    size: tuple       # (width, height)
    times_ms: np.ndarray  # (n_frames,) presentation time of each decoded frame
    variable: bool    # True if frame spacing is uneven enough to need resampling


def probe(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    nominal = cap.get(cv2.CAP_PROP_FPS) or 30.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    times = []
    while cap.grab():  # decode headers only, no pixels kept
        times.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    cap.release()
    times = np.asarray(times, dtype=float)
    if len(times) > 2 and np.all(np.diff(times) > 0):
        fps = 1000.0 / float(np.median(np.diff(times)))
        uniform = times[0] + np.arange(len(times)) * 1000.0 / fps
        variable = bool(np.max(np.abs(times - uniform)) > 0.25 * 1000.0 / fps)
    else:  # container gave no usable timestamps: trust the nominal rate
        fps, variable = float(nominal), False
        times = np.arange(len(times)) * 1000.0 / fps
    return VideoInfo(fps=fps, n_frames=len(times), size=size, times_ms=times, variable=variable)


def read_frame(video, frame: int, times_ms: np.ndarray) -> np.ndarray | None:
    """Frame `frame` of `video` (BGR), found by its timestamp in `times_ms` (from `probe`); None
    when the video has no such frame.

    Seeking lands on a keyframe near the asked time, often before it and sometimes past it. So it
    seeks a little early and decodes on until the timestamp is the frame's own; if it still landed
    past the frame it seeks from further back, and last of all reads from the start."""
    times_ms = np.asarray(times_ms, dtype=float)
    if not 0 <= frame < len(times_ms):
        return None
    target = times_ms[frame]
    half = 0.5 * float(np.median(np.diff(times_ms))) if len(times_ms) > 1 else 1.0
    for preroll in (2000.0, 8000.0, None):
        cap = cv2.VideoCapture(str(video))
        try:
            if not cap.isOpened():
                return None
            if preroll is not None:
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, target - preroll))
            if not cap.grab():
                continue
            t = cap.get(cv2.CAP_PROP_POS_MSEC)
            if t > target + half:  # landed past it: back off further
                continue
            while t < target - half:
                if not cap.grab():
                    return None
                t = cap.get(cv2.CAP_PROP_POS_MSEC)
            ok, image = cap.retrieve()
            return image if ok else None
        finally:
            cap.release()
    return None


def resample_uniform(xy: np.ndarray, conf: np.ndarray, times_ms: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Member keypoints sampled at uneven times -> an even clock at `fps`.

    xy (M, T, 17, 2), conf (M, T, 17). Positions are interpolated linearly between the two
    nearest real frames, but only when both are known. Confidence is the lower of the two.
    """
    t = times_ms - times_ms[0]
    grid = np.arange(0.0, t[-1] + 1e-6, 1000.0 / fps)
    hi = np.clip(np.searchsorted(t, grid, side="left"), 1, len(t) - 1)
    lo = hi - 1
    w = ((grid - t[lo]) / np.maximum(t[hi] - t[lo], 1e-9))[None, :, None, None]
    out_xy = (1 - w) * xy[:, lo] + w * xy[:, hi]
    out_conf = np.minimum(conf[:, lo], conf[:, hi])
    return out_xy, out_conf, grid


def read_frames(cap, frames: np.ndarray) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame index, image) for exactly `frames`, decoding the video in order.

    Seeking by index (CAP_PROP_POS_FRAMES) turns the index into a time with the *average* frame
    rate, so on variable-frame-rate phone video it can land a frame or more away, and that pose
    would be scored as the wrong moment. extract() numbers frames in decode order, so this does
    too: grab every frame, decode only the wanted ones."""
    wanted = set(int(f) for f in frames)
    last = max(wanted, default=-1)
    index = -1
    while index < last and cap.grab():
        index += 1
        if index in wanted:
            ok, image = cap.retrieve()
            if ok:
                yield index, image
