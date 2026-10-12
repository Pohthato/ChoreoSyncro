"""Video facts we must get right: frame rate, size and the real timestamp of every frame.

Phone recordings are often variable frame rate: frames are not evenly spaced in time. If we
assumed frame k sits at k / fps, timing offsets would pick up the recording's own jitter. So we
read each frame's presentation timestamp and, if spacing is uneven, resample the keypoints onto
an even clock before measuring anything.
"""
from __future__ import annotations

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
