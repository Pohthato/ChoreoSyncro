"""The shared path from keypoints to results, used by both the video and the synthetic benchmark."""
from __future__ import annotations

import numpy as np

from .config import Config
from .pose.smooth import smooth_keypoints
from .sync.measure import SyncResult, measure_sync


def smooth_group(xy: np.ndarray, conf: np.ndarray, fps: float, config: Config) -> tuple[np.ndarray, np.ndarray]:
    """Smooth every member's keypoints: (M, T, 17, 2), (M, T, 17)."""
    out_xy, out_conf = np.empty_like(xy), np.empty_like(conf)
    for i in range(len(xy)):
        out_xy[i], out_conf[i] = smooth_keypoints(xy[i], conf[i], fps, config.smoothing, config.pose.min_keypoint_conf)
    return out_xy, out_conf


def analyze_keypoints(xy: np.ndarray, conf: np.ndarray, fps: float, config: Config, names: list | None = None, smooth: bool = True) -> tuple[SyncResult, np.ndarray, np.ndarray]:
    """Smooth, then measure. Returns the result and the smoothed keypoints (for rendering)."""
    if smooth:
        xy, conf = smooth_group(xy, conf, fps, config)
    return measure_sync(xy, conf, fps, config, names), xy, conf
