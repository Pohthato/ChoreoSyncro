"""Before/after plot of the keypoint smoothing for one joint: position and speed."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .style import INK_2, SERIES, apply_style


def pick_window(raw: np.ndarray, fps: float, seconds: float = 6.0) -> tuple[int, int]:
    """The stretch with the most motion and no gaps, so fast moves are on show."""
    length = int(seconds * fps)
    if len(raw) <= length:
        return 0, len(raw)
    speed = np.nan_to_num(np.linalg.norm(np.diff(raw, axis=0), axis=-1), nan=-1e6)
    total = np.convolve(speed, np.ones(length), mode="valid")
    start = int(np.argmax(total))
    return start, start + length


def render_smoothing_plot(raw: np.ndarray, smooth: np.ndarray, fps: float, path: Path, title: str) -> None:
    """raw, smooth: (T, 2) pixel positions of one keypoint."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_style()
    start, stop = pick_window(raw, fps)
    t = np.arange(start, stop) / fps
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
    top.plot(t, raw[start:stop, 0], color=SERIES[1], lw=1.2, label="raw keypoints")
    top.plot(t, smooth[start:stop, 0], color=SERIES[0], lw=2, label="One Euro, forward + backward")
    top.set_ylabel("x position (px)")
    top.set_title(title, loc="left")
    top.legend(frameon=False, loc="upper left")
    speed = lambda p: np.linalg.norm(np.diff(p[start:stop + 1], axis=0), axis=-1) * fps
    bottom.plot(t, speed(raw)[: len(t)], color=SERIES[1], lw=1.2, label="raw")
    bottom.plot(t, speed(smooth)[: len(t)], color=SERIES[0], lw=2, label="smoothed")
    bottom.set_ylabel("speed (px/s)")
    bottom.set_xlabel("time (s)")
    bottom.set_title("Speed: jitter is removed, the fast peaks stay", loc="left", color=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
