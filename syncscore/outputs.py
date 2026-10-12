"""Write every deliverable for one analysed group into an output folder."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import Config
from .sync.measure import SyncResult
from .sync.summary import make_sections, summarize, write_summary
from .viz.heatmap import render_heatmap
from .viz.report_card import render_report_card
from .viz.skeleton_video import render_skeleton_video


def write_outputs(result: SyncResult, xy: np.ndarray, conf: np.ndarray, out: Path, config: Config, image_size: tuple[int, int], gif_seconds: float | None = None) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    result.save(out / "sync_result.npz")
    np.savez_compressed(out / "keypoints_smoothed.npz", xy=xy.astype(np.float32), conf=conf.astype(np.float32), fps=result.fps, names=np.array(result.names))
    summary = summarize(result, config)
    write_summary(summary, out)
    sections = make_sections(config, result.n_frames / result.fps)
    render_heatmap(result, sections, out / "heatmap.png")
    cards = out / "report_cards"
    cards.mkdir(exist_ok=True)
    for i, member in enumerate(summary["members"]):
        render_report_card(result, member, i, cards / f"{member['member']}.png", config.render.in_sync_ms, config.sync.shape_threshold_deg)
    render_skeleton_video(result, xy, conf, out / "skeletons.mp4", image_size, config.render.in_sync_ms, config.render.off_sync_ms,
                          config.sync.shape_threshold_deg, config.pose.min_keypoint_conf)
    if gif_seconds:
        gif = out / "skeletons_preview.gif"
        start = int(result.n_frames * 0.25)
        render_skeleton_video(result, xy, conf, out / "_preview.mp4", image_size, config.render.in_sync_ms, config.render.off_sync_ms,
                              config.sync.shape_threshold_deg, config.pose.min_keypoint_conf, width=960,
                              frame_range=(start, min(result.n_frames, start + int(gif_seconds * result.fps))), gif_path=gif)
        (out / "_preview.mp4").unlink(missing_ok=True)
    return summary
