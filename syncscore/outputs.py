"""Write every deliverable for one analysed group into an output folder."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import Config
from .sync.measure import SyncResult
from .sync.sections import section_timing
from .sync.summary import make_sections, summarize, write_summary
from .viz.heatmap import render_heatmap
from .viz.report_card import render_report_card
from .viz.skeleton_video import render_skeleton_video


RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{k}" for k in range(1, 10)), *(f"LPT{k}" for k in range(1, 10))}


def card_stems(names: list[str]) -> list[str]:
    """A file name (without .png) for each member's report card. Picked names are typed by the user,
    so characters Windows refuses become "_" and a name two members would share gets its number."""
    stems, seen = [], set()
    for i, name in enumerate(names):
        stem = "".join("_" if c in '<>:"/\\|?*' or ord(c) < 32 else c for c in str(name)).rstrip(". ")
        stem = stem or f"M{i + 1}"
        if stem.split(".")[0].upper() in RESERVED:
            stem = "_" + stem
        if stem.casefold() in seen:
            stem = f"{stem}-{i + 1}"
        seen.add(stem.casefold())
        stems.append(stem)
    return stems


def write_outputs(result: SyncResult, xy: np.ndarray, conf: np.ndarray, out: Path, config: Config, image_size: tuple[int, int], gif_seconds: float | None = None) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    result.save(out / "sync_result.npz")
    np.savez_compressed(out / "keypoints_smoothed.npz", xy=xy.astype(np.float32), conf=conf.astype(np.float32), fps=result.fps, names=np.array(result.names))
    sections = make_sections(config, result.n_frames / result.fps)
    timing = section_timing(result, sections, config)
    summary = summarize(result, config, timing)
    write_summary(summary, out)
    render_heatmap(timing, result.names, out / "heatmap.png")
    cards = out / "report_cards"
    cards.mkdir(exist_ok=True)
    stems = card_stems([member["member"] for member in summary["members"]])
    for i, member in enumerate(summary["members"]):
        render_report_card(result, timing, member, i, cards / f"{stems[i]}.png", config.render.in_sync_ms, config.sync.shape_threshold_deg)
    render_skeleton_video(result, timing, xy, conf, out / "skeletons.mp4", image_size, config.render.in_sync_ms, config.render.off_sync_ms,
                          config.sync.shape_threshold_deg, config.pose.min_keypoint_conf)
    if gif_seconds:
        gif = out / "skeletons_preview.gif"
        start = int(result.n_frames * 0.25)
        render_skeleton_video(result, timing, xy, conf, out / "_preview.mp4", image_size, config.render.in_sync_ms, config.render.off_sync_ms,
                              config.sync.shape_threshold_deg, config.pose.min_keypoint_conf, width=960,
                              frame_range=(start, min(result.n_frames, start + int(gif_seconds * result.fps))), gif_path=gif)
        (out / "_preview.mp4").unlink(missing_ok=True)
    return summary
