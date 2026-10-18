"""Skeletons only, on a dark background, coloured by how in sync each member is right now.

No source footage is drawn, only the derived keypoints, so this video can be published.
Timing comes from the member's current section (sync/sections.py), the reading that holds up on
real footage. Colour is a status with a text label next to it:
green = within the in-sync band, amber = drifting, red = off, grey = no reliable reading.
Bones whose pose is off by more than the shape threshold *beyond the group's typical error for
that bone at that moment* get a pink halo (an absolute threshold lights up most of a real group,
because 2D perspective alone puts everyone's arms well away from the consensus).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..features.bones import BONE_NAMES, BONES, mirror_name
from ..sync.measure import SyncResult
from ..sync.consensus import nanmedian_first_axis
from ..sync.sections import SectionTiming
from .style import CRITICAL, DARK_INK, DARK_INK_2, DARK_MUTED, DARK_SURFACE, GOOD, SHAPE_FLAG, WARNING, signed_ms


def _bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[k:k + 2], 16) for k in (0, 2, 4))
    return b, g, r


_EXTRA_EDGES = [(5, 11), (6, 12)]  # torso sides, drawn but not scored
_FLIP_BONE = np.array([BONE_NAMES.index(mirror_name(n)) for n in BONE_NAMES])


def _draw_labels(canvas: np.ndarray, labels: list, font: int) -> None:
    """Name and timing above each member. Dancers often stand close, so a label that would
    overlap one already placed is lifted to the next free line."""
    placed = []  # (x_left, x_right, line)
    for cx, top, name, detail in sorted(labels):
        widths = [cv2.getTextSize(t, font, s, 1)[0][0] for t, s in ((name, 0.6), (detail, 0.5))]
        half = max(widths) // 2 + 6
        line = 0
        while any(line == l and cx - half < r and cx + half > lft for lft, r, l in placed):
            line += 1
        placed.append((cx - half, cx + half, line))
        lift = 48 * line
        for text, dy, col, size, w in ((name, -38, DARK_INK, 0.6, widths[0]), (detail, -14, DARK_INK_2, 0.5, widths[1])):
            cv2.putText(canvas, text, (cx - w // 2, max(14, top + dy - lift)), font, size, _bgr(col), 1, cv2.LINE_AA)


def status(offset_ms: float, reliable: bool, in_sync: float, off_sync: float) -> tuple[str, str]:
    if not reliable or not np.isfinite(offset_ms):
        return DARK_MUTED, "no reading"
    size = abs(offset_ms)
    if size <= in_sync:
        return GOOD, "in sync"
    return (WARNING, "drifting") if size < off_sync else (CRITICAL, "off")


def render_skeleton_video(result: SyncResult, timing: SectionTiming, xy: np.ndarray, conf: np.ndarray, path: Path, image_size: tuple[int, int],
                          in_sync_ms: float, off_sync_ms: float, shape_threshold: float, min_conf: float,
                          width: int = 1280, frame_range: tuple[int, int] | None = None, gif_path: Path | None = None) -> None:
    src_w, src_h = image_size
    scale = width / src_w
    height = int(round(src_h * scale)) // 2 * 2
    footer = 56
    members, n_frames = xy.shape[:2]
    start, stop = frame_range or (0, n_frames)
    window_index = np.abs(np.arange(n_frames)[:, None] - result.window_centers_s[None, :] * result.fps).argmin(axis=1)
    frame_s = np.arange(n_frames) / result.fps
    section_index = np.clip(np.searchsorted(timing.starts_s, frame_s, side="right") - 1, 0, len(timing.starts_s) - 1)
    section_offset = np.where(timing.reliable, timing.offset_ms, np.nan)[:, section_index]  # (M, T)
    group_bone = nanmedian_first_axis(result.bone_shape_deg)  # (T, B) the group's typical error per bone
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), result.fps, (width, height + footer))
    gif_frames = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    thick = max(2, int(round(4 * scale)))

    legend = ((GOOD, f"in sync <= {in_sync_ms:.0f} ms"), (WARNING, "drifting"), (CRITICAL, f"off >= {off_sync_ms:.0f} ms"),
              (DARK_MUTED, "no reading"), (SHAPE_FLAG, f"halo: pose {shape_threshold:.0f} deg beyond the group"))
    needed = sum(30 + cv2.getTextSize(text, font, 0.5, 1)[0][0] for _, text in legend) + 16
    legend_size = min(0.5, 0.5 * (width - 16) / needed)

    for f in range(start, stop):
        canvas = np.full((height + footer, width, 3), _bgr(DARK_SURFACE), dtype=np.uint8)
        labels = []
        for i in range(members):
            pts = xy[i, f] * scale
            ok = (conf[i, f] >= min_conf) & np.isfinite(pts).all(axis=1)
            if ok.sum() < 4:
                continue
            k = window_index[f]
            off = section_offset[i, f]
            color_hex, label = status(off, bool(np.isfinite(off)), in_sync_ms, off_sync_ms)
            color = _bgr(color_hex)
            neck = (pts[5] + pts[6]) / 2
            pelvis = (pts[11] + pts[12]) / 2
            ext = np.vstack([pts, neck, pelvis])
            ext_ok = np.concatenate([ok, [ok[5] & ok[6], ok[11] & ok[12]]])
            shape = result.bone_shape_deg[i, f] - group_bone[f]
            if result.mirrored[i, k]:
                shape = shape[_FLIP_BONE]  # feature bones are mirrored; draw on the member's own limbs
            edges = [(b.start, b.end, shape[n]) for n, b in enumerate(BONES)] + [(a, b, np.nan) for a, b in _EXTRA_EDGES]
            for a, b, err in edges:
                if ext_ok[a] and ext_ok[b] and np.isfinite(err) and err > shape_threshold:
                    cv2.line(canvas, tuple(np.round(ext[a]).astype(int)), tuple(np.round(ext[b]).astype(int)), _bgr(SHAPE_FLAG), thick * 4, cv2.LINE_AA)
            for a, b, _ in edges:
                if ext_ok[a] and ext_ok[b]:
                    cv2.line(canvas, tuple(np.round(ext[a]).astype(int)), tuple(np.round(ext[b]).astype(int)), color, thick, cv2.LINE_AA)
            if ok[0]:
                cv2.circle(canvas, tuple(np.round(pts[0]).astype(int)), thick * 2, color, -1, cv2.LINE_AA)
            top = np.nanmin(pts[ok][:, 1])
            cx = int(np.round(np.nanmean(pts[ok][:, 0])))
            value = "--" if label == "no reading" else signed_ms(off)
            labels.append((cx, int(top), result.names[i], f"{value} {label}"))
        _draw_labels(canvas, labels, font)
        cv2.putText(canvas, f"{f / result.fps:6.2f} s", (16, 30), font, 0.6, _bgr(DARK_INK_2), 1, cv2.LINE_AA)
        note = "+ = behind the group, - = ahead"
        (nw, _), _ = cv2.getTextSize(note, font, 0.5, 1)
        cv2.putText(canvas, note, (width - nw - 16, 30), font, 0.5, _bgr(DARK_MUTED), 1, cv2.LINE_AA)
        x = 16
        y = height + footer // 2 + 6
        for color_hex, text in legend:
            cv2.circle(canvas, (x + 7, y - 5), 7, _bgr(color_hex), -1, cv2.LINE_AA)
            cv2.putText(canvas, text, (x + 20, y), font, legend_size, _bgr(DARK_INK_2), 1, cv2.LINE_AA)
            x += 30 + cv2.getTextSize(text, font, legend_size, 1)[0][0]
        writer.write(canvas)
        if gif_path is not None and (f - start) % 2 == 0:
            gif_frames.append(cv2.cvtColor(cv2.resize(canvas, (width // 2, (height + footer) // 2), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    writer.release()
    if gif_path is not None and gif_frames:
        from PIL import Image

        images = [Image.fromarray(frame) for frame in gif_frames]
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=int(2000 / result.fps), loop=0, optimize=True)
