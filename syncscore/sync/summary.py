"""Per-member, per-section summary: average offset, worst moments, the body part most often off."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..config import Config
from ..features.bones import BONE_NAMES, PART_OF_BONE, PARTS, mirror_name
from .measure import SyncResult


def make_sections(config: Config, duration_s: float) -> list[tuple[float, float, str]]:
    if config.sections.explicit:
        return [(float(a), float(b), str(name)) for a, b, name in config.sections.explicit]
    chunk = config.sections.chunk_s
    starts = np.arange(0.0, duration_s, chunk)
    return [(float(s), float(min(s + chunk, duration_s)), f"{s:.0f}-{min(s + chunk, duration_s):.0f}s") for s in starts]


MIN_OFF_SHARE = 0.1  # a part must be off in at least 10% of its measurements to be named


def _most_often(counts: np.ndarray, totals: np.ndarray) -> tuple[str | None, float]:
    share = np.where(totals > 0, counts / np.maximum(totals, 1), 0.0)
    if share.max(initial=0.0) < MIN_OFF_SHARE:
        return None, float(share.max(initial=0.0))
    p = int(np.argmax(share))
    return PARTS[p], float(share[p])


def _member_block(result: SyncResult, i: int, window_mask: np.ndarray, frame_mask: np.ndarray, config: Config) -> dict:
    in_sync = config.render.in_sync_ms
    ok = result.reliable[i] & window_mask
    offsets = result.offset_ms[i, ok]
    times = result.window_centers_s[ok]
    block = {
        "coverage": float(ok.sum() / max(window_mask.sum(), 1)),
        "mean_offset_ms": float(offsets.mean()) if offsets.size else None,
        "mean_abs_offset_ms": float(np.abs(offsets).mean()) if offsets.size else None,
        "consistency_sd_ms": float(offsets.std()) if offsets.size else None,
        "in_sync_share": float((np.abs(offsets) <= in_sync).mean()) if offsets.size else None,
    }
    # Timing per body part: how often each part on its own is outside the in-sync band.
    part_ok = result.part_reliable[i][:, window_mask]
    part_off = np.abs(np.nan_to_num(result.part_offset_ms[i][:, window_mask])) > in_sync
    part_mean = [float(np.nanmean(result.part_offset_ms[i, p][result.part_reliable[i, p] & window_mask])) if (result.part_reliable[i, p] & window_mask).any() else None for p in range(len(PARTS))]
    block["timing_offset_by_part_ms"] = dict(zip(PARTS, part_mean))
    block["timing_part_most_often_off"], block["timing_part_off_share"] = _most_often((part_off & part_ok).sum(axis=1), part_ok.sum(axis=1))
    # Shape per body part, judged by the part's worst bone: one shin 30 degrees off is a leg
    # problem even though the other three leg bones are fine (an average would hide it).
    shape = worst_bone_by_part(result.bone_shape_deg[i][frame_mask])
    known = np.isfinite(shape)
    off = np.nan_to_num(shape) > config.sync.shape_threshold_deg
    block["shape_error_by_part_deg"] = {PARTS[p]: (float(np.nanmean(shape[:, p])) if known[:, p].any() else None) for p in range(len(PARTS))}
    block["shape_part_most_often_off"], block["shape_part_off_share"] = _most_often((off & known).sum(axis=0), known.sum(axis=0))

    worst = []
    for k in np.argsort(-np.abs(offsets))[:3]:
        window = np.flatnonzero(ok)[k]
        parts = np.abs(np.nan_to_num(result.part_offset_ms[i, :, window]))
        parts[~result.part_reliable[i, :, window]] = -1
        worst.append({"time_s": round(float(times[k]), 2), "offset_ms": round(float(offsets[k]), 1),
                      "part_furthest_off": PARTS[int(np.argmax(parts))] if parts.max() >= 0 else None})
    block["worst_timing_moments"] = worst

    bone = result.bone_shape_deg[i].copy()
    bone[~frame_mask] = np.nan
    worst_shape = []
    if np.isfinite(bone).any():
        per_frame = np.nanmax(np.nan_to_num(bone, nan=-1), axis=1)
        frame_index = np.arange(len(per_frame))
        taken = []
        for f in np.argsort(-per_frame):
            if per_frame[f] < config.sync.shape_threshold_deg or len(worst_shape) == 3:
                break
            if any(abs(f - t) < result.fps for t in taken):  # one entry per second
                continue
            taken.append(f)
            b = int(np.nanargmax(np.nan_to_num(bone[f], nan=-1)))
            name = BONE_NAMES[b]
            if _mirrored_at(result, i, f):
                name = mirror_name(name)  # report in the member's own left/right
            worst_shape.append({"time_s": round(float(frame_index[f] / result.fps), 2), "bone": name, "error_deg": round(float(per_frame[f]), 1)})
    block["worst_shape_moments"] = worst_shape
    return block


def worst_bone_by_part(bone_deg: np.ndarray) -> np.ndarray:
    """(T, B) bone errors -> (T, P) error of the worst bone in each part (NaN if none known)."""
    out = np.full((len(bone_deg), len(PARTS)), np.nan)
    for p in range(len(PARTS)):
        cols = bone_deg[:, PART_OF_BONE == p]
        known = np.isfinite(cols).any(axis=1)
        out[known, p] = np.nanmax(cols[known], axis=1)
    return out


def _mirrored_at(result: SyncResult, i: int, frame: int) -> bool:
    k = int(np.argmin(np.abs(result.window_centers_s * result.fps - frame)))
    return bool(result.mirrored[i, k])


def summarize(result: SyncResult, config: Config) -> dict:
    duration = result.n_frames / result.fps
    sections = make_sections(config, duration)
    frames_t = np.arange(result.n_frames) / result.fps
    members = []
    for i, name in enumerate(result.names):
        entry = {"member": name, "overall": _member_block(result, i, np.ones(len(result.window_centers_s), bool), np.ones(result.n_frames, bool), config),
                 "mirrored_share": float(result.mirrored[i].mean()), "sections": []}
        for start, end, label in sections:
            wmask = (result.window_centers_s >= start) & (result.window_centers_s < end)
            fmask = (frames_t >= start) & (frames_t < end)
            entry["sections"].append({"section": label, "start_s": start, "end_s": end, **_member_block(result, i, wmask, fmask, config)})
        members.append(entry)
    return {"fps": result.fps, "duration_s": duration, "sign_convention": "+ ms = behind the group (median member), - = ahead",
            "in_sync_ms": config.render.in_sync_ms, "shape_threshold_deg": config.sync.shape_threshold_deg,
            "sections": [{"name": n, "start_s": a, "end_s": b} for a, b, n in sections], "members": members}


def write_summary(summary: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["member", "section", "coverage", "mean_offset_ms", "mean_abs_offset_ms", "in_sync_share",
                         "timing_part_most_often_off", "shape_part_most_often_off"] + [f"shape_{p}_deg" for p in PARTS])
        for member in summary["members"]:
            for block in [{"section": "overall", **member["overall"]}] + member["sections"]:
                def r(v, d=1):
                    return "" if v is None else round(v, d)
                writer.writerow([member["member"], block["section"], r(block["coverage"], 2), r(block["mean_offset_ms"]),
                                 r(block["mean_abs_offset_ms"]), r(block["in_sync_share"], 2), block["timing_part_most_often_off"] or "",
                                 block["shape_part_most_often_off"] or ""] + [r(block["shape_error_by_part_deg"][p]) for p in PARTS])
