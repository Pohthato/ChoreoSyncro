"""Per-member, per-section summary: average offset, worst moments, the body part most often off."""
from __future__ import annotations

import csv
import warnings
import json
from pathlib import Path

import numpy as np

from ..config import Config
from ..features.bones import BONE_NAMES, PART_OF_BONE, PARTS, mirror_name
from .measure import SyncResult
from .consensus import nanmedian_first_axis
from .sections import SectionTiming, section_timing


def make_sections(config: Config, duration_s: float) -> list[tuple[float, float, str]]:
    if config.sections.explicit:
        return [(float(a), float(b), str(name)) for a, b, name in config.sections.explicit]
    chunk = config.sections.chunk_s
    edges = list(np.arange(0.0, duration_s, chunk)) + [duration_s]
    if len(edges) > 2 and edges[-1] - edges[-2] < chunk / 2:
        del edges[-2]  # fold a short tail into the section before it
    return [(float(a), float(b), f"{a:.0f}-{b:.0f}s") for a, b in zip(edges[:-1], edges[1:])]


LABEL_SWAP_DEG = 150.0  # bone errors this large are flagged as a likely left/right label swap
MIN_EXTRA_SHARE = 0.1  # off at least 10 percentage points more often than the group's typical member


def _shares(counts: np.ndarray, totals: np.ndarray) -> dict:
    return {PARTS[p]: (float(counts[p] / totals[p]) if totals[p] else None) for p in range(len(PARTS))}


def _name_parts(blocks: list[dict], kind: str) -> None:
    """Name each member's body part most often off, *compared with the other members*.

    On real 2D footage some parts are noisy for everyone (arms, mostly). Judged alone, every
    member's arms come out "most often off", which says nothing about any one of them. So a
    part is named only if this member is off there at least MIN_EXTRA_SHARE more often than the
    group's median member is.
    """
    shares = np.array([[np.nan if b[f"{kind}_off_share_by_part"][p] is None else b[f"{kind}_off_share_by_part"][p] for p in PARTS] for b in blocks])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        typical = np.nanmedian(shares, axis=0)
    for b, row in zip(blocks, shares):
        extra = np.nan_to_num(row - typical, nan=-1.0)
        p = int(np.argmax(extra))
        named = extra[p] >= MIN_EXTRA_SHARE
        b[f"{kind}_part_most_often_off"] = PARTS[p] if named else None
        b[f"{kind}_part_off_share"] = float(row[p]) if named else None
        b[f"{kind}_part_off_share_group"] = float(typical[p]) if named else None


def _member_block(result: SyncResult, i: int, window_mask: np.ndarray, frame_mask: np.ndarray, config: Config,
                  group_part: np.ndarray, group_bone: np.ndarray) -> dict:
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
    block["timing_off_share_by_part"] = _shares((part_off & part_ok).sum(axis=1), part_ok.sum(axis=1))
    # Shape per body part, judged by the part's worst bone: one shin 30 degrees off is a leg
    # problem even though the other three leg bones are fine (an average would hide it).
    # And judged *against the group*: on real 2D footage camera perspective and personal styling
    # put every dancer's arms tens of degrees from the consensus, so an absolute threshold flags
    # everyone all the time. A part is "off" when the member's error is more than the threshold
    # above the group's typical error for that part in that frame.
    shape = worst_bone_by_part(result.bone_shape_deg[i][frame_mask])
    group = group_part[frame_mask]
    excess = shape - group
    known = np.isfinite(excess)
    off = np.nan_to_num(excess) > config.sync.shape_threshold_deg

    def mean_by_part(values):
        return {PARTS[p]: (float(np.nanmean(values[:, p])) if np.isfinite(values[:, p]).any() else None) for p in range(len(PARTS))}

    block["shape_error_by_part_deg"] = mean_by_part(shape)
    block["group_shape_error_by_part_deg"] = mean_by_part(np.where(np.isfinite(shape), group, np.nan))
    block["shape_excess_by_part_deg"] = mean_by_part(excess)
    block["shape_off_share_by_part"] = _shares((off & known).sum(axis=0), known.sum(axis=0))

    worst = []
    for k in np.argsort(-np.abs(offsets))[:3]:
        window = np.flatnonzero(ok)[k]
        parts = np.abs(np.nan_to_num(result.part_offset_ms[i, :, window]))
        parts[~result.part_reliable[i, :, window]] = -1
        worst.append({"time_s": round(float(times[k]), 2), "offset_ms": round(float(offsets[k]), 1),
                      "part_furthest_off": PARTS[int(np.argmax(parts))] if parts.max() >= 0 else None})
    block["worst_timing_moments"] = worst

    bone = result.bone_shape_deg[i] - group_bone  # how far above the group's typical, per bone
    bone[~frame_mask] = np.nan
    worst_shape = []
    if np.isfinite(bone).any():
        per_frame = np.nanmax(np.nan_to_num(bone, nan=-1e9), axis=1)
        frame_index = np.arange(len(per_frame))
        taken = []
        for f in np.argsort(-per_frame):
            if per_frame[f] < config.sync.shape_threshold_deg or len(worst_shape) == 3:
                break
            if any(abs(f - t) < result.fps for t in taken):  # one entry per second
                continue
            taken.append(f)
            b = int(np.nanargmax(np.nan_to_num(bone[f], nan=-1e9)))
            name = BONE_NAMES[b]
            if _mirrored_at(result, i, f):
                name = mirror_name(name)  # report in the member's own left/right
            worst_shape.append({"time_s": round(float(frame_index[f] / result.fps), 2), "bone": name,
                                "error_deg": round(float(result.bone_shape_deg[i, f, b]), 1), "above_group_deg": round(float(per_frame[f]), 1),
                                # A bone pointing almost straight against the group's is usually the pose
                                # model flipping that bone's labels; it can be real, so it is flagged, not hidden.
                                "likely_label_swap": bool(result.bone_shape_deg[i, f, b] >= LABEL_SWAP_DEG)})
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


def _r(value, digits=1):
    return None if value is None or not np.isfinite(value) else round(float(value), digits)


def summarize(result: SyncResult, config: Config, timing: SectionTiming | None = None) -> dict:
    """Two levels of timing, kept apart on purpose:
    - per section (`section_*`): a robust median over many windows with a 90% interval. This is
      the main answer on real footage.
    - per window (`mean_offset_ms`, `worst_timing_moments`...): only windows that pass the strict
      gates on their own. Precise moments, but on real dance they are rare.
    """
    duration = result.n_frames / result.fps
    sections = make_sections(config, duration)
    timing = timing or section_timing(result, sections, config)
    group_part = nanmedian_first_axis(np.stack([worst_bone_by_part(result.bone_shape_deg[i]) for i in range(len(result.names))]))
    group_bone = nanmedian_first_axis(result.bone_shape_deg)
    frames_t = np.arange(result.n_frames) / result.fps
    in_sync = config.render.in_sync_ms
    members = []
    for i, name in enumerate(result.names):
        overall = _member_block(result, i, np.ones(len(result.window_centers_s), bool), np.ones(result.n_frames, bool), config, group_part, group_bone)
        measured = timing.reliable[i]
        values = timing.offset_ms[i, measured]
        overall.update({
            "section_typical_offset_ms": _r(np.median(values)) if values.size else None,
            "sections_measured_share": round(float(measured.mean()), 3),
            "sections_in_sync_share": round(float((np.abs(values) <= in_sync).mean()), 3) if values.size else None,
        })
        overall["facing_differently_share"] = round(float(result.facing_differently[i].mean()), 3)
        entry = {"member": name, "overall": overall, "mirrored_share": float(result.mirrored[i].mean()), "sections": []}
        for s, (start, end, label) in enumerate(sections):
            wmask = (result.window_centers_s >= start) & (result.window_centers_s < end)
            fmask = (frames_t >= start) & (frames_t < end)
            block = {"section": label, "start_s": start, "end_s": end,
                     "section_offset_ms": _r(timing.offset_ms[i, s]) if timing.reliable[i, s] else None,
                     "section_interval_ms": [_r(timing.low_ms[i, s]), _r(timing.high_ms[i, s])] if timing.reliable[i, s] else None,
                     "section_reliable": bool(timing.reliable[i, s]),
                     # Kept even when not reliable, so a reader can see "measured but too uncertain".
                     "section_estimate_ms": _r(timing.offset_ms[i, s]),
                     "section_estimate_interval_ms": [_r(timing.low_ms[i, s]), _r(timing.high_ms[i, s])],
                     "section_windows": int(timing.windows[i, s]),
                     "section_blocks": int(timing.blocks[i, s]),
                     **_member_block(result, i, wmask, fmask, config, group_part, group_bone)}
            entry["sections"].append(block)
        members.append(entry)
    for kind in ("timing", "shape"):
        _name_parts([m["overall"] for m in members], kind)
        for s in range(len(sections)):
            _name_parts([m["sections"][s] for m in members], kind)
    return {"fps": result.fps, "duration_s": duration, "sign_convention": "+ ms = behind the group (median member), - = ahead",
            "in_sync_ms": config.render.in_sync_ms, "shape_threshold_deg": config.sync.shape_threshold_deg,
            "sections": [{"name": n, "start_s": a, "end_s": b} for a, b, n in sections], "members": members}


def write_summary(summary: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["member", "section", "section_offset_ms", "section_low_ms", "section_high_ms",
                         "strict_window_coverage", "strict_mean_offset_ms", "strict_mean_abs_offset_ms", "strict_in_sync_share",
                         "timing_part_most_often_off", "shape_part_most_often_off"] + [f"shape_{p}_deg" for p in PARTS])
        for member in summary["members"]:
            for block in [{"section": "overall", **member["overall"]}] + member["sections"]:
                def r(v, d=1):
                    return "" if v is None else round(v, d)
                interval = block.get("section_interval_ms") or [None, None]
                section_offset = block.get("section_offset_ms", block.get("section_typical_offset_ms"))
                writer.writerow([member["member"], block["section"], r(section_offset), r(interval[0]), r(interval[1]),
                                 r(block["coverage"], 2), r(block["mean_offset_ms"]),
                                 r(block["mean_abs_offset_ms"]), r(block["in_sync_share"], 2), block["timing_part_most_often_off"] or "",
                                 block["shape_part_most_often_off"] or ""] + [r(block["shape_error_by_part_deg"][p]) for p in PARTS])
