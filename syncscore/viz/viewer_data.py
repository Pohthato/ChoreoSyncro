"""Everything the viewer page draws, computed here and written to one script file.

The page (syncscore/viz/viewer/) does no analysis: skeletons, ghosts, halos and section status
all come from the analysis outputs (`keypoints_smoothed.npz`, `sync_result.npz`,
`summary.json`), so the page shows exactly what the report says. The data is written as
`window.SYNCSCORE = {...};` so the page opens straight from disk (`file://`), where browsers
won't let a page fetch a JSON file next to it.

Large arrays are base64: keypoints as little-endian int16 in quarter pixels (NaN = -32768),
confidences as uint8, halos as a uint16 bit per bone. A 7-member, 3:50 clip is about 10 MB.

No pixels are stored. The video is referenced by its local path and never copied.
"""
from __future__ import annotations

import base64
import json
import shutil
import warnings
from pathlib import Path

import numpy as np

from ..config import Config
from ..features.bones import BONE_NAMES, bone_directions, mirror_name
from ..sync.consensus import nanmedian_first_axis
from ..sync.landings import clean_landings, landing_lock
from ..sync.timing import velocity
from ..sync.unison import apart_intervals, member_speed
from .ghost import ghost_keypoints

VIEWER_DIR = Path(__file__).resolve().parent / "viewer"
_MISSING = -32768
_FLIP_BONE = np.array([BONE_NAMES.index(mirror_name(n)) for n in BONE_NAMES])
# A bone pointing this far or more from the group's is listed separately, not as a pose moment.
# On the BTS clip pose differences over the threshold have a median of 40 degrees, while left/right
# label swaps and facing differences sit at 120-180; a cut at 150 only let the next tier of the
# same artefacts (141-149) fill every member's list.
VERY_DIFFERENT_DEG = 90.0


def encode_int16(values: np.ndarray, scale: float = 4.0) -> str:
    scaled = np.clip(np.nan_to_num(values * scale, nan=0.0), -32767, 32767)
    packed = np.where(np.isfinite(values), np.round(scaled), _MISSING).astype("<i2")
    return base64.b64encode(packed.tobytes()).decode("ascii")


def decode_int16(text: str, shape: tuple, scale: float = 4.0) -> np.ndarray:
    raw = np.frombuffer(base64.b64decode(text), dtype="<i2").reshape(shape)
    return np.where(raw == _MISSING, np.nan, raw / scale)


def _b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def video_url(path: Path) -> str:
    """Absolute file:/// URL of a local video, percent-encoded (spaces, non-Latin titles)."""
    return Path(path).resolve().as_uri()


def measured_conf(conf: np.ndarray, min_conf: float) -> np.ndarray:
    """conf where the joint was actually detected, 0 elsewhere.

    Smoothing fills gaps of a few frames by interpolation and gives every filled joint exactly the
    threshold confidence (pose/smooth.py). The analysis may use those, but the page must not draw
    them: half a second of straight-line motion at quarter speed looks measured and isn't."""
    filled = np.isclose(conf, np.float32(min_conf), rtol=0.0, atol=1e-6)
    return np.where((conf >= min_conf) & ~filled, conf, 0.0)


def _per_frame_mirrored(sync: dict, n_frames: int) -> np.ndarray:
    centers = sync["window_centers_s"] * float(sync["fps"])
    nearest = np.abs(np.arange(n_frames)[:, None] - centers[None, :]).argmin(axis=1)
    return sync["mirrored"][:, nearest].astype(bool)


def _halo(bone_deg: np.ndarray, mirrored: np.ndarray, threshold: float) -> np.ndarray:
    """(M, T) uint16: bit b set where bone b is more than `threshold` beyond the group's typical
    error at that frame, on the member's own limb (a mirrored member's bones are compared flipped)."""
    excess = bone_deg - nanmedian_first_axis(bone_deg)[None]
    excess = np.where(mirrored[..., None], excess[..., _FLIP_BONE], excess)
    over = np.nan_to_num(excess, nan=-np.inf) > threshold
    return (over * (1 << np.arange(len(BONE_NAMES)))).sum(axis=-1).astype("<u2")


def pose_moments(bone_deg: np.ndarray, mirrored: np.ndarray, fps: float, threshold: float, count: int = 3) -> list[dict]:
    """Per member: the biggest pose differences *held* for half a second (one per second), and how
    many seconds were left out because a bone pointed `VERY_DIFFERENT_DEG` or more away.

    Ranking single frames by their worst error lists artefacts: on the BTS clip every member's top
    three were hips or shoulders turned around (the pose model swapping left and right, or a dancer
    facing away), and cutting those off at any angle only let the next tier of the same artefacts
    in. A difference held for half a second, by the median over the window, is what a dancer can
    look at. Whether it is technique, style or a different part of the choreography, the list
    doesn't say; that's the diagnosis still to come."""
    from numpy.lib.stride_tricks import sliding_window_view

    width = max(3, int(round(0.5 * fps)))
    excess = bone_deg - nanmedian_first_axis(bone_deg)[None]
    very = np.nan_to_num(bone_deg) >= VERY_DIFFERENT_DEG
    out = []
    for i in range(len(bone_deg)):
        moments, taken = [], []
        if len(bone_deg[i]) >= width:
            window = sliding_window_view(np.where(very[i], np.nan, excess[i]), width, axis=0)  # (K, B, width)
            seen = np.isfinite(window).mean(axis=-1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                held = np.where(seen >= 0.8, np.nanmedian(window, axis=-1), np.nan)
                errors = np.nanmedian(sliding_window_view(bone_deg[i], width, axis=0), axis=-1)
            best = np.nanmax(np.nan_to_num(held, nan=-np.inf), axis=1)
            for k in np.argsort(-best):
                if best[k] < threshold or len(moments) == count:
                    break
                if any(abs(k - t) < fps for t in taken):
                    continue
                taken.append(k)
                b = int(np.argmax(np.nan_to_num(held[k], nan=-np.inf)))
                f = k + width // 2
                name = mirror_name(BONE_NAMES[b]) if mirrored[i, f] else BONE_NAMES[b]
                moments.append({"time_s": round(float(f / fps), 2), "bone": name, "error_deg": round(float(errors[k, b]), 1),
                                "above_group_deg": round(float(best[k]), 1)})
        left_out = np.flatnonzero((very[i] & (np.nan_to_num(excess[i]) > threshold)).any(axis=1))
        out.append({"worst_shape_moments": moments, "likely_label_swaps": int(len(np.unique(np.floor(left_out / fps))))})
    return out


def _heights(xy: np.ndarray, conf: np.ndarray) -> list:
    """Each member's usual nose-to-feet height in pixels (median over frames where both are drawn),
    so the lineup can keep their scale fixed instead of stretching every frame to one height."""
    out = []
    for i in range(len(xy)):
        known = (conf[i, :, 0] > 0) & (conf[i, :, 15] > 0) & (conf[i, :, 16] > 0)
        span = (xy[i, known, 15, 1] + xy[i, known, 16, 1]) / 2 - xy[i, known, 0, 1]
        span = span[span > 1]
        out.append(round(float(np.median(span)), 1) if span.size else 0.0)
    return out


def _beat(video: Path | None, xy: np.ndarray, conf: np.ndarray, fps: float, config: Config) -> tuple[dict | None, str, list | None, dict | None]:
    """The music's beat grid from the video's soundtrack and every member's clean landings against
    it: (beats, note, landings, landing_lock). Only a steady grid is used; the note says why there is
    none. `landing_lock` says whether this video's landings line up with the beat at all."""
    import importlib.util

    from ..audio.beats import beat_grid, load_audio

    if video is None:
        return None, "No beat: no video was given, so there is no soundtrack.", None, None
    if importlib.util.find_spec("av") is None:
        return None, "No beat: reading the soundtrack needs PyAV (pip install av).", None, None
    grid = beat_grid(load_audio(Path(video)))
    if grid is None:
        return None, "No beat: the video has no usable soundtrack.", None, None
    beats = {"tempo_bpm": round(grid.tempo_bpm, 2), "half_beat_s": grid.half_beat_s, "beat_phase_s": grid.beat_phase_s, "steady": grid.steady}
    if not grid.steady:
        return beats, "No beat grid: the tempo wasn't steady enough to draw one.", None, None
    landings = []
    for i in range(len(xy)):
        f = bone_directions(xy[i], conf[i], config.pose.min_keypoint_conf)
        v, vw = velocity(f.u, f.weight, fps, config.sync.max_bone_speed_deg_s)
        times = clean_landings(member_speed(v, vw), fps, config.beat) / fps
        landings.append([{"time_s": round(float(t), 3), "offset_ms": round(float(o), 1)} for t, o in zip(times, grid.offset_ms(times))])
    note = (f"Beat: {grid.tempo_bpm:.0f} BPM from the soundtrack. A delay between sound and picture shifts every dancer "
            "equally, and a move that lands between beats on purpose reads as off the beat.")
    lock = landing_lock(np.array([x["offset_ms"] for member in landings for x in member]), grid.half_beat_s * 1000.0)
    return beats, note, landings, lock


def viewer_data(result_dir: Path, config: Config, image_size: tuple[int, int], video: Path | None) -> dict:
    result_dir = Path(result_dir)
    kp = np.load(result_dir / "keypoints_smoothed.npz")
    sync = dict(np.load(result_dir / "sync_result.npz"))
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    xy = kp["xy"].astype(float)
    conf = measured_conf(kp["conf"].astype(float), config.pose.min_keypoint_conf)  # drawn: detected joints only
    members, n_frames = xy.shape[:2]
    mirrored = _per_frame_mirrored(sync, n_frames)
    ghost = ghost_keypoints(xy, conf, mirrored, config.pose.min_keypoint_conf)
    status = [[{"offset_ms": s["section_offset_ms"],
                "low_ms": (s["section_interval_ms"] or [None, None])[0], "high_ms": (s["section_interval_ms"] or [None, None])[1],
                "reliable": bool(s["section_reliable"])} for s in m["sections"]] for m in summary["members"]]
    return {
        "version": 1, "fps": float(kp["fps"]), "n_frames": int(n_frames), "image_size": [int(v) for v in image_size],
        "names": [str(n) for n in kp["names"]],
        "in_sync_ms": config.render.in_sync_ms, "off_sync_ms": config.render.off_sync_ms,
        "shape_threshold_deg": config.sync.shape_threshold_deg, "video": video_url(video) if video else None,
        "sections": [{"name": s["name"], "start_s": s["start_s"], "end_s": s["end_s"]} for s in summary["sections"]],
        "section_status": status,
        "keypoints": encode_int16(xy), "ghost": encode_int16(ghost),
        # Every drawn joint is at least 1/255, every undrawn one exactly 0: the page draws conf > 0.
        "conf": _b64(np.where(conf > 0, np.clip(np.round(conf * 255), 1, 255), 0).astype(np.uint8)),
        "halo": _b64(_halo(sync["bone_shape_deg"], mirrored, config.sync.shape_threshold_deg)),
        "heights": _heights(xy, conf),
        # Dancing together with the group (sync/unison.py): the ghost is drawn only there.
        "together": _b64(sync["together"].astype(np.uint8)) if "together" in sync else _b64(np.ones((members, n_frames), np.uint8)),
        "apart": apart_intervals(sync["together"], float(kp["fps"]), seen=(conf > 0).sum(axis=-1) >= 4) if "together" in sync else [[] for _ in range(members)],
        **dict(zip(("beats", "beat_note", "landings", "landing_lock"), _beat(video, xy, conf, float(kp["fps"]), config))),
        "moments": [{"worst_timing_moments": m["overall"]["worst_timing_moments"], **pose}
                    for m, pose in zip(summary["members"], pose_moments(sync["bone_shape_deg"], mirrored, float(kp["fps"]), config.sync.shape_threshold_deg))],
    }


def write_viewer(result_dir: Path, out_dir: Path, config: Config, image_size: tuple[int, int], video: Path | None) -> Path:
    """Write the viewer folder: the data script plus the page files. Returns index.html's path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = viewer_data(result_dir, config, image_size, video)
    (out_dir / "viewer_data.js").write_text("window.SYNCSCORE = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
    if VIEWER_DIR.is_dir():
        for path in VIEWER_DIR.iterdir():
            if path.is_file():
                shutil.copy2(path, out_dir / path.name)
    return out_dir / "index.html"
