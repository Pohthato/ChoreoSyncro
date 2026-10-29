"""The members picked by name in one frame: read, check, and tie each to a detection.

The upload page shows one frame with a box per person; the user names the members there. That
arrives as `picks.json`: {"frame": n, "members": [{"name": ..., "box": [x1, y1, x2, y2]}, ...]},
members in the order they were named. Names are kept exactly as typed (trimmed); the page and the
viewer escape them when they show them.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tracks import Detections, _iou

MAX_NAME = 40
MIN_IOU = 0.5          # a pick's box must overlap its detection at least this much
MAX_OVERLAP = 0.05     # suggested frames: people overlapping more than this rank last
CLEAR_KEYPOINTS = 8
EDGE_SHARE = 0.05      # suggested frames: leave out this share of the clip at each end
MIN_APART_S = 2.0


class PickError(ValueError):
    """A problem with the picks, in words the page can show as they are."""


@dataclass
class Pick:
    name: str
    box: np.ndarray


@dataclass
class Picks:
    frame: int
    members: list[Pick]


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def parse_picks(data: dict, n_frames: int) -> Picks:
    members = data.get("members") if isinstance(data, dict) else None
    if not isinstance(members, list) or len(members) < 2:
        raise PickError("Name at least 2 members.")
    frame = data.get("frame")
    if not isinstance(frame, int) or isinstance(frame, bool) or not 0 <= frame < n_frames:
        raise PickError(f"Frame {frame} is not in this video.")
    out, seen = [], set()
    for member in members:
        member = member if isinstance(member, dict) else {}
        box = member.get("box")
        if not isinstance(box, list) or len(box) != 4 or not all(_number(v) for v in box):
            raise PickError("Each pick needs a box of 4 numbers.")
        name = member.get("name")
        name = name.strip() if isinstance(name, str) else ""
        if not name:
            raise PickError("Type a name for every member.")
        if len(name) > MAX_NAME:
            raise PickError(f"Names can be up to {MAX_NAME} characters.")
        if name.casefold() in seen:
            raise PickError(f"Another member is already called {name}.")
        seen.add(name.casefold())
        out.append(Pick(name, np.array(box, dtype=float)))
    return Picks(frame, out)


def load_picks(path, n_frames: int) -> Picks:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PickError(f"Could not read the picks: {error}") from error
    return parse_picks(data, n_frames)


def dump_picks(picks: Picks) -> dict:
    return {"frame": picks.frame, "members": [{"name": p.name, "box": [float(v) for v in p.box]} for p in picks.members]}


def resolve_picks(picks: Picks, det: Detections) -> list[int]:
    """The detection row each pick stands on, in pick order."""
    candidates = np.flatnonzero(det.frame == picks.frame)
    rows, owner = [], {}
    for pick in picks.members:
        overlap = _iou(pick.box[None], det.box[candidates]) if len(candidates) else np.zeros(0)
        if not len(overlap) or np.nanmax(overlap) < MIN_IOU:
            raise PickError(f"No person found where {pick.name} was picked in frame {picks.frame}.")
        row = int(candidates[np.nanargmax(overlap)])
        if row in owner:
            raise PickError(f"{owner[row]} and {pick.name} were picked on the same person.")
        owner[row] = pick.name
        rows.append(row)
    return rows


def _clear(det: Detections, rows: np.ndarray, min_conf: float) -> np.ndarray:
    return (det.kconf[rows] >= min_conf).sum(axis=1) >= CLEAR_KEYPOINTS


def people_in_frame(det: Detections, frame: int, min_conf: float) -> list[dict]:
    rows = np.flatnonzero(det.frame == frame)
    rows = rows[np.argsort(det.box[rows, 0] + det.box[rows, 2], kind="stable")]
    clear = _clear(det, rows, min_conf)
    return [{"row": int(r), "box": [round(float(v), 1) for v in det.box[r]], "clear": bool(c)} for r, c in zip(rows, clear)]


def suggest_frames(det: Detections, min_conf: float, count: int = 5) -> list[int]:
    """Frames good to pick in: the most clearly visible people, overlapping least, away from the
    ends of the clip, at least MIN_APART_S apart."""
    edge = int(EDGE_SHARE * det.n_frames)
    order = np.argsort(det.frame, kind="stable")
    starts = np.searchsorted(det.frame[order], np.arange(det.n_frames + 1))
    ranked = []
    for f in range(edge, det.n_frames - edge):
        rows = order[starts[f]:starts[f + 1]]
        if not len(rows):
            continue
        boxes = det.box[rows]
        pairs = _iou(boxes[:, None], boxes[None]) if len(rows) > 1 else np.zeros((1, 1))
        worst = float(np.nanmax(pairs[~np.eye(len(pairs), dtype=bool)])) if len(rows) > 1 else 0.0
        ranked.append((worst <= MAX_OVERLAP, int(_clear(det, rows, min_conf).sum()), -worst, -f))
    ranked.sort(reverse=True)
    apart = MIN_APART_S * det.fps
    chosen: list[int] = []
    for *_, minus_f in ranked:
        f = -minus_f
        if all(abs(f - c) >= apart for c in chosen):
            chosen.append(f)
            if len(chosen) == count:
                break
    return chosen
