"""Follow each picked member by name, from the frame she was picked in to both ends of the clip.

1. Tracks are split where the tracker handed an ID to someone else (motion.py).
2. Every piece is named by fingerprint (recognize.py): the picked people are the first examples,
   and the other people in the picked frame are examples of "not a member".
3. For each member, walk outward from her picked piece, forward in time and then backward. A
   piece named her is taken unless:
   - it overlaps a piece she already has: the one named more surely keeps her name, and the
     other is named again without her, once;
   - she was unseen only briefly (and the camera did not cut) and the piece is far from where
     her motion says she is heading: a lookalike on the other side of the stage.

The report lists the splits, the pieces whose naming was close, and the pieces the position
check refused, so a person can check the hard moments by eye.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..config import Config
from .camera import CameraPath
from .motion import Track, positions, split_at_crossings
from .recognize import MAX_SHARED, Naming, Recognizer, learn, overlap
from .tracks import Detections, Slot, Tracklet

HEADING_S = 2.0  # seconds of a piece the motion model follows before predicting across a gap


@dataclass
class Following:
    slots: list[Slot]       # one per member, in pick order
    recognizer: Recognizer
    report: dict


def _one_row(det: Detections, row: int) -> Tracklet:
    return Tracklet(int(det.track_id[row]), np.array([row]), det.frame[[row]], det.hist[row].astype(float))


def follow_members(det: Detections, tracklets: list[Tracklet], camera: CameraPath, picked_rows: list[int],
                   config: Config) -> Following:
    """Slots for the members picked as `picked_rows` (one detection each, all in one frame)."""
    if det.vectors is None:
        raise ValueError("Following members needs fingerprints (det.vectors).")
    pieces, splits = split_at_crossings(det, tracklets, camera, config)
    owner = {int(r): i for i, p in enumerate(pieces) for r in p.rows}

    def piece_of(row: int, taken: set[int]) -> int:
        p = owner.get(int(row))
        if p is None or p in taken:
            pieces.append(_one_row(det, int(row)))
            p = len(pieces) - 1
        taken.add(p)
        return p

    picked_frame = int(det.frame[picked_rows[0]])
    taken: set[int] = set()
    seeds = [piece_of(r, taken) for r in picked_rows]
    others = [r for r in np.flatnonzero(det.frame == picked_frame) if r not in set(map(int, picked_rows))]
    not_members = [piece_of(r, taken) for r in others]

    ident = config.identity
    recognizer, namings = learn([det.vectors[p.rows] for p in pieces], [p.frames for p in pieces], seeds,
                                not_members, ident)
    stage, heights = positions(det, np.arange(len(det.frame)), camera, config.pose.min_keypoint_conf)
    free_after = ident.free_after_s * det.fps
    fps = det.fps

    accepted: list[dict[int, float]] = [{s: math.inf} for s in seeds]  # member -> {piece: margin}
    rejections: list[dict] = []

    def gate(m: int, p: int) -> float | None:
        """How far piece p is from where member m was heading (Mahalanobis), or None when not checked."""
        P = pieces[p]
        forward = P.start > pieces[seeds[m]].start
        if forward:
            before = [q for q in accepted[m] if pieces[q].end < P.start]
            if not before:
                return None
            anchor = pieces[max(before, key=lambda q: pieces[q].end)]
            gap, near, far = P.start - anchor.end - 1, anchor.end, P.start
            rows = anchor.rows[anchor.frames > anchor.end - HEADING_S * fps]
            point, frames = stage[P.rows[0]], det.frame[rows]
        else:
            after = [q for q in accepted[m] if pieces[q].start > P.end]
            if not after:
                return None
            anchor = pieces[min(after, key=lambda q: pieces[q].start)]
            gap, near, far = anchor.start - P.end - 1, anchor.start, P.end
            rows = anchor.rows[anchor.frames < anchor.start + HEADING_S * fps][::-1]
            point, frames = stage[P.rows[-1]], -det.frame[rows]  # time runs backward
        if gap > free_after or camera.segment[near] != camera.segment[far]:
            return None
        track = Track.follow(stage[rows], frames, float(np.median(heights[rows])), fps)
        return track.predict(gap + 1).distance(point)

    def refused(m: int, p: int) -> bool:
        d = gate(m, p)
        if d is not None and d > ident.position_gate:
            P = pieces[p]
            frame = P.start if P.start > pieces[seeds[m]].start else P.end
            rejections.append({"time_s": round(frame / fps, 2), "member": m, "distance": round(d, 1)})
            return True
        return False

    def clashes(m: int, p: int) -> list[int]:
        return [q for q in accepted[m] if overlap(pieces[q].frames, pieces[p].frames) > MAX_SHARED]

    dropped: dict[int, set[int]] = {}  # piece -> members it lost

    def offer(m: int, p: int, margin: float, rescore: bool) -> None:
        clash = clashes(m, p)
        if clash:
            if margin > max(accepted[m][q] for q in clash):
                for q in clash:
                    del accepted[m][q]
                    if not rescore:
                        dropped.setdefault(q, set()).add(m)
                accepted[m][p] = margin
            elif not rescore:
                dropped.setdefault(p, set()).add(m)
            return
        if not refused(m, p):
            accepted[m][p] = margin

    fixed = set(seeds) | set(not_members)
    for m, seed in enumerate(seeds):
        mine = [p for p, n in enumerate(namings) if n.member == m and p not in fixed]
        start = pieces[seed].start
        for p in sorted((p for p in mine if pieces[p].start > start), key=lambda p: pieces[p].start):
            offer(m, p, namings[p].margin, rescore=False)
        for p in sorted((p for p in mine if pieces[p].start <= start), key=lambda p: -pieces[p].end):
            offer(m, p, namings[p].margin, rescore=False)

    held = {p for a in accepted for p in a}
    for p, lost in sorted(dropped.items(), key=lambda item: -namings[item[0]].margin):
        if p in held:
            continue
        again = recognizer.name(det.vectors[pieces[p].rows], exclude=lost)
        if again.member is not None:
            offer(again.member, p, again.margin, rescore=True)
            held = {q for a in accepted for q in a}

    slots = [Slot(tracklets=sorted((pieces[p] for p in a), key=lambda t: t.start)) for a in accepted]
    report = {
        "picked_frame": picked_frame,
        "splits": [{"time_s": round(s.frame / fps, 2), "track_id": s.track_id, "reason": s.reason} for s in splits],
        "close_calls": _close_calls(pieces, namings, fixed, ident.name_margin, fps),
        "position_rejections": sorted(rejections, key=lambda r: r["time_s"]),
    }
    return Following(slots, recognizer, report)


def _close_calls(pieces: list[Tracklet], namings: list[Naming], fixed: set[int], name_margin: float,
                 fps: float) -> list[dict]:
    out = []
    for p, n in enumerate(namings):
        if p in fixed or not math.isfinite(n.margin) or abs(n.margin - name_margin) > name_margin / 2:
            continue
        member_scores = n.scores[:-1]
        best = int(np.nanargmax(member_scores)) if np.any(~np.isnan(member_scores)) else None
        out.append({"start_s": round(pieces[p].start / fps, 2), "end_s": round(pieces[p].end / fps, 2),
                    "member": best, "margin": round(float(n.margin), 3), "named": n.member is not None})
    return sorted(out, key=lambda c: c["start_s"])


def allow_for(recognizer: Recognizer) -> Callable[[Detections, int, int], bool]:
    """A check for gap filling: may `row` be used for `member`? Only if its fingerprint names her."""
    def allow(det: Detections, member: int, row: int) -> bool:
        return recognizer.name(det.vectors[row:row + 1]).member == member
    return allow
