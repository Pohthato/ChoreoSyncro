"""Follow each picked member by name, from the frame she was picked in to both ends of the clip.

1. Tracks are split where the tracker handed an ID to someone else (motion.py).
2. Every piece is named by fingerprint (recognize.py): the picked people are the first examples,
   and the other people in the picked frame are examples of "not a member".
3. For each member, walk outward from her picked piece, forward in time and then backward. A
   piece named her is taken unless:
   - it shares frames with a piece she already has: the piece whose rows there look more like
     her keeps those frames and the other gives them up, since a track that changes hands
     unnoticed shares a moment with the next one and is still right the rest of the time. A
     piece left with nothing is named again without her, once;
   - she was unseen only briefly (and the camera did not cut) and the piece is far from where
     her motion says she is heading: a lookalike on the other side of the stage.
4. A piece that fell short of the margin is hers after all when her motion vouches for it, which
   the margin can't know about. All of these must hold, and a doubtful moment leaves her unseen:
   - she leads it by half the margin (what the report calls close) over not-a-member and over the
     members who could be this piece: one seen elsewhere during it can't be;
   - it is no longer than motion can be trusted for (`free_after_s`), so a check at one end
     speaks for all of it;
   - it picks up where her motion was heading from a piece she has on at least one side. In a
     pile-up the other side can be off, since the feet that give a position are hidden.
   Pieces taken this way can vouch for the next one, so this runs until nothing changes.

The report lists the splits, the pieces whose naming was close, the pieces the position check
refused and the pieces taken under the margin, so a person can check the hard moments by eye.
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

    def gate(m: int, p: int, forward: bool, through: bool = False) -> float | None:
        """How far piece p is from where member m was heading, seen from her nearest piece before it
        (or after it, not `forward`), in Mahalanobis distance; None when not checked. With `through`,
        that piece may run into p and the check starts where p goes past it: one frame on, passing
        means the two line up, but failing can be the tracker's boxes, so only a pass counts."""
        P = pieces[p]
        if forward:
            before = [q for q in accepted[m] if pieces[q].start < P.start and pieces[q].end < (P.end if through else P.start)]
            if not before:
                return None
            anchor = pieces[max(before, key=lambda q: pieces[q].end)]
            first = P.rows[P.frames > anchor.end][0]
            near, far = anchor.end, int(det.frame[first])
            gap = far - near - 1
            rows = anchor.rows[anchor.frames > anchor.end - HEADING_S * fps]
            point, frames = stage[first], det.frame[rows]
        else:
            after = [q for q in accepted[m] if pieces[q].start > (P.start if through else P.end) and pieces[q].end > P.end]
            if not after:
                return None
            anchor = pieces[min(after, key=lambda q: pieces[q].start)]
            last = P.rows[P.frames < anchor.start][-1]
            near, far = anchor.start, int(det.frame[last])
            gap = near - far - 1
            rows = anchor.rows[anchor.frames < anchor.start + HEADING_S * fps][::-1]
            point, frames = stage[last], -det.frame[rows]  # time runs backward
        if gap > free_after or camera.segment[near] != camera.segment[far]:
            return None
        track = Track.follow(stage[rows], frames, float(np.median(heights[rows])), fps)
        return track.predict(gap + 1).distance(point)

    def refused(m: int, p: int) -> bool:
        P = pieces[p]
        forward = P.start > pieces[seeds[m]].start
        d = gate(m, p, forward)
        if d is not None and d > ident.position_gate:
            frame = P.start if forward else P.end
            rejections.append({"time_s": round(frame / fps, 2), "member": m, "distance": round(d, 1)})
            return True
        return False

    def stronger(m: int, p: int, q: int, shared: np.ndarray) -> bool:
        """Do p's rows in the `shared` frames look more like member m than q's? Neither votes for itself."""
        def likeness(piece: int) -> float:
            rows = pieces[piece].rows[np.isin(pieces[piece].frames, shared)]
            return recognizer.likeness(m, det.vectors[rows], skip=[p, q])
        return likeness(p) > likeness(q)

    def cut(p: int, shared: np.ndarray) -> bool:
        """Take the `shared` frames out of piece p; False, leaving it as it was, if that would empty it."""
        P = pieces[p]
        rows = P.rows[~np.isin(P.frames, shared)]
        if len(rows) == 0:
            return False
        pieces[p] = Tracklet(P.track_id, rows, det.frame[rows], det.hist[rows].mean(axis=0))
        return True

    dropped: dict[int, set[int]] = {}  # piece -> members it lost

    def offer(m: int, p: int, margin: float, rescore: bool, placed: bool = False) -> None:
        beat_one = False  # then the frames were judged by look, and the piece it beat can't vouch for a heading
        for q in list(accepted[m]):
            shared = np.intersect1d(pieces[p].frames, pieces[q].frames)
            if len(shared) <= MAX_SHARED:
                continue
            won = stronger(m, p, q, shared)
            beat_one |= won
            loser = q if won else p
            if cut(loser, shared):
                continue
            if not rescore:
                dropped.setdefault(loser, set()).add(m)
            if loser == p:
                return
            del accepted[m][q]
        if placed or beat_one or not refused(m, p):
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

    def lead(m: int, p: int) -> float:
        """Her lead on piece p over not-a-member and the members who could be it."""
        scores = namings[p].scores

        def seen_during(r: int) -> bool:
            return sum(overlap(pieces[p].frames, pieces[q].frames) for q in accepted[r]) > MAX_SHARED

        rivals = [scores[-1]] + [scores[r] for r in range(len(seeds)) if r != m and not seen_during(r)]
        rivals = [s for s in rivals if not math.isnan(s)]
        return float(scores[m] - max(rivals)) if rivals else math.inf

    def continues(m: int, p: int) -> bool:
        return any(d is not None and d <= ident.position_gate
                   for d in (gate(m, p, True, through=True), gate(m, p, False, through=True)))

    under_margin: list[tuple[int, int, float]] = []  # piece, member, lead
    waiting = sorted((p for p, n in enumerate(namings) if n.member is None and p not in fixed and p not in held
                      and np.any(~np.isnan(n.scores[:-1]))), key=lambda p: -namings[p].margin)
    moved = True
    while moved:
        moved = False
        for p in list(waiting):
            m = int(np.nanargmax(namings[p].scores[:-1]))
            by = lead(m, p)
            if by < ident.name_margin / 2 or pieces[p].end - pieces[p].start > free_after or not continues(m, p):
                continue
            waiting.remove(p)
            offer(m, p, namings[p].margin, rescore=True, placed=True)
            if p in accepted[m]:
                moved = True
                under_margin.append((p, m, by))

    slots = [Slot(tracklets=sorted((pieces[p] for p in a), key=lambda t: t.start)) for a in accepted]
    report = {
        "picked_frame": picked_frame,
        "splits": [{"time_s": round(s.frame / fps, 2), "track_id": s.track_id, "reason": s.reason} for s in splits],
        "close_calls": _close_calls(pieces, namings, fixed, ident.name_margin, fps),
        "position_rejections": sorted(rejections, key=lambda r: r["time_s"]),
        "taken_under_margin": sorted(({"start_s": round(pieces[p].start / fps, 2), "end_s": round(pieces[p].end / fps, 2),
                                       "member": m, "lead": round(by, 3)} for p, m, by in under_margin),
                                     key=lambda t: t["start_s"]),
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
