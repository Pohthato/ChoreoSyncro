"""The gate: do fingerprints tell the people in a clip apart well enough to follow members by name?

Hand-labelled audit rows (detection row -> true name, `dancer` for anyone who isn't a member)
say who everyone really is. The labelled people of one frame are the picks: members seed their
names, everyone else there seeds "not a member". Then every other labelled row is named the way
`analyze --members` would name it (tracklets, fingerprints, learning rounds), and counted.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from ..config import Config
from .recognize import learn
from .tracks import Detections, build_tracklets

DANCER = "dancer"


def _share(count: int, total: int) -> float:
    return count / total if total else math.nan


def gate(det: Detections, vectors: np.ndarray, labels: dict[int, str], picked_frame: int, config: Config) -> dict:
    tracklets = build_tracklets(det, max(3, int(0.3 * det.fps)))
    pieces = [vectors[t.rows] for t in tracklets]
    frames = [t.frames for t in tracklets]
    piece_of = {int(row): p for p, t in enumerate(tracklets) for row in t.rows}

    def piece(row: int) -> int:
        """The row's tracklet, or a piece of its own when the tracker left it out."""
        if row not in piece_of:
            piece_of[row] = len(pieces)
            pieces.append(vectors[row:row + 1])
            frames.append(det.frame[row:row + 1])
        return piece_of[row]

    names = list(dict.fromkeys(n for n in labels.values() if n != DANCER))
    picked = {row: name for row, name in labels.items() if det.frame[row] == picked_frame}
    seeds = []
    for name in names:
        rows = [row for row, n in picked.items() if n == name]
        if len(rows) != 1:
            raise ValueError(f"{name} must be labelled exactly once in frame {picked_frame} (found {len(rows)})")
        seeds.append(piece(rows[0]))
    not_members = [piece(row) for row, n in picked.items() if n == DANCER]
    _, namings = learn(pieces, frames, seeds, not_members, config.identity)

    counts = {"member": 0, "dancer": 0}
    correct, unassigned, dancer_named = 0, 0, 0
    per_member = {name: [0, 0] for name in names}
    for row, name in labels.items():
        if row in picked:
            continue
        member = namings[piece_of[row]].member if row in piece_of else None  # no tracklet: unassigned
        if name == DANCER:
            counts["dancer"] += 1
            dancer_named += member is not None
            continue
        counts["member"] += 1
        right = member is not None and names[member] == name
        correct += right
        unassigned += member is None
        per_member[name][0] += right
        per_member[name][1] += 1
    return {"rows": counts, "member_correct": _share(correct, counts["member"]),
            "member_unassigned": _share(unassigned, counts["member"]),
            "dancer_as_member": _share(dancer_named, counts["dancer"]),
            "per_member": {name: _share(*c) for name, c in per_member.items()}}


def margin_grid(det: Detections, vectors: np.ndarray, labels: dict[int, str], picked_frame: int, config: Config,
                margins=(0.02, 0.04, 0.06, 0.08, 0.10)) -> dict[str, dict]:
    """gate() for each name margin, learning at twice the margin."""
    out = {}
    for margin in margins:
        identity = replace(config.identity, name_margin=margin, learn_margin=2 * margin)
        out[f"{margin:.2f}"] = gate(det, vectors, labels, picked_frame, replace(config, identity=identity))
    return out
