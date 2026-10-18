"""Detections with tracker IDs -> one consistent track per group member, plus an ID-swap report.

ByteTrack links detections frame to frame with a motion model, which is good for short gaps
and fails in two ways that matter here:
- **Fragmentation:** after an occlusion a member comes back with a *new* ID. Harmless, as long
  as we rejoin the pieces.
- **Swaps:** when two members cross, their IDs can be exchanged. Harmful: from then on one
  member's timing is attributed to the other.

The post-pass:
1. Group detections into *tracklets* (one tracker ID each) and drop very short ones.
2. Decide the group size N (config, or the most common number of clearly visible people).
3. Seed N member slots with the N tracklets that are visible together for the longest, then
   attach the remaining tracklets, longest first, to the slot they fit best. Fit means no time
   overlap, similar clothing colour (hue/saturation histogram of the torso), and if the gap is
   short, a nearby position.
   - A long tracklet (1 s or more) is always attached if any slot is free for its whole span,
     however poor the colour match: lighting changes as dancers move around the room, and the
     first version threw away 15% of a real clip's member detections (stretches up to 25 s)
     this way. Only short tracklets can be rejected for a poor fit (reflections, false
     detections, crew walking through).
   - A tracklet that overlaps every slot is dropped. (Moving shorter tracklets out of the way
     was tried: on the real clip it was attempted 14 times and never possible, so it was removed.)
4. Look for suspected swaps at every crossing (boxes overlapping): if after the crossing each
   track looks more like the *other* member's clothes than its own, flag it.

Matching outfits (common in K-pop) make the colour cue weak. Then the heuristic misses swaps,
which is why the true swap rate comes from a small hand-labelled audit (`syncscore audit`).
"""
from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Detections:
    """Every person detection in a video, one row each (flat arrays)."""
    frame: np.ndarray      # (N,) frame index
    track_id: np.ndarray   # (N,) tracker ID, -1 if the tracker gave none
    box: np.ndarray        # (N, 4) x1 y1 x2 y2 pixels
    box_conf: np.ndarray   # (N,)
    xy: np.ndarray         # (N, 17, 2)
    kconf: np.ndarray      # (N, 17)
    hist: np.ndarray       # (N, H) torso colour histogram, sums to 1 (zeros if unknown)
    fps: float
    n_frames: int
    image_size: tuple

    def save(self, path) -> None:
        np.savez_compressed(path, frame=self.frame, track_id=self.track_id, box=self.box, box_conf=self.box_conf, xy=self.xy,
                            kconf=self.kconf, hist=self.hist, fps=self.fps, n_frames=self.n_frames, image_size=np.array(self.image_size))

    @classmethod
    def load(cls, path) -> "Detections":
        d = np.load(path)
        return cls(d["frame"], d["track_id"], d["box"], d["box_conf"], d["xy"], d["kconf"], d["hist"],
                   float(d["fps"]), int(d["n_frames"]), tuple(int(v) for v in d["image_size"]))


@dataclass
class Tracklet:
    track_id: int
    rows: np.ndarray   # indices into Detections
    frames: np.ndarray
    hist: np.ndarray   # mean colour histogram

    @property
    def start(self) -> int:
        return int(self.frames[0])

    @property
    def end(self) -> int:
        return int(self.frames[-1])


@dataclass
class Slot:
    tracklets: list = field(default_factory=list)
    occupied: np.ndarray | None = None
    hist_sum: np.ndarray | None = None

    @property
    def hist(self) -> np.ndarray:
        total = self.hist_sum.sum()
        return self.hist_sum / total if total > 0 else self.hist_sum


def colour_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Hellinger distance between two histograms: 0 identical, 1 nothing in common."""
    if a.sum() <= 0 or b.sum() <= 0:
        return 0.5  # unknown: neither similar nor different
    return float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(a * b)))))


def build_tracklets(det: Detections, min_frames: int) -> list[Tracklet]:
    out = []
    for tid in np.unique(det.track_id):
        if tid < 0:
            continue
        rows = np.flatnonzero(det.track_id == tid)
        rows = rows[np.argsort(det.frame[rows])]
        if len(rows) < min_frames:
            continue
        out.append(Tracklet(int(tid), rows, det.frame[rows], det.hist[rows].mean(axis=0)))
    return out


def count_members(det: Detections, min_keypoint_conf: float) -> int:
    """Most common number of clearly visible people per frame (8+ confident keypoints)."""
    clear = (det.box_conf >= 0.5) & ((det.kconf >= min_keypoint_conf).sum(axis=1) >= 8)
    per_frame = np.bincount(det.frame[clear], minlength=det.n_frames)
    counts = Counter(per_frame[per_frame > 0].tolist())
    return max(counts, key=lambda k: (counts[k], k)) if counts else 0


def _center(det: Detections, row: int) -> np.ndarray:
    b = det.box[row]
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def _height(det: Detections, row: int) -> float:
    return float(max(det.box[row, 3] - det.box[row, 1], 1.0))


def _fit_cost(det: Detections, slot: Slot, t: Tracklet, max_gap: int, overlap_ok: int) -> float:
    """How badly tracklet t fits a member slot (inf = impossible)."""
    if (slot.occupied[t.frames]).sum() > overlap_ok:
        return np.inf
    cost = colour_distance(slot.hist, t.hist)
    # Position: compare with the slot's detection closest in time, if close enough in time.
    best_gap, best = None, None
    for other in slot.tracklets:
        for gap, a, b in ((t.start - other.end, other.rows[-1], t.rows[0]), (other.start - t.end, t.rows[-1], other.rows[0])):
            if 0 < gap <= max_gap and (best_gap is None or gap < best_gap):
                best_gap, best = gap, (a, b)
    if best is not None:
        a, b = best
        jump = np.linalg.norm(_center(det, a) - _center(det, b)) / _height(det, a)
        cost += 0.5 * min(jump, 2.0)
    else:
        cost += 0.5  # no nearby evidence either way
    return cost


def assign_members(det: Detections, tracklets: list[Tracklet], members: int, max_gap_s: float, max_cost: float = 0.9) -> list[Slot]:
    """Group tracklets into `members` slots (see module docstring)."""
    if not tracklets or members <= 0:
        return []
    n = det.n_frames
    present = np.zeros((len(tracklets), n), dtype=bool)
    for k, t in enumerate(tracklets):
        present[k, t.frames] = True
    # Seed: the frame where the most tracklets are visible, preferring long tracklets.
    lengths = present.sum(axis=1)
    score = (present * lengths[:, None]).sum(axis=0) + 1e6 * np.minimum(present.sum(axis=0), members)
    seed_frame = int(np.argmax(score))
    seeds = [k for k in np.argsort(-lengths) if present[k, seed_frame]][:members]
    slots = []
    for k in seeds:
        slot = Slot(occupied=np.zeros(n, dtype=bool), hist_sum=np.zeros_like(tracklets[k].hist))
        _attach(slot, tracklets[k])
        slots.append(slot)
    used = set(seeds)
    max_gap = int(round(max_gap_s * det.fps))
    long_frames = int(round(det.fps))
    for k in np.argsort(-lengths):
        if k in used:
            continue
        t = tracklets[k]
        costs = [_fit_cost(det, slot, t, max_gap, overlap_ok=2) for slot in slots]
        best = int(np.argmin(costs)) if costs else -1
        fits = best >= 0 and np.isfinite(costs[best])
        if len(slots) < members and (not fits or costs[best] > max_cost):
            slot = Slot(occupied=np.zeros(n, dtype=bool), hist_sum=np.zeros_like(t.hist))
            _attach(slot, t)
            slots.append(slot)
        elif fits and (costs[best] <= max_cost or len(t.frames) >= long_frames):
            _attach(slots[best], t)
        # else: busy everywhere, or short and a poor fit: not attached (dropped)
    return slots


def _attach(slot: Slot, t: Tracklet) -> None:
    slot.tracklets.append(t)
    slot.occupied[t.frames] = True
    slot.hist_sum = slot.hist_sum + t.hist * len(t.frames)


@dataclass
class MemberTracks:
    xy: np.ndarray    # (M, T, 17, 2)
    conf: np.ndarray  # (M, T, 17)
    box: np.ndarray   # (M, T, 4)
    hist: np.ndarray  # (M, T, H) torso colour, zeros where unseen
    fragments: list   # tracker IDs joined into each member
    rows: np.ndarray  # (M, T) the Detections row each member uses per frame, -1 where unseen


def assemble(det: Detections, slots: list[Slot]) -> MemberTracks:
    """Slots -> per-member arrays, members ordered left to right."""
    m, n = len(slots), det.n_frames
    xy = np.full((m, n, 17, 2), np.nan)
    conf = np.zeros((m, n, 17))
    box = np.full((m, n, 4), np.nan)
    hist = np.zeros((m, n, det.hist.shape[1]))
    best_conf = np.full((m, n), -1.0)
    rows = np.full((m, n), -1, dtype=int)
    for i, slot in enumerate(slots):
        for t in slot.tracklets:
            for row, f in zip(t.rows, t.frames):
                if det.box_conf[row] > best_conf[i, f]:  # overlapping frames keep the surer detection
                    best_conf[i, f] = det.box_conf[row]
                    xy[i, f], conf[i, f], box[i, f], hist[i, f] = det.xy[row], det.kconf[row], det.box[row], det.hist[row]
                    rows[i, f] = row
    # Name members left to right by where they stand, so names are stable and readable.
    with np.errstate(all="ignore"):
        order = np.argsort([np.nanmedian((box[i, :, 0] + box[i, :, 2]) / 2) for i in range(m)])
    fragments = [sorted(t.track_id for t in slots[i].tracklets) for i in order]
    return MemberTracks(xy[order], conf[order], box[order], hist[order], fragments, rows[order])


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1, y1 = np.maximum(a[..., 0], b[..., 0]), np.maximum(a[..., 1], b[..., 1])
    x2, y2 = np.minimum(a[..., 2], b[..., 2]), np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[..., 2] - r[..., 0]) * (r[..., 3] - r[..., 1])
    with np.errstate(invalid="ignore"):
        return inter / (area(a) + area(b) - inter)


def crossings(box: np.ndarray, fps: float, iou_threshold: float) -> list[tuple[int, int, int, int]]:
    """(member_a, member_b, first_frame, last_frame) for every stretch where two boxes overlap."""
    out = []
    m = len(box)
    for a in range(m):
        for b in range(a + 1, m):
            close = np.nan_to_num(_iou(box[a], box[b])) > iou_threshold
            f = 0
            while f < len(close):
                if not close[f]:
                    f += 1
                    continue
                end = f
                while end < len(close) and close[end]:
                    end += 1
                out.append((a, b, f, end - 1))
                f = end
    return out


def suspected_swaps(hist: np.ndarray, box: np.ndarray, fps: float, iou_threshold: float, margin: float = 0.1) -> list[dict]:
    """Crossings after which each member looks more like the other's clothes than their own.

    hist (M, T, H) per-member colour histogram per frame (zeros where unseen).
    """
    found = []
    span = int(round(fps))
    for a, b, start, end in crossings(box, fps, iou_threshold):
        def mean(i, lo, hi):
            h = hist[i, max(lo, 0):max(hi, 0)]
            h = h[h.sum(axis=1) > 0]
            return h.mean(axis=0) if len(h) else np.zeros(hist.shape[-1])
        a0, b0 = mean(a, start - span, start), mean(b, start - span, start)
        a1, b1 = mean(a, end + 1, end + 1 + span), mean(b, end + 1, end + 1 + span)
        if min(h.sum() for h in (a0, b0, a1, b1)) <= 0:
            continue
        same = colour_distance(a0, a1) + colour_distance(b0, b1)
        crossed = colour_distance(a0, b1) + colour_distance(b0, a1)
        if crossed + margin < same:
            found.append({"members": [a, b], "time_s": round(start / fps, 2), "evidence": round(same - crossed, 3)})
    return found


def score_audit(csv_path: str | Path, fps: float) -> dict:
    """Score a hand-labelled audit: rows of frame, shown_label, true_name.

    Each shown label is mapped to the true name it most often has. Accuracy is the share of
    audited sightings where the label had that name. A switch is two consecutive audited
    sightings of the same label with different true names. Switches per minute are estimated
    over the audited time span.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("true_name", "").strip():
                rows.append((int(row["frame"]), row["shown_label"].strip(), row["true_name"].strip()))
    if not rows:
        return {"labelled": 0}
    by_label: dict[str, list] = {}
    for frame, label, name in sorted(rows):
        by_label.setdefault(label, []).append(name)
    correct = switches = 0
    for label, names in by_label.items():
        majority = Counter(names).most_common(1)[0][0]
        correct += sum(n == majority for n in names)
        switches += sum(1 for x, y in zip(names, names[1:]) if x != y)
    frames = [r[0] for r in rows]
    minutes = max((max(frames) - min(frames)) / fps / 60.0, 1e-9)
    return {"labelled": len(rows), "accuracy": correct / len(rows), "switches": switches,
            "audited_minutes": round(minutes, 2), "switches_per_minute": round(switches / minutes, 3)}
