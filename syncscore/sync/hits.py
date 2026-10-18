"""Timing at the group's hits: the big movements everyone makes at the same moment.

Section timing (sections.py) pools many 2-second windows of cross-correlation. On real dance
most of the time is individually styled grooving, so the windows disagree and a section needs
about 30 s before its median can be trusted. The idea here: when the whole group snaps into a
move, everyone's limbs reach top speed together, and the moment of top speed can be read off
each member's own motion to a fraction of a frame.

**Status: measured, not reported.** `syncscore shift-test` scores it on real footage, and no
report output uses it. On the test clip the members' peaks at the same hit spread too widely
(IQR 45-100 ms) and a 32 s section held too few matched hits (0-8 per member) for an interval
under 50 ms: sections from hits read 0%, and together with windows 24% instead of 22%. See
docs/walkthrough.md, Stage B.

1. **Speed.** Per member, the RMS angular speed of their bones (the same glitch-gated bone
   velocities the windows use), divided by that member's typical speed so that big movers
   don't dominate the group, and lightly smoothed (Gaussian, 1 frame; symmetric, so no peak
   moves).
2. **Group hits.** Peaks of the members' median speed that stand clearly above their
   surroundings (`hits.prominence`).
3. **The member's peak.** The highest peak of the member's own speed within ±400 ms of the
   hit, located to a fraction of a frame with a parabola. Moving a member's motion in time moves
   this peak by exactly as much (on the real clip it moved correctly for 99% of planted shifts).
   Speed alone can't tell an arm going up from the same arm coming back down 250 ms later:
   both are speed peaks, and at this tempo both fall inside ±400 ms. On the real clip half of
   all matches had a second peak nearly as high (`max_second_peak`). Only then, the *direction*
   breaks the tie: the peak whose bone velocities point the same way as the other members' do
   at their own clear peaks (cosine at least `min_agreement`, and clearly better than the other
   candidate) wins.
   Direction is used only for ties because it costs precision: the reference direction comes
   from the other members, so it moves a little when they do. Requiring it for every match
   (tried on the real clip) matched fewer hits and was three times less precise.
   "No reading" when the member is mostly unseen, never speeds up, or the tie stays a tie.
4. **Offset** = the member's peak time minus the hit's moment according to the *other* members
   (leave-one-out, as everywhere else), each corrected by their own typical offset, so that
   "0 ms" is the group's median member as in section timing (`relative_offsets`).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .timing import parabolic_peak

SMOOTH_FRAMES = 1.0     # Gaussian sigma for speed and velocity
MIN_SEEN = 0.8          # share of the search range the member must be visible in
MIN_GROUP = 4           # members visible for the group speed to mean anything
MIN_SPACING_S = 0.3     # two group hits are at least this far apart
PEAK_PROMINENCE = 0.3   # a member's peak must stand this far (x typical speed) above its dips
MIN_PEAK_SPEED = 1.5    # ...and reach at least this (x typical speed): a real movement, not a twitch
TIE_GAP = 0.5           # in a tie, the winner's direction must agree this much better (cosine)
AGREE_MS = 100.0        # the other members' peaks must cluster this closely to define the hit's moment


@dataclass
class HitTiming:
    frames: np.ndarray     # (H,) group hit, fractional frame
    offset_ms: np.ndarray  # (M, H) + = behind the group's median member at that hit; NaN = no clean match
    agreement: np.ndarray  # (M, H) cosine of the member's movement with the others' at their peaks (-1..1)


def _smooth_known(x: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Gaussian smoothing over known samples only, so a gap doesn't drag its edges toward 0."""
    num = gaussian_filter1d(np.where(known, x, 0.0), SMOOTH_FRAMES, axis=0)
    den = gaussian_filter1d(known.astype(float), SMOOTH_FRAMES, axis=0)
    return np.where(known, num / np.maximum(den, 1e-9), np.nan)


def motion(v: np.ndarray, vw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bone velocities (T, B, 2) with weights (T, B) -> normalised speed (T,), smoothed velocity
    (T, B, 2) in the same units, and the weights (0 where the velocity is unknown)."""
    known = np.isfinite(v[..., 0]) & (vw > 0)
    w = np.where(known, vw, 0.0)
    total = w.sum(axis=1)
    sq = np.sum(np.nan_to_num(v) ** 2, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.where(total > 0.5, np.sqrt((w * sq).sum(axis=1) / total), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        typical = np.nanmedian(speed)
    if not np.isfinite(typical) or typical <= 0:
        return np.full(len(v), np.nan), np.full_like(v, np.nan), np.zeros_like(w)
    speed = _smooth_known(speed / typical, np.isfinite(speed))
    vel = np.stack([_smooth_known(v[..., c] / typical, known) for c in range(2)], axis=-1)
    return speed, vel, w


def group_hits(speed: np.ndarray, fps: float, prominence: float) -> np.ndarray:
    """(M, T) normalised speeds -> frames where the group's median speed peaks clearly."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        group = np.where(np.isfinite(speed).sum(axis=0) >= min(MIN_GROUP, len(speed)), np.nanmedian(speed, axis=0), np.nan)
    peaks, _ = find_peaks(np.nan_to_num(group), prominence=prominence, distance=max(1, int(round(MIN_SPACING_S * fps))))
    return peaks


def _direction(vel: list, weight: list, others: list, frames: list) -> tuple[np.ndarray, np.ndarray]:
    """The others' median bone velocity (B, 2), each read at their own frame, and per bone the share of them seen."""
    values = np.stack([vel[j][f] for j, f in zip(others, frames)])
    seen = np.stack([weight[j][f] > 0 for j, f in zip(others, frames)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = np.nanmedian(np.where(seen[..., None], values, np.nan), axis=0)
    trust = np.where(np.isfinite(ref[..., 0]), seen.mean(axis=0), 0.0)
    return np.nan_to_num(ref), trust


def _cosine(a: np.ndarray, aw: np.ndarray, ref: np.ndarray, trust: np.ndarray) -> float:
    w = np.where(np.isfinite(a[..., 0]), aw * trust, 0.0)[:, None]
    a = np.nan_to_num(a)
    den = np.sqrt(np.sum(w * a * a) * np.sum(w * ref * ref))
    return float(np.sum(w * a * ref) / den) if den > 1e-12 else np.nan


def _candidates(speed: np.ndarray, hit: int, reach: int, max_second_peak: float) -> np.ndarray:
    """Frames of the member's speed peaks within reach of `hit` that are nearly as high as the highest.
    Empty when the member is mostly unseen there or never speeds up."""
    a, b = hit - reach, hit + reach
    if a < 1 or b > len(speed) - 2:
        return np.zeros(0, dtype=int)
    seg = speed[a - 1:b + 2]  # seg[k] is frame a - 1 + k; one extra sample each side so only true maxima count
    if np.isfinite(seg[1:-1]).mean() < MIN_SEEN:
        return np.zeros(0, dtype=int)
    filled = np.nan_to_num(seg, nan=0.0)
    peaks, _ = find_peaks(filled, prominence=PEAK_PROMINENCE)  # never the two end samples
    peaks = peaks[(filled[peaks] >= MIN_PEAK_SPEED) & np.isfinite(seg[peaks - 1]) & np.isfinite(seg[peaks + 1])]
    if peaks.size == 0:
        return np.zeros(0, dtype=int)
    return a - 1 + peaks[filled[peaks] >= max_second_peak * filled[peaks].max()]


def _sub_frame(speed: np.ndarray, frame: int) -> float:
    return frame + parabolic_peak(speed[frame - 1], speed[frame], speed[frame + 1])


def hit_timing(velocities: list, weights: list, fps: float, cfg) -> HitTiming:
    """velocities: per member (T, B, 2) bone velocities on the video clock, weights (T, B).
    `cfg` is config.hits. Every member is measured on their own clock, so nothing needs aligning first."""
    members = len(velocities)
    signals = [motion(v, w) for v, w in zip(velocities, weights)]
    speed = np.stack([s[0] for s in signals])
    vel, weight = [s[1] for s in signals], [s[2] for s in signals]
    hits = group_hits(speed, fps, cfg.prominence)
    reach = max(1, int(round(cfg.max_offset_ms / 1000 * fps)))
    peak = np.full((members, len(hits)), np.nan)
    agreement = np.full((members, len(hits)), np.nan)
    for h, frame in enumerate(hits):
        candidates = [_candidates(speed[i], frame, reach, cfg.max_second_peak) for i in range(members)]
        # The group's direction is read where each member clearly peaks (at the hit frame if they don't).
        read_at = [int(c[0]) if c.size == 1 else int(frame) for c in candidates]
        for i in range(members):
            if candidates[i].size == 0:
                continue
            others = [j for j in range(members) if j != i]
            ref, trust = _direction(vel, weight, others, [read_at[j] for j in others])
            agree = np.nan_to_num([_cosine(vel[i][k], weight[i][k], ref, trust) for k in candidates[i]], nan=-1.0)
            order = np.argsort(-agree)
            agreement[i, h] = agree[0] if candidates[i].size == 1 else agree[order[0]]
            if candidates[i].size == 1:
                peak[i, h] = _sub_frame(speed[i], int(candidates[i][0]))
            elif agree[order[0]] >= cfg.min_agreement and agree[order[1]] <= agree[order[0]] - TIE_GAP:
                peak[i, h] = _sub_frame(speed[i], int(candidates[i][order[0]]))
    agree_frames = AGREE_MS / 1000 * fps
    return HitTiming(frames=hits.astype(float), offset_ms=relative_offsets(peak, cfg.min_others, agree_frames) * 1000.0 / fps, agreement=agreement)


def relative_offsets(peak: np.ndarray, min_others: int, tolerance: float = np.inf) -> np.ndarray:
    """(M, H) peak frames -> each member's offset at each hit, with 0 = the group's median member.

    The hit's moment is read from the *other* members (leave-one-out), each corrected by their
    own typical offset. Without that correction "the median of the others" depends on who the
    others are: with offsets 0, 40, -30, 90, 0 ms the -30 member's four others have median 20,
    so they would read -50. And a member unseen at one hit would move everyone else's reading
    there, the same trap section timing avoids. Typical offsets come from `typical_offsets`.

    The others must also agree on the moment: at least `min_others` of them within `tolerance`
    frames of their median. When they split between two moves (two peaking on one, two on the
    next), the median falls in between and every reading at that hit would be meaningless."""
    members = len(peak)
    corrected = peak - typical_offsets(peak)[:, None]
    offset = np.full_like(peak, np.nan)
    for i in range(members):
        others = corrected[[j for j in range(members) if j != i]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            reference = np.nanmedian(others, axis=0)
        agreeing = (np.abs(others - reference[None]) <= tolerance).sum(axis=0)
        offset[i] = np.where(agreeing >= min_others, peak[i] - reference, np.nan)
    return offset


def typical_offsets(peak: np.ndarray, sweeps: int = 10) -> np.ndarray:
    """Each member's typical offset (M,), 0 = the median member: Tukey's median polish of
    peak[i, h] = moment[h] + typical[i] + residual. Medians in both directions, so one odd hit or
    one odd member doesn't move it, and a member missing at some hits doesn't either. Delaying a
    member's whole performance changes only their own typical offset."""
    residual = peak.astype(float).copy()
    typical = np.zeros(len(peak))
    for _ in range(sweeps):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            moment = np.nan_to_num(np.nanmedian(residual, axis=0))
            residual -= moment[None]
            row = np.nan_to_num(np.nanmedian(residual, axis=1))
        residual -= row[:, None]
        typical += row
        if np.max(np.abs(row), initial=0.0) < 1e-6 and np.max(np.abs(moment), initial=0.0) < 1e-6:
            break
    seen = np.isfinite(peak).any(axis=1)
    if seen.any():
        typical -= np.median(typical[seen])
    return typical
