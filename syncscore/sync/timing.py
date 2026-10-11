"""Timing offsets: windowed normalised cross-correlation of bone velocities.

For each window of the reference (the consensus), slide the member's signal over a range of
lags and keep the lag where the two line up best. Positive lag = the member does the move
later than the group (behind); negative = ahead.

Why cross-correlation and not dynamic time warping (DTW):
- The question is "by how many milliseconds is this member off during this stretch". That is one
  number per window, which is exactly what the cross-correlation peak is.
- DTW returns a warping path that is free to stretch time to make the poses match. It absorbs
  part of a *shape* error into timing, and keeping the two apart is the point of the tool.
  It also needs a band constraint and has no natural sub-frame estimate.
- Cross-correlation gives a sub-frame estimate for free (fit a parabola through the peak) and
  a confidence (how high the peak is).

Why **velocity** and not position: being in sync means starting and stopping each movement at the
same moment. Velocity peaks exactly at those transitions. It also makes timing blind to a
constant shape error (a limb held 20 degrees off has the same velocity as one held correctly),
which is what keeps timing and shape errors apart.

Periodic choreography (the same move every beat) gives several peaks of the correlation. A
Viterbi pass over the windows picks the lag sequence with the best total score minus a penalty
for jumping, so one ambiguous window follows its neighbours instead of jumping to an alias.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Windows:
    starts: np.ndarray  # first frame of each window
    length: int

    @property
    def centers(self) -> np.ndarray:
        return self.starts + (self.length - 1) / 2


def make_windows(n_frames: int, window: int, hop: int) -> Windows:
    window = min(window, n_frames)
    return Windows(starts=np.arange(0, n_frames - window + 1, max(1, hop)), length=window)


def velocity(u: np.ndarray, w: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference velocity of unit vectors: (T, B, 2) per second. Its size is the angular
    speed of the bone in radians per second."""
    v = np.full_like(u, np.nan)
    v[1:-1] = (u[2:] - u[:-2]) * fps / 2
    vw = np.zeros_like(w)
    vw[1:-1] = np.minimum(w[2:], w[:-2])
    known = np.isfinite(v[..., 0]) & (vw > 0)
    return np.where(known[..., None], v, np.nan), np.where(known, vw, 0.0)


def parabolic_peak(left: float, center: float, right: float) -> float:
    """Fractional position (-0.5..0.5) of the top of a parabola through three samples."""
    denom = left - 2 * center + right
    if not np.isfinite(denom) or denom >= 0:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denom, -0.5, 0.5))


def viterbi_path(scores: np.ndarray, penalty_per_step: float) -> np.ndarray:
    """Lag index per window maximising sum(score) - penalty * sum(|lag change|)."""
    windows, states = scores.shape
    index = np.arange(states)
    jump = penalty_per_step * np.abs(index[:, None] - index[None, :])
    best = scores[0].copy()
    back = np.zeros((windows, states), dtype=int)
    for k in range(1, windows):
        candidates = best[:, None] - jump  # from state i (rows) to state j (columns)
        back[k] = np.argmax(candidates, axis=0)
        best = candidates[back[k], index] + scores[k]
    path = np.empty(windows, dtype=int)
    path[-1] = int(np.argmax(best))
    for k in range(windows - 1, 0, -1):
        path[k - 1] = back[k, path[k]]
    return path


@dataclass
class TimingResult:
    lag: np.ndarray       # (K,) frames, fractional; NaN where there was no data
    peak: np.ndarray      # (K,) normalised correlation at the chosen lag (1 = identical motion)
    motion: np.ndarray    # (K,) RMS angular speed of the reference, rad/s
    coverage: np.ndarray  # (K,) share of the window where both member and reference were seen
    margin: np.ndarray    # (K,) chosen peak minus the best separate peak (small = ambiguous)
    reliable: np.ndarray  # (K,) bool
    scores: np.ndarray    # (K, 2L+1) correlation per lag


def _flatten(v: np.ndarray, w: np.ndarray, bones: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    if bones is not None:
        v, w = v[:, bones], w[:, bones]
    frames = len(v)
    values = np.nan_to_num(v.reshape(frames, -1))
    weights = np.repeat(w, 2, axis=1)
    return values, np.where(np.isfinite(v.reshape(frames, -1)), weights, 0.0)


def estimate_lags(
    member_v: np.ndarray, member_w: np.ndarray, ref_v: np.ndarray, ref_w: np.ndarray, windows: Windows,
    max_lag: int, penalty_per_step: float, min_motion: float, min_peak: float, min_coverage: float,
    min_margin: float = 0.0, bones: np.ndarray | None = None,
) -> TimingResult:
    """Lag of the member relative to the reference in every window (both on the same frame clock).

    member_v/ref_v (T, B, 2) velocities, member_w/ref_w (T, B) weights. `bones` restricts the
    measurement to some bones (one body part).
    """
    a, wa = _flatten(member_v, member_w, bones)
    b, wb = _flatten(ref_v, ref_w, bones)
    frames = len(b)
    pad = ((max_lag, max_lag), (0, 0))
    a, wa = np.pad(a, pad), np.pad(wa, pad)
    n, length = len(windows.starts), windows.length
    lags = 2 * max_lag + 1
    starts = windows.starts

    def window_sums(per_frame: np.ndarray) -> np.ndarray:
        # Windows overlap (2 s long, every 0.25 s), so sum each frame once and take every
        # window's total as a difference of the running sum.
        total = np.concatenate([np.zeros((1,) + per_frame.shape[1:]), np.cumsum(per_frame, axis=0)])
        return total[starts + length] - total[starts]

    num, energy_a, energy_b = np.empty((frames, lags)), np.empty((frames, lags)), np.empty((frames, lags))
    for j in range(lags):  # j = max_lag + lag; the member shifted by `lag` frames
        seg, seg_w = a[j:j + frames], wa[j:j + frames]
        joint = seg_w * wb
        num[:, j] = np.sum(joint * seg * b, axis=1)
        energy_a[:, j] = np.sum(joint * seg**2, axis=1)
        energy_b[:, j] = np.sum(joint * b**2, axis=1)
    num, energy_a, energy_b = window_sums(num), window_sums(energy_a), window_sums(energy_b)
    den = np.sqrt(energy_a * energy_b)
    with np.errstate(invalid="ignore", divide="ignore"):
        scores = np.where(den > 1e-12, num / den, np.nan)
        ref_weight = window_sums(wb.sum(axis=1))
        motion = np.where(ref_weight > 0, np.sqrt(2 * window_sums(np.sum(wb * b**2, axis=1)) / ref_weight), 0.0)
    both = ((wa[max_lag:max_lag + frames] > 0) & (wb > 0)).any(axis=1)
    coverage = window_sums(both.astype(float)) / length

    informative = np.isfinite(scores).any(axis=1) & (motion >= min_motion)
    path_scores = np.where(informative[:, None], np.nan_to_num(scores, nan=-1.0), 0.0)
    path = viterbi_path(path_scores, penalty_per_step) if n else np.zeros(0, dtype=int)

    lag = np.full(n, np.nan)
    peak = np.full(n, np.nan)
    margin = np.full(n, np.nan)
    for k, j in enumerate(path):
        row = scores[k]
        if not np.isfinite(row).any():
            continue
        frac = parabolic_peak(row[j - 1], row[j], row[j + 1]) if 0 < j < lags - 1 else 0.0
        lag[k] = j - max_lag + frac
        peak[k] = row[j]
        margin[k] = row[j] - _best_other_peak(row, j)
    reliable = (informative & np.isfinite(lag) & (np.nan_to_num(peak) >= min_peak) & (coverage >= min_coverage)
                & (np.nan_to_num(margin, nan=-1.0) >= min_margin))
    return TimingResult(lag=lag, peak=peak, motion=motion, coverage=coverage, margin=margin, reliable=reliable, scores=scores)


def _best_other_peak(row: np.ndarray, chosen: int) -> float:
    """Highest local maximum of the correlation that is a separate peak from the chosen one.

    A move that repeats faster than the search range (a fast groove) gives several equally good
    lags. Then the timing is ambiguous and must not be reported as if it were known."""
    filled = np.nan_to_num(row, nan=-1.0)
    # Walk down both sides of the chosen peak; anything beyond the valleys is another peak.
    left = chosen
    while left > 0 and filled[left - 1] <= filled[left]:
        left -= 1
    right = chosen
    while right < len(filled) - 1 and filled[right + 1] <= filled[right]:
        right += 1
    others = np.concatenate([filled[:left], filled[right + 1:]])
    return float(others.max()) if others.size else -1.0
