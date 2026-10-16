"""Known shifts injected into *real* keypoints: validation without ground-truth labels.

Real footage has no answer key: nobody knows how many ms late each member truly was. But we
can plant one. Take a real member's track, delay it by a known amount (33, 67, 133 ms), and
measure again. Whatever the member's true offset was, it cancels, and the measured offset
relative to the rest of the group must change by exactly the planted amount. This keeps
everything the synthetic benchmark leaves out: real pose noise, real styling differences
between dancers, occlusion, and real tracking errors.

Shifting one member also moves the group's median a little. So the check uses the member's
offset relative to the median of the *others*, which the shift doesn't touch.
"""
from __future__ import annotations

import dataclasses
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..config import Config
from ..pipeline import analyze_keypoints

WRONG_MS = 30.0


def delay_member(xy: np.ndarray, conf: np.ndarray, member: int, shift_ms: float, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Make `member` perform everything `shift_ms` later (fractional frames, linear in time)."""
    xy, conf = xy.copy(), conf.copy()
    frames = xy.shape[1]
    src = np.arange(frames) - shift_ms / 1000.0 * fps  # at frame t they do what they did at t - shift
    lo = np.floor(src).astype(int)
    w = src - lo
    valid = (lo >= 0) & (lo + 1 < frames)
    lo_c, hi_c = np.clip(lo, 0, frames - 1), np.clip(lo + 1, 0, frames - 1)
    moved = (1 - w)[:, None, None] * xy[member, lo_c] + w[:, None, None] * xy[member, hi_c]
    moved_conf = np.minimum(conf[member, lo_c], conf[member, hi_c])
    xy[member] = np.where(valid[:, None, None], moved, np.nan)
    conf[member] = np.where(valid[:, None], moved_conf, 0.0)
    return xy, conf


def _relative(result, member: int) -> tuple[np.ndarray, np.ndarray]:
    """Member's offset minus the median of the others, per window, and where that is measured."""
    others = [j for j in range(result.offset_ms.shape[0]) if j != member]
    off = np.where(result.reliable, result.offset_ms, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rest = np.nanmedian(off[others], axis=0)
    rel = off[member] - rest
    return rel, np.isfinite(rel)


def _run(args):
    xy, conf, fps, config, member, shift = args
    if shift:
        xy, conf = delay_member(xy, conf, member, shift, fps)
    result, _, _ = analyze_keypoints(xy, conf, fps, config)
    return member, shift, _relative(result, member)


def real_shift_test(xy: np.ndarray, conf: np.ndarray, fps: float, config: Config, members: list[int], shifts: list[float], jobs: int = 8) -> dict:
    """Plant each shift on each member and score how well it comes back."""
    tasks = [(xy, conf, fps, config, m, 0.0) for m in members] + [(xy, conf, fps, config, m, s) for m in members for s in shifts]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        out = list(pool.map(_run, tasks))
    base = {m: rel for m, s, rel in out if s == 0.0}
    rows = []
    for m, s, (rel, ok) in out:
        if s == 0.0:
            continue
        rel0, ok0 = base[m]
        both = ok & ok0
        change = rel[both] - rel0[both]
        err = change - s
        rows.append({"member": m, "shift_ms": s, "coverage": float(both.mean()),
                     "median_recovered_ms": float(np.median(change)) if change.size else None,
                     "mae_ms": float(np.abs(err).mean()) if err.size else None,
                     "confidently_wrong": float((np.abs(err) > WRONG_MS).mean()) if err.size else None})
    def avg(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(vals)) if vals else None
    return {"rows": rows, "coverage": avg("coverage"), "mae_ms": avg("mae_ms"), "confidently_wrong": avg("confidently_wrong")}


def with_sync(config: Config, **changes) -> Config:
    return dataclasses.replace(config, sync=dataclasses.replace(config.sync, **changes))
