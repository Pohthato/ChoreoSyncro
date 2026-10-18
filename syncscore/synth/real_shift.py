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


def _section_relative(timing, member: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Member's section offset minus the median of the others' reliable ones, its half-interval, and where known."""
    others = [j for j in range(timing.offset_ms.shape[0]) if j != member]
    off = np.where(timing.reliable, timing.offset_ms, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rest = np.nanmedian(off[others], axis=0)
    rel = off[member] - rest
    half = (timing.high_ms[member] - timing.low_ms[member]) / 2
    return rel, half, np.isfinite(rel)


def _run(args):
    from ..sync.sections import combine_sections, hit_section_timing, section_timing
    from ..sync.summary import make_sections

    xy, conf, fps, config, member, shift = args
    if shift:
        xy, conf = delay_member(xy, conf, member, shift, fps)
    result, _, _ = analyze_keypoints(xy, conf, fps, config)
    sections = make_sections(config, result.n_frames / fps)
    timing = section_timing(result, sections, config)
    hit_timing = hit_section_timing(result, sections, config)
    sec = {"section": _section_relative(timing, member), "hit_section": _section_relative(hit_timing, member),
           "combined_section": _section_relative(combine_sections(timing, hit_timing), member)}
    return member, shift, _relative(result, member), sec, (result.hit_times_s, result.hit_offset_ms)


def real_shift_test(xy: np.ndarray, conf: np.ndarray, fps: float, config: Config, members: list[int], shifts: list[float], jobs: int = 8) -> dict:
    """Plant each shift on each member and score how well it comes back."""
    tasks = [(xy, conf, fps, config, m, 0.0) for m in members] + [(xy, conf, fps, config, m, s) for m in members for s in shifts]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        out = list(pool.map(_run, tasks))
    base = {m: (rel, sec, hits) for m, s, rel, sec, hits in out if s == 0.0}
    rows = []
    for m, s, (rel, ok), sec, hits in out:
        if s == 0.0:
            continue
        (rel0, ok0), sec0, hits0 = base[m]
        both = ok & ok0
        err = rel[both] - rel0[both] - s
        herr, hcov = _hit_errors(hits0, hits, m, s, fps)
        row = {"member": m, "shift_ms": s,
               "window_coverage": float(both.mean()), "window_mae_ms": _mean_abs(err), "window_wrong": _share(np.abs(err) > WRONG_MS, err),
               "hit_coverage": hcov, "hit_mae_ms": _mean_abs(herr), "hit_median_abs_ms": float(np.median(np.abs(herr))) if herr.size else None,
               "hit_wrong": _share(np.abs(herr) > WRONG_MS, herr)}
        for kind in ("section", "hit_section", "combined_section"):
            (srel, shalf, sok), (srel0, shalf0, sok0) = sec[kind], sec0[kind]
            sboth = sok & sok0
            serr = srel[sboth] - srel0[sboth] - s
            sbound = np.sqrt(shalf[sboth] ** 2 + shalf0[sboth] ** 2)  # the difference of two estimates
            row.update({
                f"{kind}_coverage": float(sboth.mean()), f"{kind}_mae_ms": _mean_abs(serr), f"{kind}_wrong": _share(np.abs(serr) > WRONG_MS, serr),
                f"{kind}_within_interval": _share(np.abs(serr) <= sbound, serr),
                f"{kind}_median_recovered_ms": float(np.median(srel[sboth] - srel0[sboth])) if sboth.any() else None,
            })
        rows.append(row)
    keys = [k for k in rows[0] if k not in ("member", "shift_ms")] if rows else []
    summary = {}
    for key in keys:
        vals = [r[key] for r in rows if r[key] is not None]
        summary[key] = float(np.mean(vals)) if vals else None
    return {"rows": rows, **summary}


def _hit_errors(base: tuple, shifted: tuple, member: int, shift_ms: float, fps: float) -> tuple[np.ndarray, float]:
    """Per-hit error of the planted shift, and the share of the original hits read in both runs.

    `base` and `shifted` are (hit times in s, (M, H) hit offsets). Delaying one member changes the
    group's median speed slightly, so hits are found again in the shifted run; a hit is the same
    one if it is within 2 frames of the original. Delaying a member can also move which member is
    the median one, and with it everyone's "0 ms" by the same amount: that common move is the
    median change of all the *other* members' offsets, and it is taken out."""
    (t0, off0), (t1, off1) = base, shifted
    pairs = [(k, int(np.argmin(np.abs(t1 - t)))) for k, t in enumerate(t0)] if len(t1) else []
    pairs = [(k, j) for k, j in pairs if abs(t1[j] - t0[k]) <= 2.0 / fps]
    if not pairs:
        return np.zeros(0), 0.0
    k, j = np.array(pairs).T
    change = off1[:, j] - off0[:, k]
    others = [i for i in range(len(off0)) if i != member]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        common = np.nanmedian(change[others])
    errors = change[member] - shift_ms - np.nan_to_num(common)
    errors = errors[np.isfinite(errors)]
    return errors, errors.size / len(t0)


def _mean_abs(err: np.ndarray) -> float | None:
    return float(np.abs(err).mean()) if err.size else None


def _share(mask: np.ndarray, err: np.ndarray) -> float | None:
    return float(mask.mean()) if err.size else None


def with_sync(config: Config, **changes) -> Config:
    return dataclasses.replace(config, sync=dataclasses.replace(config.sync, **changes))
