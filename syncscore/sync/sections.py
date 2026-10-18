"""Timing per section: one robust offset per member per stretch of the song, with its uncertainty.

On real dance a single 2-second window is a weak witness. Most of the time is individually
styled grooving, so the cross-correlation is often low, and windows strict enough to trust
cover only a few percent of the clip. Planting known shifts into real motion showed that the
*median* over many looser windows is accurate, even when single windows are not. So:

- every window that passes the hard gates (enough motion, member visible, no ambiguous second
  peak) and a looser correlation floor counts,
- a section's offset is the median of its windows,
- "0 ms" is the clip's median member, and it must not change meaning when someone is missing. If
  each section were re-centred on whoever is visible, hiding one member would shift everyone else
  (seen in testing: 20 ms). So each section's clock drift is estimated by comparing every visible
  member with *their own* typical offset, and only that drift is removed,
- the uncertainty is a 90% interval from a **joint block bootstrap**. Neighbouring windows overlap
  (2 s long, every 0.25 s), so they are not independent: resampling single windows would claim far
  more confidence than the data has. Non-overlapping blocks are resampled instead, the same
  blocks for every member at once, and the re-centring is redone in every draw, so the noise of
  the reference itself ends up in every member's interval,
- a section with too few windows, too few independent blocks (`min_blocks`; with 2 or 3 blocks a
  bootstrap interval is meaningless: it came out far too narrow in testing), or an interval
  wider than `max_ci_ms`, is "no reading".
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from ..config import Config
from .measure import SyncResult


@dataclass
class SectionTiming:
    starts_s: np.ndarray   # (S,)
    ends_s: np.ndarray     # (S,)
    names: list
    offset_ms: np.ndarray  # (M, S) + = behind the group's median member
    low_ms: np.ndarray     # (M, S) 90% interval
    high_ms: np.ndarray    # (M, S)
    windows: np.ndarray    # (M, S) how many windows went in
    blocks: np.ndarray     # (M, S) how many independent (non-overlapping) stretches they came from
    enough: np.ndarray     # (M, S) enough windows and blocks for an estimate to mean something
    reliable: np.ndarray   # (M, S) enough, and the interval is narrow enough to state
    from_hits: np.ndarray | None = None  # (M, S) True where the reading comes from the group's hits


def usable_windows(result: SyncResult, config: Config) -> np.ndarray:
    """(M, K) windows good enough to be pooled into a section estimate."""
    sc = config.sync
    return (np.isfinite(result.offset_ms) & (np.nan_to_num(result.peak) >= config.sections.min_peak)
            & (result.motion >= sc.min_motion) & (result.coverage >= sc.min_coverage)
            & (np.nan_to_num(result.margin, nan=-1.0) >= sc.min_margin))


def _block_bootstrap_median(values: np.ndarray, block: int, rng: np.random.Generator, draws: int = 400) -> tuple[float, float]:
    """5th and 95th percentile of the median under a block bootstrap (blocks of `block` windows)."""
    blocks = [values[i:i + block] for i in range(0, len(values), block)]
    n = len(blocks)
    medians = np.empty(draws)
    for d in range(draws):
        pick = rng.integers(0, n, size=n)
        medians[d] = np.median(np.concatenate([blocks[k] for k in pick]))
    return float(np.percentile(medians, 5)), float(np.percentile(medians, 95))


def _section_medians(offsets: np.ndarray, usable: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-member median over the usable windows given (M, k) -> (M,), and window counts."""
    count = usable.sum(axis=1)
    med = np.array([np.median(offsets[i, usable[i]]) if count[i] else np.nan for i in range(len(offsets))])
    return med, count


def _drift(med: np.ndarray, count: np.ndarray, typical: np.ndarray, min_windows: int) -> float:
    """How far this section's clock sits from the clip's, judged by members with enough windows."""
    voters = (count >= min_windows) & np.isfinite(med) & np.isfinite(typical)
    return float(np.median(med[voters] - typical[voters])) if voters.any() else 0.0


def section_timing(result: SyncResult, sections: list, config: Config, draws: int = 400) -> SectionTiming:
    cfg = config.sections
    usable = usable_windows(result, config)
    members, n_sections = len(result.names), len(sections)
    hop = float(np.median(np.diff(result.window_centers_s))) if len(result.window_centers_s) > 1 else result.window_s
    block = max(1, int(round(result.window_s / hop)))  # windows per non-overlapping stretch
    spans = [np.flatnonzero((result.window_centers_s >= a) & (result.window_centers_s < b)) for a, b, _ in sections]

    raw = np.full((members, n_sections), np.nan)
    count = np.zeros((members, n_sections), dtype=int)
    n_blocks = np.zeros((members, n_sections), dtype=int)
    for s, idx in enumerate(spans):
        raw[:, s], count[:, s] = _section_medians(result.offset_ms[:, idx], usable[:, idx])
        for k in range(0, len(idx), block):
            n_blocks[:, s] += usable[:, idx[k:k + block]].any(axis=1)
    enough = (count >= cfg.min_windows) & (n_blocks >= cfg.min_blocks)
    with np.errstate(all="ignore"):
        typical = np.array([np.median(raw[i, enough[i]]) if enough[i].any() else np.nan for i in range(members)])
    finite_typical = typical[np.isfinite(typical)]
    zero = float(np.median(finite_typical)) if finite_typical.size else 0.0  # the clip's median member

    offset = np.full((members, n_sections), np.nan)
    low, high = offset.copy(), offset.copy()
    rng = np.random.default_rng(0)  # fixed: the same input always gives the same intervals
    for s, idx in enumerate(spans):
        drift = _drift(raw[:, s], count[:, s], typical, cfg.min_windows)
        offset[:, s] = raw[:, s] - drift - zero
        if len(idx) == 0:
            continue
        blocks = [idx[k:k + block] for k in range(0, len(idx), block)]
        samples = np.full((draws, members), np.nan)
        for d in range(draws):
            pick = np.concatenate([blocks[k] for k in rng.integers(0, len(blocks), size=len(blocks))])
            med, cnt = _section_medians(result.offset_ms[:, pick], usable[:, pick])
            samples[d] = med - _drift(med, cnt, typical, max(1, cfg.min_windows // 2)) - zero
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            low[:, s] = np.nanpercentile(samples, 5, axis=0)
            high[:, s] = np.nanpercentile(samples, 95, axis=0)
    narrow = np.nan_to_num(high - low, nan=np.inf) <= cfg.max_ci_ms
    return SectionTiming(
        starts_s=np.array([a for a, _, _ in sections]), ends_s=np.array([b for _, b, _ in sections]),
        names=[n for _, _, n in sections], offset_ms=offset, low_ms=low, high_ms=high, windows=count, blocks=n_blocks, enough=enough,
        reliable=enough & narrow & np.isfinite(offset),
    )


def combine_sections(windows: SectionTiming, hits: SectionTiming) -> SectionTiming:
    """Per member and section, the reliable reading with the narrower interval (windows or hits).

    The two come from different evidence (correlation over all the dancing vs. the moments the
    whole group hits together), so they are not averaged: a section reads whichever one is more
    certain, and `from_hits` says which. Used by the planted-shift test; the report uses
    window sections only (hits added no reliable readings on real footage, see hits.py)."""
    width_w = np.where(windows.reliable, windows.high_ms - windows.low_ms, np.inf)
    width_h = np.where(hits.reliable, hits.high_ms - hits.low_ms, np.inf)
    use_hits = hits.reliable & (width_h < width_w)

    def pick(a, b):
        return np.where(use_hits, b, a)

    return SectionTiming(
        starts_s=windows.starts_s, ends_s=windows.ends_s, names=windows.names,
        offset_ms=pick(windows.offset_ms, hits.offset_ms), low_ms=pick(windows.low_ms, hits.low_ms), high_ms=pick(windows.high_ms, hits.high_ms),
        windows=pick(windows.windows, hits.windows), blocks=pick(windows.blocks, hits.blocks),
        enough=windows.enough | hits.enough, reliable=windows.reliable | hits.reliable, from_hits=use_hits,
    )


def hit_section_timing(result: SyncResult, sections: list, config: Config, draws: int = 400) -> SectionTiming:
    """The same per-section answer, from the group's hits (hits.py) instead of windows.

    Each hit offset is already relative to the median of the *other* members at that hit, so
    there is no clock drift to remove. Hits are separate events (at least 0.3 s apart, each
    read from a few frames around its own peak), so they are resampled one by one, the same
    hits for every member in each draw. `windows` and `blocks` both count the matched hits.
    """
    cfg = config.hits
    offsets = result.hit_offset_ms
    members, n_sections = offsets.shape[0], len(sections)
    offset = np.full((members, n_sections), np.nan)
    low, high = offset.copy(), offset.copy()
    count = np.zeros((members, n_sections), dtype=int)
    rng = np.random.default_rng(0)
    for s, (a, b, _) in enumerate(sections):
        idx = np.flatnonzero((result.hit_times_s >= a) & (result.hit_times_s < b))
        if idx.size == 0:
            continue
        values = offsets[:, idx]
        count[:, s] = np.isfinite(values).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            offset[:, s] = np.nanmedian(values, axis=1)
            samples = np.stack([np.nanmedian(values[:, rng.integers(0, idx.size, size=idx.size)], axis=1) for _ in range(draws)])
            low[:, s] = np.nanpercentile(samples, 5, axis=0)
            high[:, s] = np.nanpercentile(samples, 95, axis=0)
    enough = count >= cfg.min_hits
    narrow = np.nan_to_num(high - low, nan=np.inf) <= cfg.max_ci_ms
    return SectionTiming(
        starts_s=np.array([a for a, _, _ in sections]), ends_s=np.array([b for _, b, _ in sections]),
        names=[n for _, _, n in sections], offset_ms=np.where(count > 0, offset, np.nan), low_ms=low, high_ms=high,
        windows=count, blocks=count.copy(), enough=enough, reliable=enough & narrow & np.isfinite(offset),
        from_hits=np.ones((members, n_sections), dtype=bool),
    )
