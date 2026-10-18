"""One image per member: headline numbers, offset over time, timing and shape by body part."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..features.bones import PARTS
from ..sync.measure import SyncResult
from ..sync.sections import SectionTiming
from .style import GRID, INK, INK_2, MUTED, SERIES, apply_style, signed_ms


def _signed(ms: float | None) -> str:
    if ms is None:
        return "no reading"
    if abs(ms) < 0.5:
        return "on the beat"
    return f"{abs(ms):.0f} ms {'behind' if ms > 0 else 'ahead'}"


def render_report_card(result: SyncResult, timing: SectionTiming, summary_member: dict, i: int, path: Path, in_sync_ms: float, shape_threshold: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_style()
    overall = summary_member["overall"]
    fig = plt.figure(figsize=(7.5, 8.6))
    grid = fig.add_gridspec(4, 2, height_ratios=[0.9, 1.4, 1.2, 1.6], hspace=0.65, wspace=0.45)

    head = fig.add_subplot(grid[0, :])
    head.axis("off")
    head.text(0, 0.95, summary_member["member"], fontsize=20, fontweight="bold", color=INK, va="top")
    measured = timing.reliable[i]
    spread = np.ptp(timing.offset_ms[i, measured]) if measured.sum() > 1 else None
    stats = [
        ("typical (median section)", _signed(overall["section_typical_offset_ms"])),
        ("in sync", "-" if overall["sections_in_sync_share"] is None else f"{overall['sections_in_sync_share'] * 100:.0f}% of sections"),
        ("range over sections", "-" if spread is None else f"{spread:.0f} ms"),
        ("measured", f"{overall['sections_measured_share'] * 100:.0f}% of sections"),
    ]
    for k, (label, value) in enumerate(stats):
        head.text(k * 0.26, 0.38, value, fontsize=11.5, fontweight="bold", color=INK, va="top")
        head.text(k * 0.26, 0.02, label, fontsize=9, color=INK_2, va="bottom")

    ax = fig.add_subplot(grid[1, :])
    ax.axhspan(-in_sync_ms, in_sync_ms, color=GRID, alpha=0.6, lw=0)
    ax.axhline(0, color=MUTED, lw=1)
    for s in np.flatnonzero(measured):
        a, b = timing.starts_s[s], timing.ends_s[s]
        ax.fill_between([a, b], timing.low_ms[i, s], timing.high_ms[i, s], color=SERIES[0], alpha=0.2, lw=0)
        ax.plot([a, b], [timing.offset_ms[i, s]] * 2, color=SERIES[0], lw=2.5, solid_capstyle="butt")
    strict = result.reliable[i]
    ax.plot(result.window_centers_s[strict], result.offset_ms[i, strict], "o", ms=3, color=SERIES[1], label="single windows that pass the strict gates")
    ax.set_title(f"Offset per section, 90% interval shaded (band = in sync, ±{in_sync_ms:.0f} ms)", loc="left")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("ms (+ behind)")
    ax.set_xlim(0, timing.ends_s[-1])
    if strict.any():
        ax.legend(frameon=False, loc="upper left", fontsize=8)

    ax = fig.add_subplot(grid[2, 0])
    timing = [overall["timing_offset_by_part_ms"][p] for p in PARTS]
    vals = [0 if v is None else v for v in timing]
    ax.barh(PARTS, vals, color=SERIES[0], height=0.55)
    ax.axvline(0, color=MUTED, lw=1)
    for k, v in enumerate(timing):
        ax.text(vals[k], k, "  -" if v is None else f" {signed_ms(v)[:-3]} ", va="center", ha="left" if vals[k] >= 0 else "right", color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_title("Timing by body part (ms)", loc="left")
    lim = max(10.0, max(abs(v) for v in vals) * 1.6)
    ax.set_xlim(-lim, lim)
    ax.grid(axis="y", visible=False)

    ax = fig.add_subplot(grid[2, 1])
    shape = [overall["shape_excess_by_part_deg"][p] for p in PARTS]
    vals = [0 if v is None else v for v in shape]
    ax.barh(PARTS, vals, color=SERIES[0], height=0.55)
    ax.axvline(0, color=MUTED, lw=1)
    ax.axvline(shape_threshold, color=MUTED, lw=1, ls="--")
    for k, v in enumerate(shape):
        ax.text(vals[k], k, "  -" if v is None else f" {v:+.1f}°", va="center", ha="left" if vals[k] >= 0 else "right", color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_title("Pose error above the group's typical", loc="left")
    span = max(shape_threshold * 1.3, max(abs(v) for v in vals) * 1.4)
    ax.set_xlim(min(-5.0, min(vals) * 1.4), span)
    ax.grid(axis="y", visible=False)
    ax.text(shape_threshold, len(PARTS) - 0.35, f" {shape_threshold:.0f}° = off", color=MUTED, fontsize=8, va="top")

    text = fig.add_subplot(grid[3, :])
    text.axis("off")
    lines = ["Worst timing moments"]
    for m in overall["worst_timing_moments"]:
        part = f", mostly the {m['part_furthest_off']}" if m["part_furthest_off"] else ""
        lines.append(f"   {m['time_s']:6.1f} s   {_signed(m['offset_ms'])}{part}")
    lines.append("Worst pose moments")
    if overall["worst_shape_moments"]:
        for m in overall["worst_shape_moments"]:
            note = " (likely a left/right label swap)" if m.get("likely_label_swap") else ""
            lines.append(f"   {m['time_s']:6.1f} s   {m['bone'].replace('_', ' ')} {m['above_group_deg']:.0f}° beyond the group{note}")
    else:
        lines.append(f"   none more than {shape_threshold:.0f}° beyond the group")
    part = overall["timing_part_most_often_off"]
    shape_part = overall["shape_part_most_often_off"]
    def note(kind, name, unit):
        if not name:
            return "nothing more than the others"
        return f"{name}, {overall[f'{kind}_part_off_share'] * 100:.0f}% of {unit} (group {overall[f'{kind}_part_off_share_group'] * 100:.0f}%)"

    lines.append(f"Off more than the others: timing - {note('timing', part, 'windows')}; pose - {note('shape', shape_part, 'frames')}")
    facing = overall.get("facing_differently_share") or 0.0
    if facing > 0:
        lines.append(f"Facing a different way to the group (pose not compared): {facing * 100:.0f}% of frames")
    step = 1.0 / (len(lines) + 0.5)
    for k, line in enumerate(lines):
        bold = line in ("Worst timing moments", "Worst pose moments") or line.startswith("Most often")
        text.text(0, 1 - k * step, line, fontsize=9.5, color=INK if bold else INK_2, fontweight="bold" if bold else "normal", va="top", family="monospace" if not bold else None)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
