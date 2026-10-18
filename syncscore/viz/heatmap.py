"""Timeline heatmap: members down the side, the song's sections along the bottom, colour = offset.

Each cell is one member in one section: the robust section offset (see sync/sections.py),
written in the cell with its uncertainty so nobody has to read a number off a colour.
Diverging blue (ahead) <- grey (in sync) -> red (behind). Sections without a reliable reading are
hatched instead of coloured, so "no reading" never looks like "in sync". If there was an estimate
but its interval was too wide to trust, it is written in muted grey (≈+8 ±32), so "measured but
too uncertain" stays distinguishable from "nothing to measure".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..sync.sections import SectionTiming
from .style import DIVERGING, INK, INK_2, MUTED, SURFACE, apply_style, signed_ms


def render_heatmap(timing: SectionTiming, names: list, path: Path, limit_ms: float = 150.0) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle

    apply_style()
    cmap = LinearSegmentedColormap.from_list("sync", DIVERGING)
    norm = Normalize(-limit_ms, limit_ms)
    members, n_sections = timing.offset_ms.shape
    duration = float(timing.ends_s[-1])
    fig, ax = plt.subplots(figsize=(max(8, min(18, duration / 8)), 0.6 * members + 1.9))
    show_text = duration / max(n_sections, 1) * (fig.get_figwidth() / duration) > 0.45  # inches per cell
    for i in range(members):
        for s in range(n_sections):
            x, width = timing.starts_s[s], timing.ends_s[s] - timing.starts_s[s]
            if timing.reliable[i, s]:
                value = timing.offset_ms[i, s]
                ax.add_patch(Rectangle((x, i), width, 1, facecolor=cmap(norm(value)), edgecolor=SURFACE, linewidth=1.5))
                if show_text:
                    half = (timing.high_ms[i, s] - timing.low_ms[i, s]) / 2
                    ink = SURFACE if abs(value) > 0.6 * limit_ms else INK
                    ax.text(x + width / 2, i + 0.42, signed_ms(value)[:-3], ha="center", va="center", fontsize=8, color=ink, fontweight="bold")
                    ax.text(x + width / 2, i + 0.75, f"±{half:.0f}", ha="center", va="center", fontsize=6.5, color=ink)
            else:
                ax.add_patch(Rectangle((x, i), width, 1, facecolor=SURFACE, edgecolor=MUTED, hatch="////", linewidth=0))
                value = timing.offset_ms[i, s]
                if show_text and np.isfinite(value) and timing.enough[i, s]:
                    half = (timing.high_ms[i, s] - timing.low_ms[i, s]) / 2
                    ax.text(x + width / 2, i + 0.5, f"≈{signed_ms(value)[:-3]} ±{half:.0f}", ha="center", va="center", fontsize=7, color=MUTED,
                            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    ax.set_xlim(0, duration)
    ax.set_ylim(members, 0)
    ax.set_yticks(np.arange(members) + 0.5, names)
    ax.set_xlabel("time (s)")
    ax.grid(False)
    ax.set_title("Timing per member and section, ms (hatched = no reliable reading; grey ≈ value = too uncertain)", loc="left", pad=10)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01, fraction=0.03)
    bar.set_label("ms   ahead  <  0  >  behind", color=INK_2)
    bar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
