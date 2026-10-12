"""Timeline heatmap: members down the side, time along the bottom, colour = offset.

Diverging blue (ahead) <- grey (in sync) -> red (behind). Windows the method could not measure
are hatched instead of coloured, so "no reading" never looks like "in sync".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..sync.measure import SyncResult
from .style import DIVERGING, INK_2, MUTED, SURFACE, apply_style


def render_heatmap(result: SyncResult, sections: list, path: Path, limit_ms: float = 150.0) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    apply_style()
    cmap = LinearSegmentedColormap.from_list("sync", DIVERGING)
    members = len(result.names)
    centers = result.window_centers_s
    hop = float(np.median(np.diff(centers))) if len(centers) > 1 else result.window_s
    edges = np.concatenate([centers - hop / 2, [centers[-1] + hop / 2]])
    values = np.where(result.reliable, result.offset_ms, np.nan)

    fig, ax = plt.subplots(figsize=(max(8, min(16, centers[-1] / 6)), 0.55 * members + 1.8))
    mesh = ax.pcolormesh(edges, np.arange(members + 1), np.ma.masked_invalid(values), cmap=cmap, vmin=-limit_ms, vmax=limit_ms, shading="flat")
    for i in range(members):
        missing = ~result.reliable[i]
        k = 0
        while k < len(missing):
            if not missing[k]:
                k += 1
                continue
            end = k
            while end < len(missing) and missing[end]:
                end += 1
            ax.add_patch(Rectangle((edges[k], i), edges[end] - edges[k], 1, facecolor=SURFACE, edgecolor=MUTED, hatch="////", linewidth=0))
            k = end
    for start, _, name in sections:
        ax.axvline(start, color=SURFACE, lw=2)
        ax.text(start, 1.01, f" {name}", color=INK_2, fontsize=8, va="bottom", transform=ax.get_xaxis_transform())
    ax.set_yticks(np.arange(members) + 0.5, result.names)
    ax.set_ylim(members, 0)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel("time (s)")
    ax.grid(False)
    ax.set_title("Timing offset per member (hatched = could not be measured)", loc="left", pad=20)
    bar = fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.03)
    bar.set_label("ms   ahead  <  0  >  behind", color=INK_2)
    bar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
