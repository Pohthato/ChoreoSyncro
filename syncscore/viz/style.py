"""One visual system for every figure: light charts, a dark skeleton video.

Colours come from a validated palette (categorical slots in fixed order, a blue <-> red
diverging pair with a grey midpoint, reserved status colours). Text stays in ink colours, never
in a series colour.
"""
from __future__ import annotations

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Diverging: ahead (blue) <- in sync (grey) -> behind (red).
DIVERGING = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f4a5a4", "#e34948", "#a32a2a"]
NO_DATA = "#d9d8d2"

# Dark surface for the skeleton video.
DARK_SURFACE = "#1a1a19"
DARK_INK = "#ffffff"
DARK_INK_2 = "#c3c2b7"
DARK_MUTED = "#898781"

# Status colours: they mean in sync / drifting / off. Always shown with a text label too.
GOOD = "#0ca30c"
WARNING = "#fab219"
CRITICAL = "#d03b3b"
SHAPE_FLAG = "#e87ba4"  # bones with a large pose error, drawn as a halo


def signed_ms(value: float) -> str:
    """'+12 ms' / '-8 ms' / '0 ms' (never '-0')."""
    rounded = int(round(value))
    return "0 ms" if rounded == 0 else f"{rounded:+d} ms"


def apply_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 9, "text.color": INK,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    })
