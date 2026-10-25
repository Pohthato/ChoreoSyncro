"""Clean landings: the moment a big move stops in a hold.

The user's definition of "on the beat" is the moment a move lands. On the BTS clip, timing every
movement event against the music's beat showed no lock at all (landings within 50 ms of a half
beat 39% of the time, 40% by chance); only big moves that end in stillness lined up somewhat. So
only those are taken: a speed peak at least `landing_peak` times the member's usual moving speed,
falling to `landing_hold` or less within `landing_within_s`.

The landing time is where the speed has fallen `landing_level` of the way from that hold back up to
the peak (0.25: three quarters of the way down), interpolated between frames. On a smooth stroke
that is about one frame before the move comes to rest; the half-way point would be two frames
early, and the exact stop can't be located (speed approaches zero gradually).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

PROMINENCE = 0.8   # a landing's peak must stand this far above its surroundings (x usual speed)
SMOOTH_FRAMES = 1.0


def _smooth(speed: np.ndarray) -> np.ndarray:
    known = np.isfinite(speed)
    num = gaussian_filter1d(np.where(known, speed, 0.0), SMOOTH_FRAMES)
    den = gaussian_filter1d(known.astype(float), SMOOTH_FRAMES)
    return np.where(known, num / np.maximum(den, 1e-9), np.nan)


def clean_landings(speed: np.ndarray, fps: float, cfg) -> np.ndarray:
    """Fractional frames of clean landings. speed (T,) in units of the member's usual moving
    speed (unison.member_speed), NaN where unseen. `cfg` is config.beat."""
    s = _smooth(np.asarray(speed, dtype=float))
    z = np.nan_to_num(s, nan=0.0)
    within = max(1, int(round(cfg.landing_within_s * fps)))
    peaks, _ = find_peaks(z, height=cfg.landing_peak, prominence=PROMINENCE)
    out = []
    for p in peaks:
        segment = s[p:p + within + 1]
        if len(segment) < 2 or not np.isfinite(segment).all():
            continue
        q = p + int(np.argmin(segment))
        if s[q] > cfg.landing_hold:
            continue
        level = s[q] + cfg.landing_level * (s[p] - s[q])
        j = p + 1 + int(np.argmax(s[p + 1:q + 1] <= level))
        out.append(j - 1 + (s[j - 1] - level) / (s[j - 1] - s[j]))
    return np.array(out)


MIN_LOCK_LANDINGS = 20
LOCK_P = 0.01
LOCK_MARGIN = 0.15  # share within 50 ms must beat chance by this much


def landing_lock(offsets_ms: np.ndarray, half_beat_ms: float) -> dict:
    """Do this video's landings line up with the beat at all?

    On the BTS clip they don't: 31 clean landings, 42% within 50 ms of a half beat where chance is
    40%, spread over the whole +-125 ms (Rayleigh p 0.65). Colouring each landing "on" or "off" the
    beat would then be a verdict the data can't support. So the viewer colours them only when the
    whole video's landings demonstrably lock: at least MIN_LOCK_LANDINGS, clustered on the
    half-beat circle (Rayleigh p below LOCK_P), and within 50 ms clearly more often than chance."""
    off = np.asarray(offsets_ms, dtype=float)
    n = int(off.size)
    chance = min(1.0, 100.0 / half_beat_ms)  # +-50 ms out of one half beat
    if n == 0:
        return {"n": 0, "within_50ms": None, "chance": chance, "p": None, "locked": False, "min_n": MIN_LOCK_LANDINGS}
    resultant = abs(np.mean(np.exp(2j * np.pi * off / half_beat_ms)))
    p = float(np.exp(-n * resultant ** 2))  # Rayleigh test, large-sample form
    within = float(np.mean(np.abs(off) <= 50.0))
    locked = n >= MIN_LOCK_LANDINGS and p < LOCK_P and within >= chance + LOCK_MARGIN
    return {"n": n, "within_50ms": round(within, 3), "chance": round(chance, 3), "p": round(p, 4), "locked": bool(locked),
            "min_n": MIN_LOCK_LANDINGS}
