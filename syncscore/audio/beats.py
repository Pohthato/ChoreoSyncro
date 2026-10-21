"""The beat of the music, from the video's soundtrack.

Only the music is used here; nothing about the dancers. The steps are the standard ones:
1. **Onset curve:** how suddenly the sound changes, frame by frame (positive spectral flux of the
   log-magnitude spectrogram, its local average removed).
2. **Tempo:** the lag at which that curve repeats best (autocorrelation), with a soft preference
   for tempos near 120 BPM so a beat isn't mistaken for half or twice the tempo.
3. **Beats:** the sequence of onsets that is both strong and evenly spaced (Ellis's dynamic
   programming).
4. **A steady grid:** on the BTS clip the tracked beats sat within 10-20 ms of a strict 120 BPM grid,
   but kept flipping between the beat and the off-beat (half a beat apart). The *half-beat* grid is
   the same either way, so it is fitted to the tracked beats (period refined to the best fit, phase
   from their circular mean). Of its two phases, the beat is the one where the sound is louder (the
   kick); the other is the "and". The grid is `steady` only if most tracked beats sit on it.

Decoding uses PyAV (an optional dependency). Without it, or without an audio stream, there is no
beat and the viewer says so.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import stft

SR = 22050
HOP = 256
NFFT = 1024
STEADY_MS = 40.0     # a tracked beat within this of the half-beat grid is "on" it
STEADY_SHARE = 0.7   # ...and this share of them must be, for the grid to be called steady
MIN_SECONDS = 10.0


@dataclass
class BeatGrid:
    tempo_bpm: float
    half_beat_s: float
    beat_phase_s: float   # the time of a beat (not an off-beat), modulo the beat period
    steady: bool
    beat_share: float     # share of tracked beats on the half-beat grid

    def offset_ms(self, times_s: np.ndarray) -> np.ndarray:
        """Signed ms from each time to the nearest half beat (+ = after it), within +-half_beat/2."""
        hb = self.half_beat_s
        return ((np.asarray(times_s, dtype=float) - self.beat_phase_s + hb / 2) % hb - hb / 2) * 1000.0

    def beats_between(self, t0: float, t1: float) -> list[tuple[float, bool]]:
        """Half beats in [t0, t1): (time, True for a beat / False for an off-beat)."""
        hb = self.half_beat_s
        first = int(np.ceil((t0 - self.beat_phase_s) / hb))
        out = []
        k = first
        while self.beat_phase_s + k * hb < t1:
            out.append((self.beat_phase_s + k * hb, k % 2 == 0))
            k += 1
        return out


def load_audio(path: Path, sr: int = SR) -> np.ndarray | None:
    """Mono float32 samples at `sr`, aligned so sample 0 is the start of the video; None without
    PyAV or without an audio stream."""
    try:
        import av
    except ImportError:
        return None
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                return None
            stream = container.streams.audio[0]
            video = container.streams.video[0] if container.streams.video else None
            lead = 0.0  # audio starting after the picture: pad with silence so times line up
            if stream.start_time is not None and video is not None and video.start_time is not None:
                lead = float(stream.start_time * stream.time_base) - float(video.start_time * video.time_base)
            resampler = av.AudioResampler(format="flt", layout="mono", rate=sr)
            chunks = []
            for frame in container.decode(stream):
                chunks.extend(out.to_ndarray().ravel() for out in resampler.resample(frame))
            chunks.extend(out.to_ndarray().ravel() for out in resampler.resample(None))
    except (av.error.FFmpegError, OSError):
        return None
    if not chunks:
        return None
    x = np.concatenate(chunks).astype(np.float32)
    if lead > 0:
        x = np.concatenate([np.zeros(int(round(lead * sr)), np.float32), x])
    elif lead < 0:
        x = x[int(round(-lead * sr)):]
    return x


def onset_envelope(x: np.ndarray, sr: int = SR, hop: int = HOP) -> tuple[np.ndarray, np.ndarray]:
    _, _, z = stft(x, fs=sr, nperseg=NFFT, noverlap=NFFT - hop, boundary=None, padded=False)
    s = np.log1p(100.0 * np.abs(z))
    flux = np.maximum(np.diff(s, axis=1), 0.0).sum(axis=0)
    env = np.maximum(flux - uniform_filter1d(flux, 40), 0.0)
    times = (np.arange(len(env)) + 1) * hop / sr + NFFT / 2 / sr
    return env, times


def _tempo_lag(env: np.ndarray, frames_per_s: float) -> float:
    centred = env - env.mean()
    ac = np.correlate(centred, centred, mode="full")[len(env) - 1:]
    lags = np.arange(len(ac))
    with np.errstate(divide="ignore"):
        bpm = 60.0 * frames_per_s / np.maximum(lags, 1)
    prior = np.exp(-0.5 * np.log2(bpm / 120.0) ** 2)
    score = np.where((bpm >= 60) & (bpm <= 200), ac * prior, 0.0)
    best = int(np.argmax(score))
    y0, y1, y2 = score[best - 1], score[best], score[best + 1]
    denom = y0 - 2 * y1 + y2
    return best + (0.5 * (y0 - y2) / denom if denom < 0 else 0.0)


def _track(env: np.ndarray, period: float, tightness: float = 100.0) -> np.ndarray:
    """Frames of beats: strong onsets about `period` frames apart (Ellis 2007)."""
    lo, hi = max(1, int(round(period / 2))), int(round(2 * period))
    gaps = np.arange(lo, hi + 1)
    cost = -tightness * np.log(gaps / period) ** 2
    n = len(env)
    score = env.astype(float).copy()
    back = np.full(n, -1)
    for i in range(lo, n):
        prev = i - gaps
        ok = prev >= 0
        cand = score[prev[ok]] + cost[ok]
        k = int(np.argmax(cand))
        score[i] = env[i] + cand[k]
        back[i] = prev[ok][k]
    i = n - hi + int(np.argmax(score[n - hi:])) if n > hi else int(np.argmax(score))
    beats = [i]
    while back[beats[-1]] >= 0:
        beats.append(back[beats[-1]])
    return np.array(beats[::-1])


def beat_grid(x: np.ndarray, sr: int = SR) -> BeatGrid | None:
    """The music's beat grid, or None when there is too little sound to find one."""
    if x is None or len(x) < MIN_SECONDS * sr:
        return None
    env, times = onset_envelope(x, sr)
    if env.std() < 1e-6:
        return None
    env = env / env.std()
    frames_per_s = sr / HOP
    lag = _tempo_lag(env, frames_per_s)
    tracked = times[_track(env, lag)]
    # Refine the period on the half-beat grid, where beat/off-beat flips don't matter.
    best = None
    for period in np.linspace(0.99, 1.01, 401) * lag / frames_per_s:
        hb = period / 2
        v = np.mean(np.exp(2j * np.pi * tracked / hb))
        if best is None or abs(v) > best[0]:
            best = (abs(v), period, (np.angle(v) / (2 * np.pi) * hb) % hb)
    _, period, phase = best
    hb = period / 2
    near = np.abs(((tracked - phase + hb / 2) % hb) - hb / 2) * 1000.0 <= STEADY_MS
    share = float(near.mean())
    # Beat or off-beat: the louder of the two phases is the beat.
    def strength(start):
        grid = np.arange(start, times[-1], period)
        return float(np.mean(env[np.clip(np.searchsorted(times, grid), 0, len(env) - 1)]))
    beat_phase = phase if strength(phase) >= strength(phase + hb) else (phase + hb) % period
    return BeatGrid(tempo_bpm=60.0 / period, half_beat_s=hb, beat_phase_s=beat_phase,
                    steady=share >= STEADY_SHARE, beat_share=share)
