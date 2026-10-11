import numpy as np

from syncscore.config import SmoothingConfig
from syncscore.pose.smooth import fill_short_gaps, one_euro, smooth_keypoints


def _best_lag_ms(reference, signal, fps):
    """Shift (ms) that best aligns `signal` to `reference`, on a 0.01-frame grid."""
    frames = np.arange(len(reference))
    shifts = np.arange(-5, 5, 0.01)
    errors = [np.nanmean((np.interp(frames + s, frames, signal) - reference)[20:-20] ** 2) for s in shifts]
    return shifts[int(np.argmin(errors))] / fps * 1000


def test_one_euro_reduces_jitter_on_a_still_point():
    rng = np.random.default_rng(0)
    x = 5.0 + rng.normal(0, 0.05, size=(300, 1, 2))
    cfg = SmoothingConfig()
    out = one_euro(x, fps=30, min_cutoff=cfg.min_cutoff, beta=cfg.beta, d_cutoff=cfg.d_cutoff)
    assert out[30:].std() < x[30:].std() / 2


def test_forward_backward_smoothing_adds_no_lag():
    # Lag is exactly what this tool measures, so the smoother must not add any. A causal
    # (forward-only) filter does, and a speed-dependent one adds different lag to different
    # members. Averaging a forward and a backward pass cancels it.
    fps = 30.0
    t = np.arange(300) / fps
    clean = np.stack([np.sin(2 * np.pi * 1.5 * t), np.cos(2 * np.pi * 1.5 * t)], axis=-1)[:, None, :]
    xy = clean[:, None, 0, :].repeat(17, axis=1)
    conf = np.ones((300, 17))
    cfg = SmoothingConfig()
    smoothed, _ = smooth_keypoints(xy * 100, conf, fps, cfg, min_conf=0.3, scale=100.0)
    forward = one_euro(xy, fps, cfg.min_cutoff, cfg.beta, cfg.d_cutoff) * 100
    zero_phase_lag = _best_lag_ms(xy[:, 9, 0] * 100, smoothed[:, 9, 0], fps)
    causal_lag = _best_lag_ms(xy[:, 9, 0] * 100, forward[:, 9, 0], fps)
    assert abs(zero_phase_lag) < 1.0
    assert causal_lag > 5.0  # the causal output trails the input


def test_fast_motion_survives():
    # A wrist moving one torso length in 150 ms (a hit) keeps most of its peak speed.
    fps = 60.0
    t = np.arange(240) / fps
    tau = np.clip((t - 2.0) / 0.15, 0, 1)
    pos = tau**3 * (10 - 15 * tau + 6 * tau**2)
    xy = np.zeros((240, 17, 2))
    xy[:, :, 0] = pos[:, None] * 100
    smoothed, _ = smooth_keypoints(xy, np.ones((240, 17)), fps, SmoothingConfig(), min_conf=0.3, scale=100.0)
    peak = np.abs(np.diff(xy[:, 9, 0])).max()
    kept = np.abs(np.diff(smoothed[:, 9, 0])).max()
    assert kept > 0.8 * peak


def test_short_gaps_are_filled_and_long_ones_kept():
    x = np.arange(20, dtype=float)[:, None, None].repeat(2, axis=2)
    x[5:8] = np.nan     # 3 frames
    x[10:16] = np.nan   # 6 frames
    filled, was_filled = fill_short_gaps(x, max_gap=4)
    assert np.allclose(filled[5:8, 0, 0], [5, 6, 7])
    assert was_filled[5:8, 0].all()
    assert np.isnan(filled[10:16]).all()
