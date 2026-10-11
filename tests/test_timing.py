"""Unit tests for the alignment math, each on a tiny hand-built case."""
import warnings

import numpy as np
import pytest

from syncscore.sync.consensus import leave_one_out_median, warp
from syncscore.sync.timing import estimate_lags, make_windows, parabolic_peak, velocity, viterbi_path


def _unit(angle):
    return np.stack([np.cos(angle), np.sin(angle)], axis=-1)


def test_parabolic_peak_is_exact_on_a_parabola():
    for true in (-0.4, -0.1, 0.0, 0.25, 0.49):
        y = [-(x - true) ** 2 for x in (-1, 0, 1)]
        assert parabolic_peak(*y) == pytest.approx(true)


def test_parabolic_peak_ignores_a_minimum():
    assert parabolic_peak(1.0, 0.0, 1.0) == 0.0


def test_viterbi_keeps_a_smooth_path_through_an_alias():
    # Window 2 has a slightly higher false peak far away (periodic choreography); the smooth
    # path through its neighbours should win.
    scores = np.full((5, 9), 0.1)
    scores[:, 4] = 0.9
    scores[2, 0] = 0.95
    assert list(viterbi_path(scores, penalty_per_step=0.05)) == [4, 4, 4, 4, 4]
    assert viterbi_path(scores, penalty_per_step=0.0)[2] == 0  # without the penalty it jumps


def test_windows_cover_the_clip():
    windows = make_windows(n_frames=300, window=60, hop=15)
    assert windows.starts[0] == 0
    assert windows.starts[-1] + 60 <= 300
    assert np.allclose(windows.centers, windows.starts + 29.5)


def test_warp_integer_and_fractional():
    angles = np.linspace(0, 1, 10)[:, None]  # (T, B=1)
    u = _unit(angles)
    w = np.ones((10, 1))
    shifted, sw = warp(u, w, np.full(10, 2.0))
    assert np.allclose(shifted[:8], u[2:])
    assert (sw[8:] == 0).all() and np.isnan(shifted[8:]).all()
    half, _ = warp(u, w, np.full(10, 0.5))
    expected = _unit((angles[:-1] + angles[1:]) / 2)
    assert np.allclose(half[:9], expected, atol=1e-3)  # normalised lerp is close to the true midpoint


def test_leave_one_out_ignores_the_member_itself_and_one_outlier():
    m, t, b = 5, 4, 1
    u = np.tile(_unit(np.zeros((t, b))), (m, 1, 1, 1))
    u[0] = _unit(np.full((t, b), 1.5))  # member 0 is far off
    w = np.ones((m, t, b))
    c, cw = leave_one_out_median(u, w)
    assert np.allclose(c[0], _unit(np.zeros((t, b))))   # the others agree
    assert np.allclose(c[1], _unit(np.zeros((t, b))), atol=1e-9)  # one outlier among the others does not move the median


def test_velocity_of_a_steady_rotation():
    fps = 30.0
    times = np.arange(30) / fps
    u = _unit(2.0 * times[:, None])  # 2 rad/s
    v, vw = velocity(u, np.ones((30, 1)), fps)
    speed = np.linalg.norm(v[1:-1], axis=-1)
    assert np.allclose(speed, 2.0, rtol=1e-3)
    assert np.isnan(v[0]).all() and vw[0, 0] == 0


def test_estimate_lags_recovers_a_shift_of_a_random_smooth_signal():
    rng = np.random.default_rng(0)
    fps, frames, true_lag = 30.0, 600, 3.4
    knots = rng.normal(size=(40, 4))
    times = np.arange(frames + 40)
    signal = np.stack([np.interp(times, np.linspace(0, frames + 40, 40), knots[:, k]) for k in range(4)], axis=-1)
    angles = np.cumsum(signal, axis=0) * 0.02
    ref_u = _unit(angles[20:20 + frames])
    member_angles = np.stack([np.interp(np.arange(frames) - true_lag, np.arange(frames + 40) - 20, angles[:, k]) for k in range(4)], axis=-1)
    member_u = _unit(member_angles)
    ones = np.ones((frames, 4))
    rv, rw = velocity(ref_u, ones, fps)
    mv, mw = velocity(member_u, ones, fps)
    result = estimate_lags(mv, mw, rv, rw, make_windows(frames, 60, 15), max_lag=12, penalty_per_step=0.01, min_motion=0.0, min_peak=0.0, min_coverage=0.0)
    inner = result.lag[3:-3]
    assert np.allclose(inner, true_lag, atol=0.15)


def test_fast_nanmedian_matches_numpy():
    from syncscore.sync.consensus import nanmedian_first_axis

    rng = np.random.default_rng(3)
    for members in (1, 2, 3, 4, 6):
        x = rng.normal(size=(members, 50, 3))
        x[rng.random(x.shape) < 0.3] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            expected = np.nanmedian(x, axis=0)
        assert np.allclose(nanmedian_first_axis(x), expected, equal_nan=True)


def test_a_second_equal_peak_makes_the_window_ambiguous():
    from syncscore.sync.timing import _best_other_peak

    single = np.array([0.0, 0.2, 0.6, 1.0, 0.6, 0.2, 0.0, -0.1, 0.0])
    double = np.array([0.0, 0.9, 0.5, 1.0, 0.6, 0.2, 0.0, -0.1, 0.0])
    assert single[3] - _best_other_peak(single, 3) > 0.9
    assert double[3] - _best_other_peak(double, 3) < 0.15
