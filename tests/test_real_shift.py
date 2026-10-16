import numpy as np

from syncscore.synth.real_shift import delay_member


def test_delay_member_shifts_only_that_member_by_a_fraction_of_a_frame():
    frames = 40
    xy = np.zeros((2, frames, 17, 2))
    xy[:, :, :, 0] = np.arange(frames)[None, :, None]  # x = frame index
    conf = np.ones((2, frames, 17))
    out, out_conf = delay_member(xy, conf, member=1, shift_ms=50.0, fps=30.0)  # 1.5 frames late
    assert np.allclose(out[0], xy[0])
    assert np.allclose(out[1, 10:, 0, 0], np.arange(10, frames) - 1.5)
    assert np.isnan(out[1, 0]).all() and out_conf[1, 0, 0] == 0
