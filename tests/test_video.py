import cv2
import numpy as np

from syncscore.pose.video import probe, resample_uniform


def test_resample_puts_uneven_frames_on_an_even_clock():
    # A point moving at exactly 1 px per ms, recorded with uneven frame spacing.
    times = np.array([0.0, 30.0, 70.0, 100.0, 133.0, 170.0, 200.0])
    xy = np.zeros((1, len(times), 17, 2))
    xy[0, :, :, 0] = times[:, None]
    conf = np.ones((1, len(times), 17))
    out, out_conf, grid = resample_uniform(xy, conf, times, fps=30.0)
    assert np.allclose(np.diff(grid), 1000 / 30)
    assert np.allclose(out[0, :, 0, 0], grid)  # positions land where the true motion was


def test_probe_reads_a_constant_rate_video(tmp_path):
    path = tmp_path / "blank.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 48))
    for _ in range(50):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    info = probe(path)
    assert info.n_frames == 50
    assert abs(info.fps - 25.0) < 0.1
    assert info.size == (64, 48)
    assert not info.variable
