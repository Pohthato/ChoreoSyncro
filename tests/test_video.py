import cv2
import numpy as np
import pytest

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


def test_read_frame_finds_a_frame_by_its_time(tmp_path):
    from syncscore.pose.video import read_frame

    video = tmp_path / "grey.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for k in range(25):
        writer.write(np.full((48, 64, 3), 10 + 8 * k, np.uint8))
    writer.release()
    times = probe(video).times_ms
    for frame, grey in ((0, 10), (17, 146), (24, 202)):
        assert abs(float(read_frame(video, frame, times).mean()) - grey) <= 4, frame
    assert read_frame(video, 25, times) is None


def test_read_frame_finds_the_frame_on_variable_rate_video(tmp_path):  # Review Focus 1
    # Phone video: mostly 33 ms apart, a burst 12 ms apart, one 100 ms drop. Frame k is grey 10 + 5k,
    # and the frame read in decode order is the truth, as extract() numbers frames that way.
    av = pytest.importorskip("av")
    from fractions import Fraction

    from syncscore.pose.video import read_frame, read_frames

    video = tmp_path / "phone.mp4"
    pts_ms = np.concatenate([[0], np.cumsum([33] * 20 + [12, 12, 12] + [33] * 10 + [100] + [33] * 10)])
    container = av.open(str(video), "w")
    stream = container.add_stream("libx264", rate=30)
    stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
    stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
    stream.options = {"bf": "0", "g": "10"}
    for k, t in enumerate(pts_ms):
        frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), 10 + 5 * k, np.uint8), format="rgb24")
        frame.pts, frame.time_base = int(t), Fraction(1, 1000)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    info = probe(video)
    assert info.variable and info.n_frames == len(pts_ms)
    cap = cv2.VideoCapture(str(video))
    truth = {f: float(image.mean()) for f, image in read_frames(cap, np.arange(info.n_frames))}
    cap.release()
    served = {f: float(read_frame(video, f, info.times_ms).mean()) for f in range(info.n_frames)}
    assert [f for f in served if abs(served[f] - truth[f]) > 1.5] == []
