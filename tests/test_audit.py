import csv

import cv2
import numpy as np
import pytest

from syncscore.cli import main
from syncscore.pose.tracks import Detections, audit_frames, parse_window


def test_audit_frames_add_every_step_inside_each_window():
    frames = audit_frames(np.ones(3000, bool), fps=30.0, count=3, windows=[(10.0, 10.4)], step_s=0.2)
    assert list(frames) == [0, 300, 306, 312, 1499, 2999]


def test_a_window_must_end_after_it_starts():
    assert parse_window("148-150.5") == (148.0, 150.5)
    with pytest.raises(ValueError):
        parse_window("150-148")


def test_audit_draws_every_detection_and_writes_rows(tmp_path):
    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
    for _ in range(30):
        writer.write(np.full((240, 320, 3), 90, np.uint8))
    writer.release()
    member, other = [20.0, 40, 100, 220], [200.0, 40, 280, 220]
    box = np.full((1, 30, 4), np.nan, np.float32)
    box[0, :20] = member  # the member is tracked in frames 0-19 only
    np.savez_compressed(tmp_path / "keypoints_raw.npz", xy=np.zeros((1, 30, 17, 2), np.float32),
                        conf=np.zeros((1, 30, 17), np.float32), box=box, fps=30.0, names=np.array(["Ann"]))
    frames = np.repeat(np.arange(30), 2)
    det = Detections(frame=frames, track_id=np.tile([1, 2], 30), box=np.tile([member, other], (30, 1)),
                     box_conf=np.ones(60), xy=np.zeros((60, 17, 2)), kconf=np.zeros((60, 17)), hist=np.zeros((60, 4)),
                     fps=30.0, n_frames=30, image_size=(320, 240))
    det.save(tmp_path / "detections.npz")

    assert main(["audit", str(video), "--results", str(tmp_path), "--frames", "3", "--times", "0.2-0.4"]) == 0

    out = tmp_path / "audit"
    with open(out / "ids.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == ["frame", "row", "shown_label", "true_name"]
    audited = sorted({int(r["frame"]) for r in rows})
    assert audited == [0, 6, 9, 12, 19]  # 3 evenly over frames 0-19, plus 0.2 s, 0.3 s, 0.4 s
    others = [r for r in rows if r["shown_label"] == "?1"]
    assert len(others) == len(audited) and all(int(r["row"]) == 2 * int(r["frame"]) + 1 for r in others)
    members = [r for r in rows if r["shown_label"] == "Ann"]
    assert [int(r["row"]) for r in members] == [2 * f for f in audited]
    assert sorted(p.name for p in out.glob("frame_*.jpg")) == [f"frame_{f:06d}.jpg" for f in audited]
