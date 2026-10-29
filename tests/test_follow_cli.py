"""`analyze --pose-only` and `analyze --members` on a small synthetic video: six people, each a
filled rectangle in their own colour on a textured still background, so a fingerprint can be the
crop's mean colour."""
import json

import cv2
import numpy as np
import pytest

from syncscore.cli import main
from syncscore.pose.fingerprint import Fingerprinter
from syncscore.pose.tracks import Detections, detections_key
from syncscore.synth.generate import GroupSpec, make_group

SIZE = (640, 360)
COLOURS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]  # BGR
MEMBERS = [0, 2, 3, 5]   # people (left to right) who are members; 1 and 4 are dancers
PICKED = 90


class _MeanColour:
    dim, batch = 3, 64

    def __call__(self, crops):
        v = np.array([c.reshape(-1, 3).mean(axis=0) for c in crops], np.float32).reshape(-1, 3)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def _people():
    group = make_group(GroupSpec(n_members=6, duration_s=6, fps=30, seed=1))
    xy = group.keypoints / 3  # 1920x1080 -> 640x360
    lo, hi = xy.min(axis=2), xy.max(axis=2)
    pad = 0.05 * (hi[..., 1] - lo[..., 1])[..., None]
    return xy, group.conf, np.concatenate([lo - pad, hi + pad], axis=-1)  # box (6, T, 4)


@pytest.fixture
def scene_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(Fingerprinter, "load", classmethod(lambda cls, name: _MeanColour()))
    xy, conf, box = _people()
    n = box.shape[1]
    texture = cv2.GaussianBlur(np.random.default_rng(0).uniform(0, 255, (SIZE[1], SIZE[0], 3)).astype(np.float32), (0, 0), 2)
    background = cv2.normalize(texture, None, 40, 200, cv2.NORM_MINMAX).astype(np.uint8)
    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, SIZE)
    for f in range(n):
        image = background.copy()
        for p in range(6):
            x1, y1, x2, y2 = box[p, f].round().astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), COLOURS[p], -1)
        writer.write(image)
    writer.release()
    out = tmp_path / "out"
    out.mkdir()
    hist = np.zeros((6, 32))
    hist[np.arange(6), np.arange(6)] = 1.0
    frame = np.repeat(np.arange(n), 6)
    person = np.tile(np.arange(6), n)
    Detections(frame=frame, track_id=person + 1, box=box[person, frame], box_conf=np.full(len(frame), 0.9),
               xy=xy[person, frame], kconf=conf[person, frame], hist=hist[person], fps=30.0, n_frames=n,
               image_size=SIZE).save(out / "detections.npz")
    config = tmp_path / "config.yaml"
    config.write_text("tracking:\n  recover: false\n  members: 4\n", encoding="utf-8")
    picks = tmp_path / "picks.json"
    names = ["D", "C", "B", "A"]  # right to left
    picks.write_text(json.dumps({"frame": PICKED, "members": [
        {"name": name, "box": box[p, PICKED].tolist()} for name, p in zip(names, MEMBERS[::-1])]}), encoding="utf-8")
    return {"video": video, "out": out, "config": config, "picks": picks, "box": box}


def _analyze(scene, *flags):
    return main(["--config", str(scene["config"]), "analyze", str(scene["video"]), "--out", str(scene["out"]), *flags])


def test_pose_only_prepares_for_picking_and_stops(scene_dir, capsys):
    assert _analyze(scene_dir, "--pose-only") == 0
    out = scene_dir["out"]
    assert (out / "camera.npz").exists() and (out / "fingerprints.npz").exists()
    assert not (out / "keypoints_raw.npz").exists()
    frames = json.loads((out / "pick_frames.json").read_text(encoding="utf-8"))
    assert frames["n_frames"] == 180 and frames["size"] == list(SIZE) and len(frames["times_ms"]) == 180
    assert frames["suggested"] and all(0 <= f < 180 for f in frames["suggested"])
    assert "ready to name the members (suggested frames: " in capsys.readouterr().out


def test_members_are_followed_by_name_in_pick_order(scene_dir):
    assert _analyze(scene_dir, "--members", str(scene_dir["picks"])) == 0
    out, truth = scene_dir["out"], scene_dir["box"]
    raw = np.load(out / "keypoints_raw.npz")
    assert raw["names"].tolist() == ["D", "C", "B", "A"]
    centre = lambda b: (b[..., :2] + b[..., 2:]) / 2
    for slot, person in enumerate(MEMBERS[::-1]):
        seen = np.isfinite(raw["box"][slot, :, 0])
        assert seen.mean() > 0.9
        assert np.abs(centre(raw["box"][slot, seen]) - centre(truth[person, seen])).max() < 5
        for dancer in (1, 4):
            assert np.linalg.norm(centre(raw["box"][slot, seen]) - centre(truth[dancer, seen]), axis=1).min() > 5
    report = json.loads((out / "tracking_report.json").read_text(encoding="utf-8"))
    assert "suspected_swaps" not in report and "note" not in report
    identity = report["identity"]
    assert identity["picked_frame"] == PICKED and identity["model"]
    assert set(identity["tracked_share"]) == {"A", "B", "C", "D"}
    assert {"splits", "close_calls", "position_rejections"} <= set(identity)


def test_a_bad_picks_file_stops_with_the_reason(scene_dir, capsys):
    scene_dir["picks"].write_text(json.dumps({"frame": PICKED, "members": [{"name": "A", "box": [0, 0, 10, 10]}]}),
                                  encoding="utf-8")
    assert _analyze(scene_dir, "--members", str(scene_dir["picks"])) == 2
    assert "Name at least 2 members." in capsys.readouterr().err


def test_prepare_recomputes_when_the_detections_change(scene_dir):  # Review Focus 2
    out = scene_dir["out"]
    assert _analyze(scene_dir, "--pose-only") == 0
    files = [out / "camera.npz", out / "fingerprints.npz"]
    stamps = [f.stat().st_mtime_ns for f in files]
    assert _analyze(scene_dir, "--pose-only") == 0
    assert [f.stat().st_mtime_ns for f in files] == stamps  # same detections: reused
    det = Detections.load(out / "detections.npz")
    det.box = det.box + 1.0
    det.save(out / "detections.npz")
    assert _analyze(scene_dir, "--pose-only") == 0
    for f in files:
        with np.load(f) as data:
            assert str(data["key"]) == detections_key(det)


def test_without_picks_analyze_is_unchanged(scene_dir):
    assert _analyze(scene_dir) == 0
    out = scene_dir["out"]
    assert np.load(out / "keypoints_raw.npz")["names"].tolist() == ["M1", "M2", "M3", "M4"]
    report = json.loads((out / "tracking_report.json").read_text(encoding="utf-8"))
    assert "suspected_swaps" in report and "identity" not in report
    assert not (out / "camera.npz").exists()


def test_the_two_flags_cannot_be_used_together(scene_dir):
    with pytest.raises(SystemExit):
        _analyze(scene_dir, "--pose-only", "--members", str(scene_dir["picks"]))
