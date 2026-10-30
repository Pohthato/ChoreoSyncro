import numpy as np
import pytest

from scenes import unit
from syncscore.config import load_config
from syncscore.pose.camera import still_camera
from syncscore.pose.gate import gate
from test_motion import _crossing
from test_tracks import _detections


def _people():
    """Two members (tracks 1, 2) and a dancer (track 3) over 60 frames, plus member A once with no
    tracker ID at frame 30 (a second sighting the tracker did not link)."""
    det = _detections([(1, range(60), lambda f: 300, 0), (2, range(60), lambda f: 900, 1),
                       (3, range(60), lambda f: 1500, 2), (-1, [30], lambda f: 600, 0)], n_frames=60)
    rng = np.random.default_rng(0)
    looks = {1: unit(rng), 2: unit(rng), 3: unit(rng), -1: None}
    looks[-1] = looks[1]
    vectors = np.stack([looks[int(t)] + 0.05 * rng.normal(size=32) for t in det.track_id])
    det.vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    labels = {}
    for row, (f, t) in enumerate(zip(det.frame, det.track_id)):
        if f in (0, 30, 59):
            labels[row] = {1: "A", 2: "B", 3: "dancer", -1: "A"}[int(t)]
    return det, labels


def test_gate_counts_right_names_unassigned_members_and_named_dancers():
    det, labels = _people()
    result = gate(det, det.vectors, labels, picked_frame=0, config=load_config(), camera=still_camera(det.n_frames))
    # member rows outside the picked frame: A at 30 and 59, B at 30 and 59, the untracked A at 30
    assert result["rows"] == {"member": 5, "dancer": 2}
    assert result["member_correct"] == pytest.approx(4 / 5)
    assert result["member_unassigned"] == pytest.approx(1 / 5)  # the row with no tracker ID has no piece
    assert result["dancer_as_member"] == 0.0
    assert result["per_member"] == {"A": pytest.approx(2 / 3), "B": 1.0}


def test_tracks_that_change_hands_at_a_crossing_are_split_before_naming_as_analyze_does():
    """A and a backup dancer swap tracker IDs where they cross at frame 150. Named whole, the
    seed's track would carry A's name onto the dancer and the dancer's track would hide A."""
    rng = np.random.default_rng(1)
    det, truth = _crossing(unit(rng), unit(rng), handover=True)
    labels = {int(row): ("A", "dancer")[truth[row]] for row in np.flatnonzero(np.isin(det.frame, (0, 100, 200, 280)))}
    result = gate(det, det.vectors, labels, picked_frame=0, config=load_config(), camera=still_camera(det.n_frames))
    assert result["rows"] == {"member": 3, "dancer": 3}
    assert result["member_correct"] == 1.0
    assert result["dancer_as_member"] == 0.0


def test_id_gate_reuses_the_saved_camera_path_and_fingerprints_and_writes_its_report(tmp_path):
    import csv
    import json

    from syncscore.cli import main
    from syncscore.pose.fingerprint import save_fingerprints
    from syncscore.pose.tracks import detections_key

    det, labels = _people()
    det.save(tmp_path / "detections.npz")
    key = detections_key(det)
    still_camera(det.n_frames).save(tmp_path / "camera.npz", key)
    save_fingerprints(tmp_path / "gate_fingerprints_youtu.npz", det.vectors, "youtu", key)
    with open(tmp_path / "ids.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "true_name"])
        writer.writerows(labels.items())
    code = main(["id-gate", str(tmp_path / "no-video.mp4"), "--results", str(tmp_path), "--labels", str(tmp_path / "ids.csv"),
                 "--frame", "0", "--model", "youtu"])
    assert code == 0
    report = json.loads((tmp_path / "gate.json").read_text(encoding="utf-8"))
    assert report["models"]["youtu"]["margins"]["0.06"]["member_correct"] == pytest.approx(4 / 5)
