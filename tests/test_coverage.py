"""How much of the clip has every member tracked, and why members go missing."""
import numpy as np
import pytest

from syncscore.pose.coverage import coverage
from syncscore.pose.tracks import assemble, assign_members, build_tracklets
from test_tracks import _detections


def _row(det, track_id, frame):
    return int(np.flatnonzero((det.track_id == track_id) & (det.frame == frame))[0])


def _tracks(det, members):
    return assemble(det, assign_members(det, build_tracklets(det, min_frames=9), members=members, max_gap_s=2.0))


def test_assemble_records_rows():
    det = _detections([(1, range(0, 20), lambda f: 300, 0), (2, range(5, 25), lambda f: 900, 1)], n_frames=30)
    tracks = _tracks(det, 2)
    assert tracks.rows[0, 3] == _row(det, 1, 3)
    assert tracks.rows[1, 10] == _row(det, 2, 10)
    assert tracks.rows[0, 25] == -1 and tracks.rows[1, 2] == -1


def _three_members(extra=()):
    people = [
        (1, range(100), lambda f: 300, 0),
        (2, [f for f in range(100) if not 40 <= f < 50], lambda f: 900, 1),
        (3, range(100), lambda f: 1500, 2),
    ]
    return _detections(people + list(extra), n_frames=100)


def test_coverage_counts_frames_with_everyone():
    det = _three_members()
    result = coverage(det, _tracks(det, 3), min_keypoint_conf=0.3)
    assert result["all_members_share"] == pytest.approx(0.9)
    assert result["gaps"] == {"<0.5s": 1, "0.5-2s": 0, ">=2s": 0}
    assert result["members_seen_share"] == pytest.approx([1.0, 0.9, 1.0])


def test_coverage_splits_the_cause():
    # In frames 40-44 the detector did see member 2 (a 5-frame piece, too short to be a
    # tracklet), so they went missing in assignment; in 45-49 the detector saw nobody there.
    det = _three_members(extra=[(7, range(40, 45), lambda f: 905, 1)])
    result = coverage(det, _tracks(det, 3), min_keypoint_conf=0.3)
    assert result["missing_unassigned_share"] == pytest.approx(0.5)
    assert result["missing_detector_share"] == pytest.approx(0.5)
