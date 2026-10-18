"""Filling a member's short gap: only with a real detection, at the right place and size, that
isn't another member or a copy of the front dancer's skeleton."""
import numpy as np

from syncscore.pose.fill import accept, expected_box

H = 300.0  # box height in pixels


def _box(x_center, height=H, top=100.0):
    return np.array([x_center - height / 4, top, x_center + height / 4, top + height])


def _skeleton(box):
    """17 keypoints spread down the box, all confident."""
    x = (box[0] + box[2]) / 2 + np.linspace(-20, 20, 17)
    y = np.linspace(box[1], box[3], 17)
    return np.stack([x, y], axis=1), np.full(17, 0.9)


def _member_boxes(seen, frames=40):
    """One member's (1, frames, 4) boxes: {frame: x_center}, NaN elsewhere."""
    box = np.full((1, frames, 4), np.nan)
    for f, x in seen.items():
        box[0, f] = _box(x)
    return box


def _no_others():
    return np.zeros((0, 4)), np.zeros((0, 17, 2)), np.zeros((0, 17))


def _accept(candidate, expected, others=None):
    xy, conf = _skeleton(candidate)
    o_box, o_xy, o_conf = others if others is not None else _no_others()
    return accept(candidate, xy, conf, expected, o_box, o_xy, o_conf, min_keypoint_conf=0.3)


def test_expected_box_interpolates_inside_a_short_gap():
    box = _member_boxes({9: 100.0, 20: 210.0})
    got = expected_box(box, member=0, frame=15, max_gap=60)
    assert (got[0] + got[2]) / 2 == np.float64(160.0)


def test_edge_gaps_have_no_expected_box():
    box = _member_boxes({10: 100.0, 11: 100.0})
    assert expected_box(box, member=0, frame=5, max_gap=60) is None
    assert expected_box(box, member=0, frame=20, max_gap=60) is None


def test_long_gap_has_no_expected_box():
    box = _member_boxes({0: 100.0, 62: 100.0}, frames=70)  # 61 unseen frames
    assert expected_box(box, member=0, frame=30, max_gap=60) is None
    box = _member_boxes({0: 100.0, 61: 100.0}, frames=70)  # 60 unseen frames
    assert expected_box(box, member=0, frame=30, max_gap=60) is not None


def test_candidate_near_the_expected_spot_is_accepted():
    assert np.isfinite(_accept(_box(510), _box(500)))


def test_far_candidate_is_rejected():
    assert _accept(_box(500 + 0.6 * H), _box(500)) == np.inf


def test_wrong_size_is_rejected():
    assert _accept(_box(500, height=1.5 * H), _box(500)) == np.inf


def test_candidate_that_is_another_member_is_rejected():
    other = _box(500)
    o_xy, o_conf = _skeleton(other)
    candidate = _box(510)  # IoU with the other member's box is about 0.88
    assert _accept(candidate, _box(520), (other[None], o_xy[None], o_conf[None])) == np.inf


def test_duplicate_of_the_occluder_is_rejected():
    # The hidden member's box overlaps the front dancer's (IoU about 0.6, allowed), but
    # the keypoints inside it are the front dancer's own, within 2 px.
    front = _box(500)
    f_xy, f_conf = _skeleton(front)
    candidate = _box(540)
    c_xy = f_xy + 1.5
    cost = accept(candidate, c_xy, f_conf, _box(540), front[None], f_xy[None], f_conf[None], min_keypoint_conf=0.3)
    assert cost == np.inf
    # The same box with its own skeleton is fine.
    assert np.isfinite(_accept(candidate, _box(540), (front[None], f_xy[None], f_conf[None])))


# --- filling from detections the member assignment left unused ---

from syncscore.pose.fill import fill_gaps  # noqa: E402
from syncscore.pose.tracks import assemble, assign_members, build_tracklets  # noqa: E402
from test_tracks import _detections  # noqa: E402


def _tracks(det, members):
    return assemble(det, assign_members(det, build_tracklets(det, min_frames=9), members=members, max_gap_s=2.0))


def _seen(tracks):
    return tracks.rows >= 0


def test_leftover_detection_fills_a_short_gap():
    det = _detections([
        (1, [f for f in range(100) if not 40 <= f < 46], lambda f: 300, 0),
        (7, range(40, 46), lambda f: 305, 0),  # 6 frames: too short to be a tracklet
        (2, range(100), lambda f: 900, 1),
    ])
    tracks = _tracks(det, 2)
    assert not _seen(tracks)[0, 40:46].any()
    filled, count = fill_gaps(det, tracks, max_gap_s=2.0, min_keypoint_conf=0.3)
    assert count == 6
    assert _seen(filled)[0, 40:46].all()
    assert all(det.track_id[filled.rows[0, f]] == 7 for f in range(40, 46))
    assert np.isfinite(filled.xy[0, 42]).all()


def test_one_detection_fills_one_member():
    det = _detections([
        (1, [f for f in range(100) if f != 50], lambda f: 300, 0),
        (2, [f for f in range(100) if f != 50], lambda f: 420, 1),
        (7, [50], lambda f: 310, 0),
    ])
    filled, count = fill_gaps(det, _tracks(det, 2), max_gap_s=2.0, min_keypoint_conf=0.3)
    assert count == 1
    assert _seen(filled)[0, 50] and not _seen(filled)[1, 50]


def test_edge_gaps_are_not_filled():
    det = _detections([(1, range(10, 100), lambda f: 300, 0), (7, range(0, 6), lambda f: 300, 0)])
    filled, count = fill_gaps(det, _tracks(det, 1), max_gap_s=2.0, min_keypoint_conf=0.3)
    assert count == 0 and not _seen(filled)[0, :10].any()


def test_fill_off_changes_nothing():
    det = _detections([
        (1, [f for f in range(100) if not 40 <= f < 46], lambda f: 300, 0),
        (7, range(40, 46), lambda f: 305, 0),
    ])
    tracks = _tracks(det, 1)
    filled, count = fill_gaps(det, tracks, max_gap_s=0.0, min_keypoint_conf=0.3)
    assert count == 0
    assert np.array_equal(filled.rows, tracks.rows) and np.array_equal(filled.xy, tracks.xy, equal_nan=True)
