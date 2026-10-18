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
