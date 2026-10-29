import numpy as np
import pytest

from scenes import Person, near, scene, unit
from syncscore.config import load_config
from syncscore.pose.camera import still_camera
from syncscore.pose.identity import allow_for, follow_members
from syncscore.pose.tracks import assemble, build_tracklets

CONFIG = load_config()
PICKED = 30


def _follow(people, members):
    """Run follow_members with `members` (indices into people) picked at frame 30."""
    det, truth = scene(people)
    picked = [int(np.flatnonzero((det.frame == PICKED) & (truth == m))[0]) for m in members]
    following = follow_members(det, build_tracklets(det, 9), still_camera(det.n_frames), picked, CONFIG)
    return det, truth, following


def _rows(following, member):
    return np.concatenate([t.rows for t in following.slots[member].tracklets])


def _ids(det, following, member):
    return set(det.track_id[_rows(following, member)].tolist())


def test_lookalikes_crossing_keep_their_own_names():
    rng = np.random.default_rng(0)
    a = unit(rng)
    a_track = lambda f: 1 if f <= 150 else 2   # the tracker swaps their IDs as they cross
    b_track = lambda f: 2 if f <= 150 else 1
    people = [Person(a, lambda f: 200 + 4 * f, a_track), Person(near(a, 0.85, rng), lambda f: 1400 - 4 * f, b_track)]
    det, truth, following = _follow(people, [0, 1])
    for m in (0, 1):
        rows = _rows(following, m)
        assert set(truth[rows].tolist()) == {m}
        assert len(set(det.frame[rows].tolist())) >= 0.9 * det.n_frames


def test_someone_walking_in_front_of_a_member_does_not_take_her_name():
    rng = np.random.default_rng(1)
    # D passes in front of A around frame 150. A is hidden 145-155; the tracker moves A's ID to D
    # from 145, and A comes back as ID 3 at 156.
    a_track = lambda f: 1 if f < 145 else (None if f <= 155 else 3)
    d_track = lambda f: 2 if f < 145 else 1
    people = [Person(unit(rng), lambda f: 600, a_track), Person(unit(rng), lambda f: 100 + 10 * f / 3, d_track),
              Person(unit(rng), lambda f: 1600, lambda f: 5)]
    det, truth, following = _follow(people, [0, 2])
    rows = _rows(following, 0)
    assert 1 not in truth[rows]
    assert 3 in _ids(det, following, 0)


def test_a_member_who_leaves_the_frame_is_found_again_by_fingerprint():
    rng = np.random.default_rng(2)
    a_x = lambda f: 300 if f < 100 else 1200
    a_track = lambda f: 1 if f < 100 else (None if f < 250 else 2)
    people = [Person(unit(rng), a_x, a_track), Person(unit(rng), lambda f: 800, lambda f: 5)]
    det, truth, following = _follow(people, [0, 1])
    assert 2 in _ids(det, following, 0)


def test_a_lookalike_far_from_where_a_hidden_member_is_heading_is_refused():
    rng = np.random.default_rng(3)
    a = unit(rng)
    a_track = lambda f: 1 if f < 150 else (None if f < 180 else 3)
    people = [Person(a, lambda f: 200 + 4 * f, a_track),
              Person(a, lambda f: 100, lambda f: 5 if f >= 160 else None),  # dressed exactly like her
              Person(unit(rng), lambda f: 1700, lambda f: 2)]
    det, truth, following = _follow(people, [0, 2])
    ids = _ids(det, following, 0)
    assert 3 in ids and 5 not in ids
    rejections = following.report["position_rejections"]
    assert len(rejections) == 1
    assert rejections[0]["member"] == 0 and rejections[0]["time_s"] == pytest.approx(160 / 30, abs=0.01)


def test_two_pieces_with_one_name_at_once_keep_the_stronger():
    rng = np.random.default_rng(4)
    a = unit(rng)
    people = [Person(a, lambda f: 600, lambda f: 1),
              Person(near(a, 0.93, rng), lambda f: 1200, lambda f: 7 if 100 <= f <= 200 else None),
              Person(unit(rng), lambda f: 200, lambda f: 2)]
    det, truth, following = _follow(people, [0, 2])
    assert _ids(det, following, 0) == {1}


def test_a_picked_person_with_no_tracker_id_still_seeds_her_name():  # Review Focus 3
    rng = np.random.default_rng(5)
    people = [Person(unit(rng), lambda f: 600, lambda f: -1 if f == PICKED else 4),
              Person(unit(rng), lambda f: 1200, lambda f: 2)]
    det, truth, following = _follow(people, [0, 1])
    assert 4 in _ids(det, following, 0)
    assert set(truth[_rows(following, 0)].tolist()) == {0}


def test_the_report_lists_splits_in_seconds():
    rng = np.random.default_rng(6)
    a_track = lambda f: 1 if f <= 150 else 2
    b_track = lambda f: 2 if f <= 150 else 1
    people = [Person(unit(rng), lambda f: 200 + 4 * f, a_track), Person(unit(rng), lambda f: 1400 - 4 * f, b_track)]
    det, truth, following = _follow(people, [0, 1])
    report = following.report
    assert report["picked_frame"] == PICKED
    assert {s["track_id"] for s in report["splits"]} == {1, 2}
    assert all(abs(s["time_s"] - 5.0) < 0.2 and s["reason"] == "fingerprint" for s in report["splits"])


def test_gap_filling_only_allows_rows_that_look_like_the_member():
    rng = np.random.default_rng(7)
    people = [Person(unit(rng), lambda f: 600, lambda f: 1), Person(unit(rng), lambda f: 1200, lambda f: 2)]
    det, truth, following = _follow(people, [0, 1])
    allow = allow_for(following.recognizer)
    a_row, b_row = int(np.flatnonzero(truth == 0)[100]), int(np.flatnonzero(truth == 1)[100])
    assert allow(det, 0, a_row) and not allow(det, 0, b_row) and allow(det, 1, b_row)


def test_slots_come_in_pick_order():
    rng = np.random.default_rng(8)
    people = [Person(unit(rng), lambda f: 1500, lambda f: 1), Person(unit(rng), lambda f: 300, lambda f: 2)]
    det, truth, following = _follow(people, [0, 1])
    tracks = assemble(det, following.slots, keep_order=True)
    assert tracks.box[0, 100, 0] > tracks.box[1, 100, 0]  # the first pick stands on the right
