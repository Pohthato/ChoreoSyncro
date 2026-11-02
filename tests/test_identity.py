import numpy as np
import pytest

from scenes import Person, at, near, scene, unit
from syncscore.config import load_config
from syncscore.pose.camera import still_camera
from syncscore.pose.identity import allow_for, follow_members
from syncscore.pose.tracks import assemble, build_tracklets

CONFIG = load_config()
PICKED = 30


def _follow(people, members, noise=0.05):
    """Run follow_members with `members` (indices into people) picked at frame 30."""
    det, truth = scene(people, noise=noise)
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


@pytest.mark.parametrize("dancer_id, a_id", [
    (3, lambda f: 1 if f < 100 else 3 if f < 195 else 4),  # track 3 wanders onto the dancer as A goes on as track 4
    (4, lambda f: 1 if f < 100 else 3 if f < 205 else 4),  # track 4 starts on the dancer and only then finds A
])
def test_pieces_sharing_a_few_frames_lose_only_those_frames(dancer_id, a_id):
    rng = np.random.default_rng(7)
    a = unit(rng)
    people = [Person(a, lambda f: 600, a_id),
              Person(near(a, 0.9, rng), lambda f: 1200, lambda f: dancer_id if 195 <= f < 205 else None),
              Person(unit(rng), lambda f: 200, lambda f: 2)]
    det, truth, following = _follow(people, [0, 2])
    rows = _rows(following, 0)
    assert set(truth[rows].tolist()) == {0}
    assert set(det.frame[rows].tolist()) == set(range(det.n_frames))


def test_a_piece_that_beat_one_she_has_is_not_checked_against_where_that_one_was_heading():
    rng = np.random.default_rng(8)
    a = unit(rng)
    a_track = lambda f: 1 if f < 100 else 3 if f < 190 else (None if f < 200 else 4)
    # While A is hidden a dancer leaves her spot at speed, carrying her name, and overlaps A's return.
    people = [Person(a, lambda f: 600, a_track),
              Person(near(a, 0.9, rng), lambda f: 600 + 30 * (f - 190), lambda f: 5 if 190 <= f < 210 else None),
              Person(unit(rng), lambda f: 200, lambda f: 2)]
    det, truth, following = _follow(people, [0, 2])
    assert set(range(200, det.n_frames)) <= set(det.frame[_rows(following, 0)].tolist())
    assert following.report["position_rejections"] == []


def test_a_piece_named_by_look_is_not_refused_from_one_it_shares_frames_with():
    rng = np.random.default_rng(14)
    a = unit(rng)
    # The tracker hands her over with two frames where both IDs are on her, the new box a body-width off.
    # One frame apart, that offset is far outside where she was heading, but it says nothing about a lookalike.
    people = [Person(a, lambda f: 600, lambda f: 1 if f <= 150 else None),
              Person(near(a, 0.95, rng), lambda f: 680, lambda f: 2 if f >= 149 else None),
              Person(unit(rng), lambda f: 1500, lambda f: 5)]
    det, truth, following = _follow(people, [0, 2])
    assert 2 in _ids(det, following, 0)
    assert following.report["position_rejections"] == []


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


SHARP = 0.01  # noise low enough that a score is the cosine it was built from


def _piece_she_barely_leads(rng, a, to_dancer, dancer, x=lambda f: 600, frames=(150, 210), track=2, **rivals):
    """A person seen only in `frames`, about (0.5 - to_dancer) closer to A than to the dancer."""
    look = at(rng, (a, 0.5), (dancer, to_dancer), *rivals.values())
    return Person(look, x, lambda f: track if frames[0] <= f <= frames[1] else None)


def _hidden_a(lead, x=lambda f: 600, end=210):
    """A is plain up to frame 150 and after `end`; in between she is a piece leading the dancer's look by `lead`."""
    rng = np.random.default_rng(9)
    a, dancer = unit(rng), unit(rng)
    people = [Person(a, lambda f: 600, lambda f: 1 if f < 150 else (3 if f > end else None)),
              _piece_she_barely_leads(rng, a, 0.5 - lead, dancer, x, (150, end)),
              Person(dancer, lambda f: 1500, lambda f: 5)]
    det, truth, following = _follow(people, [0], noise=SHARP)
    margin = following.recognizer.name(det.vectors[truth == 1]).margin
    assert margin < CONFIG.identity.name_margin
    return det, truth, following, margin


def test_a_piece_she_barely_leads_is_hers_when_it_continues_her_motion():
    det, truth, following, margin = _hidden_a(0.05)
    assert margin >= CONFIG.identity.name_margin / 2
    assert 2 in _ids(det, following, 0)


def test_the_report_lists_the_pieces_taken_under_the_margin():
    det, truth, following, margin = _hidden_a(0.05)
    taken = following.report["taken_under_margin"]
    assert len(taken) == 1
    assert taken[0]["start_s"] == 5.0 and taken[0]["end_s"] == 7.0 and taken[0]["member"] == 0
    assert taken[0]["lead"] == pytest.approx(margin, abs=0.01)


def test_a_piece_she_leads_by_under_half_the_margin_stays_unnamed():
    det, truth, following, margin = _hidden_a(0.01)
    assert 0 < margin < CONFIG.identity.name_margin / 2
    assert 2 not in _ids(det, following, 0)


def test_a_piece_too_long_for_motion_to_vouch_for_stays_unnamed():
    too_long = 150 + int(CONFIG.identity.free_after_s * 30) + 10
    det, truth, following, margin = _hidden_a(0.05, end=too_long)
    assert margin >= CONFIG.identity.name_margin / 2
    assert 2 not in _ids(det, following, 0)


def test_a_piece_she_leads_far_from_where_she_was_heading_stays_unnamed():
    det, truth, following, margin = _hidden_a(0.05, x=lambda f: 1300)
    assert margin >= CONFIG.identity.name_margin / 2
    assert 2 not in _ids(det, following, 0)
    assert following.report["position_rejections"] == []  # it was never named her


def _a_or_b(b_track):
    """A piece close to A, a little less close to B, and further from the dancer. B's tracker ID is `b_track`."""
    rng = np.random.default_rng(10)
    a, b, dancer = unit(rng), unit(rng), unit(rng)
    people = [Person(a, lambda f: 600, lambda f: 1 if f < 150 else (3 if f > 210 else None)),
              Person(b, lambda f: 1000, b_track),
              _piece_she_barely_leads(rng, a, 0.45, dancer, b=(b, 0.48)),
              Person(dancer, lambda f: 1500, lambda f: 5)]
    det, truth, following = _follow(people, [0, 1], noise=SHARP)
    naming = following.recognizer.name(det.vectors[truth == 2])
    assert 0 < naming.margin < CONFIG.identity.name_margin / 2 and np.nanargmax(naming.scores[:-1]) == 0
    return det, following


def test_a_member_seen_elsewhere_during_a_piece_is_no_rival_for_it():
    det, following = _a_or_b(lambda f: 4)
    assert 2 in _ids(det, following, 0)


def test_a_member_out_of_sight_during_a_piece_is_still_a_rival_for_it():
    det, following = _a_or_b(lambda f: None if 140 <= f <= 220 else 4)
    assert 2 not in _ids(det, following, 0)
    assert 2 not in _ids(det, following, 1)


def test_a_piece_can_continue_her_motion_from_another_taken_the_same_way():
    rng = np.random.default_rng(11)
    a, dancer = unit(rng), unit(rng)
    # The second piece has the better lead, so it is tried first, but only the first is in reach of what she has.
    people = [Person(a, lambda f: 600, lambda f: 1 if f < 150 else (3 if f >= 370 else None)),
              _piece_she_barely_leads(rng, a, 0.46, dancer, frames=(150, 170), track=2),
              _piece_she_barely_leads(rng, a, 0.45, dancer, frames=(250, 275), track=4),
              Person(dancer, lambda f: 1500, lambda f: 5)]
    det, truth, following = _follow(people, [0], noise=SHARP)
    assert {2, 4} <= _ids(det, following, 0)


def test_a_piece_sharing_a_frame_with_her_last_piece_still_continues_its_motion():
    rng = np.random.default_rng(12)
    a, dancer = unit(rng), unit(rng)
    people = [Person(a, lambda f: 600, lambda f: 1 if f <= 150 else (3 if f > 310 else None)),
              _piece_she_barely_leads(rng, a, 0.45, dancer),
              Person(dancer, lambda f: 1500, lambda f: 5)]
    det, truth, following = _follow(people, [0], noise=SHARP)
    assert 2 in _ids(det, following, 0)


def test_a_piece_that_fits_only_the_side_she_comes_back_on_is_hers():
    rng = np.random.default_rng(13)
    a, dancer = unit(rng), unit(rng)
    # She reappears out of reach of what she had, so that piece is hers by look alone and it can vouch for this one.
    people = [Person(a, lambda f: 600 if f < 150 else 1200, lambda f: 1 if f < 150 else (3 if f > 240 else None)),
              _piece_she_barely_leads(rng, a, 0.45, dancer, x=lambda f: 1200, frames=(150, 240)),
              Person(dancer, lambda f: 1500, lambda f: 5)]
    det, truth, following = _follow(people, [0], noise=SHARP)
    assert {2, 3} <= _ids(det, following, 0)
    assert following.report["position_rejections"] == []
