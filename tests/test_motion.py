import numpy as np

from scenes import Person, near, scene, unit
from syncscore.config import load_config
from syncscore.pose.camera import still_camera
from syncscore.pose.motion import Track, positions, split_at_crossings
from syncscore.pose.tracks import build_tracklets

CONFIG = load_config()


def _split(det):
    tracklets = build_tracklets(det, 9)
    return split_at_crossings(det, tracklets, still_camera(det.n_frames), CONFIG)


def test_a_steady_walk_is_predicted_and_doubt_grows_while_unseen():
    frames = np.arange(30)
    points = np.stack([100 + 10.0 * frames, np.full(30, 400.0)], 1)
    track = Track.follow(points, frames, height=320.0, fps=30.0)
    last = points[-1]
    ahead, behind = last + [100, 0], last - [300, 0]
    track.predict(10)
    assert np.linalg.norm(track.position - ahead) < 5
    assert track.distance(ahead) < 1
    near_doubt = track.distance(behind)
    assert near_doubt > 3
    track.predict(60)
    assert track.distance(behind) < near_doubt


def test_positions_are_hip_centres_and_body_heights_on_the_stage():
    a = unit(np.random.default_rng(0))
    det, _ = scene([Person(a, lambda f: 300 + f, lambda f: 1)], n_frames=5)
    det.kconf[2, 11] = 0.1  # one hip unsure: the box centre stands in
    points, heights = positions(det, np.arange(5), still_camera(5), CONFIG.pose.min_keypoint_conf)
    assert np.allclose(points[[0, 1, 3, 4]], [[300, 400], [301, 400], [303, 400], [304, 400]])
    assert np.allclose(points[2], [302, 360])
    assert np.allclose(heights, 320)


def _crossing(look_a, look_b, handover: bool):
    """A walks right, the other walks left; they meet at frame 150."""
    first = lambda f: 1 if (f <= 150 or not handover) else 2
    second = lambda f: 2 if (f <= 150 or not handover) else 1
    return scene([Person(look_a, lambda f: 50 + 5 * f, first), Person(look_b, lambda f: 1550 - 5 * f, second)])


def test_a_track_handed_to_a_differently_dressed_dancer_is_split_at_the_crossing():
    rng = np.random.default_rng(1)
    det, truth = _crossing(unit(rng), unit(rng), handover=True)
    pieces, splits = _split(det)
    mine = [s for s in splits if s.track_id == 1]
    assert len(mine) == 1 and abs(mine[0].frame - 150) <= 5 and mine[0].reason == "fingerprint"
    for piece in pieces:  # every piece is one person
        assert len(set(truth[piece.rows])) == 1
    assert {p.track_id for p in pieces} == {1, 2}


def test_a_track_handed_to_a_lookalike_going_the_other_way_is_split_by_heading():
    rng = np.random.default_rng(2)
    a = unit(rng)
    det, truth = _crossing(a, near(a, 0.85, rng), handover=True)
    pieces, splits = _split(det)
    assert {s.reason for s in splits} == {"heading"}
    assert all(abs(s.frame - 150) <= 5 for s in splits)
    for piece in pieces:
        assert len(set(truth[piece.rows])) == 1


def test_a_dancer_turning_around_alone_is_not_split():
    rng = np.random.default_rng(3)
    a = unit(rng)
    later = near(a, 0.5, rng)  # she turns her back: the fingerprint changes a lot
    det, _ = scene([Person(a, lambda f: 200 + 5 * f if f <= 150 else 950 - 5 * (f - 150), lambda f: 1)])
    det.vectors[det.frame > 150] = later
    pieces, splits = _split(det)
    assert splits == [] and len(pieces) == 1


def test_two_lookalikes_passing_without_a_handover_are_not_split():
    rng = np.random.default_rng(4)
    a = unit(rng)
    det, _ = _crossing(a, near(a, 0.85, rng), handover=False)
    pieces, splits = _split(det)
    assert splits == [] and len(pieces) == 2
