"""The ghost: what the other members are doing at this moment, drawn on each member's own body."""
import numpy as np

from syncscore.features.bones import bone_directions
from syncscore.synth.generate import GroupSpec, Interval, make_group
from syncscore.viz.ghost import best_lag, ghost_keypoints, pose_from_directions

DRAWN = [0] + list(range(5, 17))  # nose, shoulders, arms, hips, legs
MIN_CONF = 0.3


def _ghost(group, mirrored=None):
    m, t = group.keypoints.shape[:2]
    flags = np.zeros((m, t), dtype=bool) if mirrored is None else mirrored
    return ghost_keypoints(group.keypoints, group.conf, flags, MIN_CONF)


def test_own_directions_rebuild_the_members_own_pose():
    group = make_group(GroupSpec(n_members=3, duration_s=4, fps=30, seed=50))
    xy, conf = group.keypoints[1], group.conf[1]
    f = bone_directions(xy, conf, MIN_CONF)
    rebuilt = pose_from_directions(f.u, f.weight, xy, conf, MIN_CONF)
    assert np.allclose(rebuilt[:, DRAWN], xy[:, DRAWN], atol=1e-6)
    assert np.isnan(rebuilt[:, 1:5]).all()


def test_a_late_member_trails_the_ghost():
    group = make_group(GroupSpec(n_members=5, duration_s=20, fps=30, offsets_ms=[0, 0, 133, 0, 0], seed=51))
    ghost = _ghost(group)
    for member, expected in ((2, 4), (0, 0)):
        own = bone_directions(group.keypoints[member], group.conf[member], MIN_CONF).u
        theirs = bone_directions(ghost[member], np.where(np.isfinite(ghost[member][..., 0]), 1.0, 0.0), MIN_CONF).u
        assert best_lag(own, theirs, max_lag=10) == expected


def test_unseen_member_has_no_ghost():
    group = make_group(GroupSpec(n_members=4, duration_s=8, fps=30, occlusions=[Interval(1, 3.0, 5.0)], seed=52))
    ghost = _ghost(group)
    hidden = (group.times >= 3.0) & (group.times < 5.0)
    assert np.isnan(ghost[1, hidden]).all()
    assert np.isfinite(ghost[1, ~hidden][:, DRAWN]).all()


def test_a_mirrored_member_gets_a_mirrored_ghost():
    group = make_group(GroupSpec(n_members=4, duration_s=6, fps=30, mirrored=[False, False, False, True], seed=53))
    t = group.keypoints.shape[1]
    flags = np.zeros((4, t), dtype=bool)
    flags[3] = True
    own = group.keypoints[3][:, 5:17]
    right = np.nanmedian(np.linalg.norm(_ghost(group, flags)[3][:, 5:17] - own, axis=-1))
    wrong = np.nanmedian(np.linalg.norm(_ghost(group)[3][:, 5:17] - own, axis=-1))
    assert right < 2.0
    assert wrong > 20.0
