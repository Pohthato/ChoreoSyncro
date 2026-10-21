"""When is the group dancing together? Only then is the group a reference for one member."""
import numpy as np

from syncscore.config import load_config
from syncscore.features.bones import bone_directions
from syncscore.sync.timing import velocity
from syncscore.sync.unison import apart_intervals, member_speed, together
from syncscore.synth.generate import GroupSpec, Interval, make_group

CFG = load_config()


def _together(keypoints, conf, fps):
    u, w, speed = [], [], []
    for i in range(len(keypoints)):
        f = bone_directions(keypoints[i], conf[i], CFG.pose.min_keypoint_conf)
        v, vw = velocity(f.u, f.weight, fps, CFG.sync.max_bone_speed_deg_s)
        u.append(f.u)
        w.append(f.weight)
        speed.append(member_speed(v, vw))
    return together(np.stack(u), np.stack(w), np.stack(speed), fps, CFG.unison)


def _share(flags, times, t0, t1):
    inside = (times >= t0) & (times < t1)
    return float(flags[inside].mean())


def test_a_group_in_unison_is_together():
    group = make_group(GroupSpec(n_members=5, duration_s=20, fps=30, seed=63))
    assert _together(group.keypoints, group.conf, group.fps).mean() >= 0.95


def test_a_group_holding_still_is_together():
    group = make_group(GroupSpec(n_members=5, duration_s=20, fps=30, seed=64))
    hold = (group.times >= 6.0) & (group.times < 12.0)
    first = np.flatnonzero(hold)[0]
    group.keypoints[:, hold] = group.keypoints[:, [first]]
    flags = _together(group.keypoints, group.conf, group.fps)
    assert all(_share(flags[i], group.times, 7.0, 11.0) >= 0.95 for i in range(5))


def test_a_soloist_is_not_together_during_the_solo():
    group = make_group(GroupSpec(n_members=5, duration_s=20, fps=30, solos=[Interval(2, 8.0, 14.0)], seed=65))
    flags = _together(group.keypoints, group.conf, group.fps)
    assert _share(flags[2], group.times, 9.0, 13.0) <= 0.10
    outside = (group.times < 7.0) | (group.times >= 15.0)
    assert flags[2, outside].mean() >= 0.90


def test_split_parts_are_not_together():
    group = make_group(GroupSpec(n_members=7, duration_s=20, fps=30, parts=[([4, 5, 6], 8.0, 14.0)], seed=66))
    flags = _together(group.keypoints, group.conf, group.fps)
    for i in range(7):
        assert _share(flags[i], group.times, 9.0, 13.0) <= 0.20, i


def test_unseen_member_is_not_together():
    group = make_group(GroupSpec(n_members=5, duration_s=12, fps=30, occlusions=[Interval(1, 4.0, 8.0)], seed=67))
    flags = _together(group.keypoints, group.conf, group.fps)
    assert not flags[1, (group.times >= 4.0) & (group.times < 8.0)].any()


def test_apart_intervals_merge_runs():
    fps = 30.0
    row = np.ones(300, dtype=bool)
    row[60:120] = False   # 2 s apart
    row[200:215] = False  # 0.5 s: too short to mark
    out = apart_intervals(row[None], fps, min_s=1.0)
    assert len(out) == 1 and len(out[0]) == 1
    start, end = out[0][0]
    assert abs(start - 2.0) < 1e-9 and abs(end - 4.0) < 1e-9
