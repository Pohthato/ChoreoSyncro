"""Clean landings: big moves that end in stillness, timed to a fraction of a frame."""
import dataclasses

import numpy as np

from syncscore.audio.beats import BeatGrid
from syncscore.config import load_config
from syncscore.features.bones import bone_directions
from syncscore.sync.landings import clean_landings
from syncscore.sync.timing import velocity
from syncscore.sync.unison import member_speed
from syncscore.synth.generate import GroupSpec, make_choreography, make_group
from syncscore.synth.real_shift import delay_member

CFG = load_config()
# Synthetic moves are all alike (every one a single smooth stroke), so their peaks sit lower above
# the member's usual speed than real hits do; the mechanism is what's tested here.
SYNTH = dataclasses.replace(CFG.beat, landing_peak=1.5)


def _speed(xy, conf, fps):
    f = bone_directions(xy, conf, CFG.pose.min_keypoint_conf)
    v, vw = velocity(f.u, f.weight, fps, CFG.sync.max_bone_speed_deg_s)
    return member_speed(v, vw)


def test_moves_into_a_hold_are_found():
    spec = GroupSpec(n_members=1, duration_s=30, fps=30, seed=81)
    group = make_group(spec)
    plan = make_choreography(spec.duration_s, spec.bpm, np.random.default_rng(spec.seed))  # make_group's first draw
    changes = np.flatnonzero(np.any(plan.poses[1:] != plan.poses[:-1], axis=1))
    move_ends = (plan.times[changes] + plan.move_fraction * (plan.times[changes + 1] - plan.times[changes])) * spec.fps
    found = clean_landings(_speed(group.keypoints[0], group.conf[0], spec.fps), spec.fps, SYNTH)
    assert len(found) >= 20
    nearest = np.min(np.abs(found[:, None] - move_ends[None, :]), axis=1)
    assert np.all(nearest <= 2.0)


def test_a_planted_delay_moves_every_landing():
    group = make_group(GroupSpec(n_members=2, duration_s=30, fps=30, style_deg=3.0, seed=82))
    base = clean_landings(_speed(group.keypoints[1], group.conf[1], group.fps), group.fps, SYNTH)
    xy, conf = delay_member(group.keypoints, group.conf, member=1, shift_ms=40.0, fps=group.fps)
    moved = clean_landings(_speed(xy[1], conf[1], group.fps), group.fps, SYNTH)
    pairs = [(a, moved[np.argmin(np.abs(moved - a - 1.2))]) for a in base]
    shift_ms = np.array([(b - a) / group.fps * 1000 for a, b in pairs if abs(b - a - 1.2) < 2])
    assert len(shift_ms) >= 0.8 * len(base)
    assert abs(np.median(shift_ms) - 40.0) < 3.0


def test_continuous_motion_has_no_landings():
    t = np.arange(600) / 30.0
    speed = 2.0 + np.sin(2 * np.pi * 1.5 * t)  # never drops below 1: no stillness to land in
    assert len(clean_landings(speed, 30.0, CFG.beat)) == 0


def test_offsets_wrap_at_half_a_half_beat():
    grid = BeatGrid(tempo_bpm=120.0, half_beat_s=0.25, beat_phase_s=0.0, steady=True, beat_share=1.0)
    assert np.allclose(grid.offset_ms(np.array([10.130, 10.020, 10.250 - 0.030])), [-120.0, 20.0, -30.0])


def test_landings_spread_over_the_beat_are_not_locked():
    from syncscore.sync.landings import landing_lock

    rng = np.random.default_rng(3)
    lock = landing_lock(rng.uniform(-125, 125, 60), half_beat_ms=250.0)
    assert not lock["locked"] and lock["n"] == 60 and abs(lock["chance"] - 0.4) < 1e-9


def test_landings_bunched_near_the_beat_are_locked():
    from syncscore.sync.landings import landing_lock

    rng = np.random.default_rng(4)
    lock = landing_lock(rng.normal(10, 20, 40), half_beat_ms=250.0)
    assert lock["locked"] and lock["within_50ms"] > 0.8 and lock["p"] < 0.01


def test_too_few_landings_are_never_locked():
    from syncscore.sync.landings import landing_lock

    assert not landing_lock(np.zeros(10), half_beat_ms=250.0)["locked"]


def test_the_lock_says_how_many_landings_it_needs():
    from syncscore.sync.landings import MIN_LOCK_LANDINGS, landing_lock

    assert landing_lock(np.zeros(3), half_beat_ms=250.0)["min_n"] == MIN_LOCK_LANDINGS
    assert landing_lock(np.zeros(0), half_beat_ms=250.0)["min_n"] == MIN_LOCK_LANDINGS


def test_the_shipped_peak_keeps_only_big_moves():
    # Speed in units of the usual moving speed, holding still at 0.1 between strokes of 0.3 s. Ordinary
    # moves peak at 1.8, big ones at 3; each falls into a hold. With the shipped settings
    # (landing_peak 2.5) only the big ones land, three quarters of the way down: 0.25 s into the stroke.
    fps = 30.0
    t = np.arange(int(20 * fps)) / fps
    speed = np.full(t.size, 0.1)
    big, ordinary = [2.0, 6.0, 10.0, 14.0], [4.0, 8.0, 12.0, 16.0]
    for start, peak in [(s, 3.0) for s in big] + [(s, 1.8) for s in ordinary]:
        inside = (t >= start) & (t <= start + 0.3)
        speed[inside] = 0.1 + (peak - 0.1) * np.sin(np.pi * (t[inside] - start) / 0.3) ** 2
    found = clean_landings(speed, fps, CFG.beat) / fps
    assert len(found) == 4
    assert np.all(np.abs(found - (np.array(big) + 0.25)) < 1.5 / fps)
