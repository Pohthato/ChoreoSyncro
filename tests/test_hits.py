"""Timing at the group's hits: known delays go in, does each member's peak come back where it should?"""
import dataclasses

import numpy as np
import pytest

from syncscore.config import load_config
from syncscore.features.bones import bone_directions
from syncscore.sync.hits import hit_timing, relative_offsets
from syncscore.sync.sections import SectionTiming, combine_sections, hit_section_timing
from syncscore.sync.timing import velocity
from syncscore.synth.generate import GroupSpec, Interval, make_group
from syncscore.synth.real_shift import _hit_errors, delay_member


@pytest.fixture(scope="module")
def config():
    return load_config()


def _hits(xy, conf, fps, config):
    """The same velocities measure_sync hands to hit_timing (no mirroring)."""
    moving = [velocity(*_directions(xy[i], conf[i], config), fps, config.sync.max_bone_speed_deg_s) for i in range(len(xy))]
    return hit_timing([v for v, _ in moving], [w for _, w in moving], fps, config.hits)


def _directions(xy, conf, config):
    f = bone_directions(xy, conf, config.pose.min_keypoint_conf)
    return f.u, f.weight


def test_offsets_are_recovered_to_a_fraction_of_a_frame(config):
    offsets = [0, 0, 0, 20, -40]  # 20 ms is 0.6 of a frame at 30 fps
    group = make_group(GroupSpec(n_members=5, duration_s=30, fps=30, offsets_ms=offsets, seed=41))
    hits = _hits(group.keypoints, group.conf, group.fps, config)
    assert len(hits.frames) >= 20
    matched = np.isfinite(hits.offset_ms)
    assert matched.mean() > 0.6
    median = np.array([np.median(hits.offset_ms[i, matched[i]]) for i in range(5)])
    assert median == pytest.approx(offsets, abs=3.0)


def test_a_planted_delay_moves_every_matched_peak_by_exactly_that(config):
    group = make_group(GroupSpec(n_members=5, duration_s=24, fps=30, noise=0.005, style_deg=3.0, seed=42))
    base = _hits(group.keypoints, group.conf, group.fps, config)
    xy, conf = delay_member(group.keypoints, group.conf, member=2, shift_ms=50.0, fps=group.fps)
    shifted = _hits(xy, conf, group.fps, config)
    errors, coverage = _hit_errors((base.frames / group.fps, base.offset_ms), (shifted.frames / group.fps, shifted.offset_ms), 2, 50.0, group.fps)
    assert coverage > 0.5
    assert np.median(np.abs(errors)) < 2.0
    assert np.mean(np.abs(errors) > 30) < 0.05


def _pulse(frames, center, width=2.0):
    return np.exp(-0.5 * ((np.arange(frames) - center) / width) ** 2)


def _member(frames, strokes):
    """Bone velocities for a toy member: bone 0 moves in `strokes` [(frame, direction), ...] (x
    component, +1 or -1), bone 1 drifts slowly all the time so the member has a typical speed."""
    v = np.zeros((frames, 2, 2))
    for center, direction in strokes:
        v[:, 0, 0] += 4.0 * direction * _pulse(frames, center)
    v[:, 1, 1] = 0.5
    return v, np.ones((frames, 2))


def test_offsets_are_relative_to_the_median_member_whoever_is_missing():
    peak = np.tile(np.array([0.0, 40, -30, 90, 0])[:, None], (1, 6)) + np.arange(6)[None] * 100  # 6 hits
    peak[3, 2] = np.nan  # the 90 ms member unseen at one hit
    offset = relative_offsets(peak, min_others=3)
    # A plain median of the four others would give -50 for the -30 member, and -20 for the on-time
    # members at the hit where the late member is missing.
    assert np.allclose(np.nanmedian(offset, axis=1), [0, 40, -30, 90, 0], atol=0.01)
    assert np.allclose(offset[[0, 1, 2, 4], 2], [0, 40, -30, 0], atol=0.01)


def test_the_return_stroke_does_not_steal_the_match(config):
    # Everyone moves bone 0 in +x at frame 100. The last member does it 6 frames (200 ms) late,
    # and brings the arm back (-x) 6 frames early: two equal speed peaks, one each side. Speed
    # alone is a tie; the direction says which one is the hit.
    frames = 200
    members = [_member(frames, [(100, +1)]) for _ in range(4)] + [_member(frames, [(94, -1), (106, +1)])]
    hits = hit_timing([v for v, _ in members], [w for _, w in members], 30.0, config.hits)
    assert len(hits.frames) == 1 and hits.frames[0] == pytest.approx(100, abs=1)
    assert hits.offset_ms[4, 0] == pytest.approx(200.0, abs=2.0)
    assert np.allclose(hits.offset_ms[:4, 0], 0.0, atol=1.0)


def test_a_tie_that_direction_cannot_break_is_no_reading(config):
    # Two equal strokes the same way, one each side of the hit: either could be the hit.
    frames = 200
    members = [_member(frames, [(100, +1)]) for _ in range(4)] + [_member(frames, [(94, +1), (106, +1)])]
    hits = hit_timing([v for v, _ in members], [w for _, w in members], 30.0, config.hits)
    assert np.isnan(hits.offset_ms[4, 0])
    assert np.isfinite(hits.offset_ms[:4, 0]).all()


def test_a_hidden_member_has_no_hits_while_hidden(config):
    group = make_group(GroupSpec(n_members=5, duration_s=24, fps=30, occlusions=[Interval(1, 8.0, 16.0)], seed=43))
    hits = _hits(group.keypoints, group.conf, group.fps, config)
    times = hits.frames / group.fps
    hidden = (times > 8.4) & (times < 15.6)  # the whole +-400 ms search range is hidden
    assert hidden.any()
    assert np.isnan(hits.offset_ms[1, hidden]).all()
    assert np.isfinite(hits.offset_ms[1, (times < 7) | (times > 17)]).any()


@pytest.fixture(scope="module")
def hit_sections():
    from syncscore.pipeline import analyze_keypoints
    from syncscore.sync.summary import make_sections

    base = load_config()
    config = dataclasses.replace(base, sections=dataclasses.replace(base.sections, chunk_s=20.0))
    offsets = [0, 40, -30, 90, 0]
    group = make_group(GroupSpec(n_members=5, duration_s=60, fps=30, offsets_ms=offsets, noise=0.01, style_deg=3.0, seed=44))
    result, _, _ = analyze_keypoints(group.keypoints, group.conf, group.fps, config)
    timing = hit_section_timing(result, make_sections(config, 60.0), config)
    return np.array(offsets, float), timing


def test_hit_sections_recover_known_offsets_with_honest_intervals(hit_sections):
    truth, timing = hit_sections
    rel = (truth - np.median(truth))[:, None]
    ok = timing.reliable
    assert ok.mean() > 0.7
    assert np.median(np.abs(timing.offset_ms - rel)[ok]) < 6.0
    inside = (timing.low_ms - 3.0 <= rel) & (rel <= timing.high_ms + 3.0)
    assert inside[ok].mean() > 0.8
    assert timing.from_hits.all()


def _timing(offset, low, high, reliable):
    a = np.array
    shape = np.shape(offset)
    return SectionTiming(starts_s=a([0.0, 10.0]), ends_s=a([10.0, 20.0]), names=["a", "b"], offset_ms=a(offset, float),
                         low_ms=a(low, float), high_ms=a(high, float), windows=np.full(shape, 9), blocks=np.full(shape, 5),
                         enough=a(reliable), reliable=a(reliable))


def test_combining_takes_the_more_certain_reliable_reading():
    windows = _timing([[10, 20]], [[0, 0]], [[20, 40]], [[True, True]])  # widths 20 and 40
    hits = _timing([[11, 25]], [[0, 15]], [[30, 35]], [[True, True]])    # widths 30 and 20
    combined = combine_sections(windows, hits)
    assert combined.offset_ms.tolist() == [[10, 25]]
    assert combined.from_hits.tolist() == [[False, True]]
    only_hits = combine_sections(_timing([[10, 20]], [[0, 0]], [[20, 40]], [[False, False]]), hits)
    assert only_hits.reliable.all() and only_hits.from_hits.all()
