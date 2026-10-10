"""End-to-end recovery on synthetic groups: known offsets and limb errors go in, are they measured?"""
import dataclasses

import numpy as np
import pytest

from syncscore.config import load_config
from syncscore.features.bones import BONE_NAMES, PARTS
from syncscore.sync.measure import measure_sync
from syncscore.synth.generate import GroupSpec, LimbError, make_group


@pytest.fixture(scope="module")
def config():
    return load_config()


def _relative_truth(group, result):
    """True offset per member per window, relative to the median member (the group's timing)."""
    frames = np.arange(group.keypoints.shape[1])
    truth = np.stack([np.interp(result.window_centers_s * group.fps, frames, group.offsets_ms[i]) for i in range(len(group.names))])
    return truth - np.median(truth, axis=0)


def _member_offsets(result):
    est = np.where(result.reliable, result.offset_ms, np.nan)
    return np.nanmedian(est, axis=1)


def test_known_constant_offsets_are_recovered(config):
    offsets = [0, 33, -67, 133, 17]
    group = make_group(GroupSpec(n_members=5, duration_s=30, fps=30, offsets_ms=offsets, seed=11))
    result = measure_sync(group.keypoints, group.conf, group.fps, config)
    expected = np.array(offsets) - np.median(offsets)
    assert result.reliable.mean() > 0.7
    assert np.allclose(_member_offsets(result), expected, atol=1.5)


def test_two_members_converge(config):
    # With two members each is the other's whole consensus. Without damping the estimates flip
    # between the full gap and zero every pass and never settle on +-50.
    group = make_group(GroupSpec(n_members=2, duration_s=20, fps=30, offsets_ms=[0, 100], seed=16))
    result = measure_sync(group.keypoints, group.conf, group.fps, config)
    assert np.allclose(_member_offsets(result), [-50.0, 50.0], atol=1.5)


def test_sub_frame_offset_is_recovered(config):
    # 17 ms is about half a frame at 30 fps: the hardest case for sub-frame interpolation.
    group = make_group(GroupSpec(n_members=3, duration_s=30, fps=30, offsets_ms=[0, 0, 17], seed=12))
    result = measure_sync(group.keypoints, group.conf, group.fps, config)
    assert _member_offsets(result)[2] == pytest.approx(17.0, abs=2.0)


def test_drifting_offset_is_tracked_over_time(config):
    drift = [(0.0, 0.0), (30.0, 120.0)]  # member 2 slides from on time to 120 ms late
    group = make_group(GroupSpec(n_members=4, duration_s=30, fps=30, offsets_ms=[0, 0, drift, 0], seed=13))
    result = measure_sync(group.keypoints, group.conf, group.fps, config)
    truth = _relative_truth(group, result)
    ok = result.reliable[2]
    assert np.nanmean(np.abs(result.offset_ms[2, ok] - truth[2, ok])) < 8.0


def test_limb_error_is_shape_not_timing(config):
    error = LimbError(member=1, joint="l_elbow", degrees=20.0, t0=8.0, t1=20.0)
    group = make_group(GroupSpec(n_members=5, duration_s=30, fps=30, offsets_ms=[0, 67, 0, 0, 0], limb_errors=[error], seed=14))
    result = measure_sync(group.keypoints, group.conf, group.fps, config)
    # Timing is unaffected: member 2 (index 1) is still measured 67 ms late.
    assert _member_offsets(result)[1] == pytest.approx(67.0, abs=4.0)
    inside = slice(int(9 * 30), int(19 * 30))
    bone_err = np.nanmean(result.bone_shape_deg[1, inside], axis=0)
    assert BONE_NAMES[int(np.nanargmax(bone_err))] == "left_forearm"
    assert bone_err[BONE_NAMES.index("left_forearm")] == pytest.approx(20.0, abs=2.0)
    part_err = np.nanmean(result.part_shape_deg[1, inside], axis=0)
    assert PARTS[int(np.nanargmax(part_err))] == "arms"
    # Nobody else picks up the error through the consensus.
    others = np.nanmean(result.bone_shape_deg[[0, 2, 3, 4]][:, inside, BONE_NAMES.index("left_forearm")])
    assert others < 2.0


@pytest.mark.parametrize("mirror", [[3], "auto"])
def test_mirrored_member(config, mirror):
    cfg = dataclasses.replace(config, features=dataclasses.replace(config.features, mirror=mirror))
    group = make_group(GroupSpec(n_members=5, duration_s=30, fps=30, offsets_ms=[0, 0, 100, 0, 0], mirrored=[False, False, True, False, False], seed=15))
    result = measure_sync(group.keypoints, group.conf, group.fps, cfg)
    assert result.mirrored[2].mean() > 0.9 and result.mirrored[[0, 1, 3, 4]].mean() < 0.05
    assert _member_offsets(result)[2] == pytest.approx(100.0, abs=4.0)
    assert np.nanmean(result.part_shape_deg[2]) < 2.0
