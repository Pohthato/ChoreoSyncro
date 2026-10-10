import numpy as np
import pytest

from syncscore.features.bones import (
    BONE_NAMES,
    BONES,
    FLIP,
    PARTS,
    angle_between,
    bone_directions,
    mirror_keypoints,
    mirror_name,
)
from syncscore.synth.generate import GroupSpec, make_group


def _random_pose(rng, frames=10):
    xy = rng.uniform(0, 500, size=(frames, 17, 2))
    conf = np.full((frames, 17), 0.9)
    return xy, conf


def test_flip_is_an_involution():
    assert np.array_equal(FLIP[FLIP], np.arange(17))


def test_mirror_twice_is_identity():
    xy, conf = _random_pose(np.random.default_rng(0))
    back, back_conf = mirror_keypoints(*mirror_keypoints(xy, conf))
    assert np.allclose(back, xy)
    assert np.allclose(back_conf, conf)


def test_mirror_name_swaps_sides_only():
    assert mirror_name("left_forearm") == "right_forearm"
    assert mirror_name("right_shin") == "left_shin"
    assert mirror_name("spine") == "spine"


def test_every_bone_belongs_to_a_part():
    assert {bone.part for bone in BONES} == set(PARTS)


def test_directions_ignore_position_and_scale():
    xy, conf = _random_pose(np.random.default_rng(1))
    a = bone_directions(xy, conf)
    b = bone_directions(xy * 2.7 + np.array([140.0, -60.0]), conf)
    assert np.allclose(a.u, b.u)


def test_low_confidence_bones_are_unknown():
    xy, conf = _random_pose(np.random.default_rng(2))
    conf[:, 9] = 0.1  # left wrist
    features = bone_directions(xy, conf, min_conf=0.3)
    forearm = BONE_NAMES.index("left_forearm")
    assert np.isnan(features.u[:, forearm]).all()
    assert (features.weight[:, forearm] == 0).all()


def test_angle_between():
    right = np.array([1.0, 0.0])
    up = np.array([0.0, 1.0])
    assert angle_between(right, up) == pytest.approx(90.0)
    assert angle_between(right, -right) == pytest.approx(180.0)
    assert np.isnan(angle_between(right, np.array([np.nan, np.nan])))


def test_tall_and_short_member_doing_the_same_move_are_identical():
    # Two members with different heights, limb proportions, stage positions and distance to
    # the camera, doing the same choreography at the same time with no noise.
    group = make_group(GroupSpec(n_members=2, duration_s=6.0, seed=3))
    heights = [body.height for body in group.bodies]
    assert abs(heights[0] - heights[1]) > 20  # really different sizes on screen
    a = bone_directions(group.keypoints[0], group.conf[0])
    b = bone_directions(group.keypoints[1], group.conf[1])
    assert not np.allclose(group.keypoints[0], group.keypoints[1])
    assert np.allclose(a.u, b.u, atol=1e-9)


def test_mirrored_member_matches_after_mirroring():
    group = make_group(GroupSpec(n_members=2, duration_s=6.0, mirrored=[False, True], seed=4))
    plain = bone_directions(group.keypoints[0], group.conf[0])
    raw = bone_directions(group.keypoints[1], group.conf[1])
    fixed = bone_directions(*mirror_keypoints(group.keypoints[1], group.conf[1]))
    assert not np.allclose(plain.u, raw.u, atol=1e-3)
    assert np.allclose(plain.u, fixed.u, atol=1e-9)
