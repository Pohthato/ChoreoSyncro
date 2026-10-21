import numpy as np

from syncscore.features.bones import BONE_NAMES, angle_between, bone_directions
from syncscore.synth.generate import GroupSpec, Interval, LimbError, make_group


def test_shapes_and_truth():
    group = make_group(GroupSpec(n_members=3, duration_s=4.0, fps=30.0, offsets_ms=[0, 33, -67]))
    frames = 120
    assert group.keypoints.shape == (3, frames, 17, 2)
    assert group.conf.shape == (3, frames, 17)
    assert np.allclose(group.offsets_ms[:, 0], [0, 33, -67])


def test_offset_is_a_pure_time_shift():
    # A member 100 ms late must at time t + 0.1 s hold exactly the pose the on-time member
    # holds at t (same body, so keypoints are directly comparable after removing position).
    spec = GroupSpec(n_members=2, duration_s=6.0, fps=50.0, offsets_ms=[0, 100], same_body=True)
    group = make_group(spec)
    shift = 5  # 100 ms at 50 fps
    a = bone_directions(group.keypoints[0], group.conf[0]).u[:-shift]
    b = bone_directions(group.keypoints[1], group.conf[1]).u[shift:]
    assert np.allclose(a, b, atol=1e-9)


def test_limb_error_rotates_only_that_bone():
    error = LimbError(member=1, joint="l_elbow", degrees=20.0, t0=2.0, t1=4.0, ramp_s=0.0)
    group = make_group(GroupSpec(n_members=2, duration_s=6.0, fps=30.0, limb_errors=[error]))
    a = bone_directions(group.keypoints[0], group.conf[0]).u
    b = bone_directions(group.keypoints[1], group.conf[1]).u
    angles = angle_between(a, b)  # (frames, bones)
    inside = slice(int(2.2 * 30), int(3.8 * 30))
    forearm = BONE_NAMES.index("left_forearm")
    assert np.allclose(angles[inside, forearm], 20.0, atol=1e-6)
    others = [i for i in range(len(BONE_NAMES)) if i != forearm]
    assert np.allclose(angles[inside][:, others], 0.0, atol=1e-6)
    assert np.allclose(angles[:int(1.8 * 30)], 0.0, atol=1e-6)
    assert group.limb_error_bones == ["left_forearm"]


def test_occlusion_removes_the_member():
    from syncscore.synth.generate import Interval

    group = make_group(GroupSpec(n_members=2, duration_s=4.0, fps=30.0, occlusions=[Interval(1, 1.0, 2.0)]))
    assert (group.conf[1, 30:60] == 0).all()
    assert np.isnan(group.keypoints[1, 30:60]).all()
    assert (group.conf[1, :30] > 0).all()


def test_seed_makes_it_reproducible():
    spec = GroupSpec(n_members=3, duration_s=3.0, noise=0.01, dropout=0.05, seed=7)
    assert np.array_equal(make_group(spec).keypoints, make_group(spec).keypoints, equal_nan=True)


def test_cropped_member_loses_only_the_lower_legs():
    from syncscore.synth.generate import Interval

    group = make_group(GroupSpec(n_members=2, duration_s=4.0, fps=30.0, cropped=[Interval(0, 1.0, 3.0)]))
    assert (group.conf[0, 30:90, 13:17] == 0).all()
    assert (group.conf[0, 30:90, :13] > 0).all()


def test_loop_repeats_with_its_period():
    group = make_group(GroupSpec(n_members=1, duration_s=12.0, fps=50.0, loops=[(4.0, 10.0, 0.18, 2)]))
    u = bone_directions(group.keypoints[0], group.conf[0]).u
    period = 18  # 0.36 s at 50 fps
    a, b = u[int(5 * 50):int(9 * 50)], u[int(5 * 50) + period:int(9 * 50) + period]
    assert np.allclose(a, b, atol=1e-6)


def test_glitches_swap_labels_or_move_a_wrist_at_full_confidence():
    clean = make_group(GroupSpec(n_members=1, duration_s=10.0, fps=30.0, seed=5))
    glitchy = make_group(GroupSpec(n_members=1, duration_s=10.0, fps=30.0, seed=5, glitches=0.05))
    changed = ~np.isclose(clean.keypoints[0], glitchy.keypoints[0]).all(axis=(1, 2))
    assert 0.02 < changed.mean() < 0.15
    assert (glitchy.conf[0] > 0.8).all()  # glitches are not marked as uncertain


# --- solos and split parts ---

import hashlib  # noqa: E402

from syncscore.features.bones import BONE_NAMES, angle_between, bone_directions  # noqa: E402

_LIMBS = [BONE_NAMES.index(n) for n in ("left_upper_arm", "left_forearm", "right_upper_arm", "right_forearm",
                                        "left_thigh", "left_shin", "right_thigh", "right_shin")]


def _limb_angle(group, a, b, frames):
    ua = bone_directions(group.keypoints[a], group.conf[a]).u[frames][:, _LIMBS]
    ub = bone_directions(group.keypoints[b], group.conf[b]).u[frames][:, _LIMBS]
    return float(np.nanmedian(angle_between(ua, ub)))


def test_in_a_solo_the_others_hold_still():
    group = make_group(GroupSpec(n_members=5, duration_s=20, fps=30, solos=[Interval(2, 8.0, 14.0)], seed=61))
    inside = (group.times > 8.2) & (group.times < 13.8)
    step = np.linalg.norm(np.diff(group.keypoints, axis=1), axis=-1).max(axis=-1)  # (M, T-1) largest joint move per frame
    for member in (0, 1, 3, 4):
        assert step[member, inside[1:]].max() < 0.5
    assert step[2, inside[1:]].max() > 5.0


def test_a_part_dances_other_moves():
    group = make_group(GroupSpec(n_members=5, duration_s=20, fps=30, parts=[([3, 4], 8.0, 14.0)], seed=62))
    inside = np.flatnonzero((group.times > 8.5) & (group.times < 13.5))
    assert _limb_angle(group, 3, 4, inside) < 3.0
    assert _limb_angle(group, 0, 3, inside) > 10.0
    before = np.flatnonzero(group.times < 7.5)
    assert _limb_angle(group, 0, 3, before) < 3.0


def test_existing_seeds_are_unchanged():
    # Checksums of make_group output taken before solos and parts existed.
    specs = (GroupSpec(n_members=4, duration_s=6, noise=0.01, style_deg=3.0, seed=7),
             GroupSpec(n_members=3, duration_s=5, offsets_ms=[0, 40, -30], dropout=0.05, occlusions=[Interval(1, 1.0, 2.0)], seed=11))
    expected = ("a63a7482a50fc9f124fe4f86e967eeb6e420863a", "0b38b59708539389238ef05153b484f0584a0f81")
    for spec, digest in zip(specs, expected):
        g = make_group(spec)
        assert hashlib.sha1(np.nan_to_num(g.keypoints).round(6).tobytes() + g.conf.round(6).tobytes()).hexdigest() == digest
