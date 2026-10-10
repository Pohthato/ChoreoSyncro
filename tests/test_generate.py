import numpy as np

from syncscore.features.bones import BONE_NAMES, angle_between, bone_directions
from syncscore.synth.generate import GroupSpec, LimbError, make_group


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
