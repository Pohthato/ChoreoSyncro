"""Synthetic dance groups with known answers.

To prove the method works we need videos where the truth is known: exactly how late each member
is, and exactly which limb is wrong by how many degrees. Real footage never comes with that, so
this module builds a group from scratch:

1. One choreography: a list of key poses (joint angles), one per beat or half beat. Between key
   poses the joints move with a minimum-jerk profile (the smooth S-curve human reaching
   movements follow) during the first part of the interval, then hold. Move then hold is how
   "hits" in K-pop choreography look.
2. Each member gets their own body (height, limb proportions, stage position) and performs the
   choreography at time t - offset(t), so an offset is an exact, sub-frame time shift.
3. Optional corruptions with known values: a limb held at the wrong angle, keypoint noise,
   dropped keypoints, occlusions, spins, mirrored members.

The output looks like what pose extraction produces (COCO-17 keypoints in pixels plus
confidences), so everything after pose extraction is tested exactly as it runs on video.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..features.bones import FLIP, mirror_name

JOINTS = (
    "lean", "head",
    "l_shoulder", "l_elbow", "r_shoulder", "r_elbow",
    "l_hip", "l_knee", "r_hip", "r_knee",
    "bounce", "sway",
)
_J = {name: i for i, name in enumerate(JOINTS)}
_ANGLE_JOINTS = 10  # the first ten are angles in degrees; bounce and sway are fractions of height
_RANGES = np.array([
    (-15, 15), (-25, 25),
    (-10, 170), (0, 140), (-10, 170), (0, 140),
    (-5, 50), (0, 80), (-5, 50), (0, 80),
    (-0.05, 0.0), (-0.06, 0.06),
])
_LOWER_LEGS = [13, 14, 15, 16]  # knees and ankles
_REST = np.array([0, 0, 10, 10, 10, 10, 5, 5, 5, 5, 0, 0], dtype=float)
# The bone whose direction a joint error changes (elbow -> forearm and so on).
JOINT_BONE = {"l_elbow": "left_forearm", "r_elbow": "right_forearm", "l_knee": "left_shin", "r_knee": "right_shin", "head": "head"}

_PROPORTIONS = {  # segment lengths as a fraction of standing height
    "upper_arm": 0.186, "forearm": 0.146, "thigh": 0.245, "shin": 0.246,
    "spine": 0.29, "shoulder_w": 0.23, "hip_w": 0.17, "head": 0.13,
}


@dataclass
class Body:
    height: float      # standing height on screen, pixels
    center_x: float    # stage position, pixels
    floor_y: float     # where the feet are, pixels
    proportions: dict


@dataclass
class LimbError:
    """Member `member` holds `joint` off by `degrees` between t0 and t1 (seconds)."""
    member: int
    joint: str
    degrees: float
    t0: float
    t1: float
    ramp_s: float = 0.15


@dataclass
class Interval:
    member: int
    t0: float
    t1: float


@dataclass
class GroupSpec:
    n_members: int = 5
    duration_s: float = 24.0
    fps: float = 30.0
    bpm: float = 120.0
    offsets_ms: list | None = None   # per member: a number, or a list of (t_s, ms) knots
    mirrored: list | None = None     # per member: True if they dance the mirrored version
    limb_errors: list = field(default_factory=list)
    noise: float = 0.0               # keypoint noise std, as a fraction of body height
    dropout: float = 0.0             # chance that a keypoint comes back with low confidence
    glitches: float = 0.0            # chance per frame of a pose-model glitch: a chain's left/right labels
                                     # swapped for 1-2 frames, or a wrist put somewhere else, still confident
    style_deg: float = 0.0           # each member's own small, slow deviation from the choreography
    spins: list = field(default_factory=list)  # (t0, duration_s) in choreography time
    spin_label_flip: bool | list = False  # pose model swaps left/right while the back faces the camera (per member if a list)
    occlusions: list = field(default_factory=list)  # Interval: member not visible at all
    cropped: list = field(default_factory=list)     # Interval: knees and ankles cut off by the frame edge
    loops: list = field(default_factory=list)       # (t0, t1, step_s, n_poses): a fast repeated groove
    same_body: bool = False          # every member gets member 0's body (position still differs)
    image_size: tuple = (1920, 1080)
    seed: int = 0


@dataclass
class SyntheticGroup:
    fps: float
    times: np.ndarray        # (T,)
    keypoints: np.ndarray    # (M, T, 17, 2)
    conf: np.ndarray         # (M, T, 17)
    offsets_ms: np.ndarray   # (M, T) true lateness of each member at each frame
    mirrored: np.ndarray     # (M,)
    bodies: list
    limb_error_bones: list   # the bone each LimbError should show up on
    names: list
    spec: GroupSpec


class Choreography:
    def __init__(self, times: np.ndarray, poses: np.ndarray, move_fraction: float = 0.6):
        self.times = times
        self.poses = poses
        self.move_fraction = move_fraction

    def angles(self, t: np.ndarray) -> np.ndarray:
        """Joint values at times t (seconds): (len(t), len(JOINTS))."""
        t = np.asarray(t, dtype=float)
        k = np.clip(np.searchsorted(self.times, t, side="right") - 1, 0, len(self.times) - 2)
        start, end = self.times[k], self.times[k + 1]
        tau = np.clip((t - start) / (self.move_fraction * (end - start)), 0.0, 1.0)
        s = tau**3 * (10 - 15 * tau + 6 * tau**2)  # minimum jerk: zero velocity and acceleration at both ends
        return self.poses[k] + s[:, None] * (self.poses[k + 1] - self.poses[k])


def _vary(pose: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A new key pose: 3 to 7 random joints move somewhere new, the rest stay."""
    pose = pose.copy()
    for j in rng.choice(len(JOINTS), size=rng.integers(3, 8), replace=False):
        pose[j] = rng.uniform(*_RANGES[j])
    return pose


def make_choreography(duration_s: float, bpm: float, rng: np.random.Generator, loops: list = (), hold_prob: float = 0.2, half_beat_prob: float = 0.25) -> Choreography:
    """Key poses on beats and half beats. A loop (t0, t1, step_s, n) instead cycles through n
    poses every step_s seconds: a groove that repeats faster than the lag search range, which
    makes the correlation ambiguous on purpose."""
    beat = 60.0 / bpm
    times, poses = [], []
    t, pose = -1.5, _REST.copy()
    while t < duration_s + 1.5:  # margin so members shifted in time still have choreography
        loop = next((spec for spec in loops if spec[0] <= t < spec[1]), None)
        if loop is not None:
            _, t1, step, count = loop
            cycle = [_vary(pose, rng) for _ in range(count)]
            n = 0
            while t < t1:
                times.append(t)
                poses.append(cycle[n % count].copy())
                n, t = n + 1, t + step
            pose = cycle[(n - 1) % count].copy()
            continue
        times.append(t)
        poses.append(pose.copy())
        if rng.random() >= hold_prob:
            pose = _vary(pose, rng)
        t += beat / 2 if rng.random() < half_beat_prob else beat
    return Choreography(np.array(times), np.array(poses))


def _spin_yaw(t: np.ndarray, spins: list) -> np.ndarray:
    yaw = np.zeros_like(t)
    for t0, duration in spins:
        tau = np.clip((t - t0) / duration, 0.0, 1.0)
        yaw += 2 * np.pi * tau**3 * (10 - 15 * tau + 6 * tau**2)
    return yaw


def _rotate(v: np.ndarray, radians: np.ndarray) -> np.ndarray:
    c, s = np.cos(radians), np.sin(radians)
    return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], axis=-1)


def pose_keypoints(values: np.ndarray, yaw: np.ndarray, body: Body) -> np.ndarray:
    """Forward kinematics: joint values (T, J) -> COCO-17 keypoints (T, 17, 2) in image pixels.

    Built facing the camera in a y-up frame centred on the pelvis, so the dancer's left side is
    image +x. A spin rotates the body about the vertical axis; seen from the front that squashes
    every horizontal offset by cos(yaw).
    """
    p, h = body.proportions, body.height
    a = np.radians(values[:, :_ANGLE_JOINTS])
    lean = a[:, _J["lean"]]
    up = np.stack([np.sin(lean), np.cos(lean)], axis=-1)
    side = np.stack([np.cos(lean), -np.sin(lean)], axis=-1)  # toward the dancer's left
    down = -up

    pelvis = np.zeros_like(up)
    neck = pelvis + up * p["spine"] * h
    l_shoulder, r_shoulder = neck + side * p["shoulder_w"] * h / 2, neck - side * p["shoulder_w"] * h / 2
    l_hip, r_hip = pelvis + side * p["hip_w"] * h / 2, pelvis - side * p["hip_w"] * h / 2

    l_upper = _rotate(down, a[:, _J["l_shoulder"]])
    r_upper = _rotate(down, -a[:, _J["r_shoulder"]])
    l_elbow = l_shoulder + l_upper * p["upper_arm"] * h
    r_elbow = r_shoulder + r_upper * p["upper_arm"] * h
    l_wrist = l_elbow + _rotate(l_upper, a[:, _J["l_elbow"]]) * p["forearm"] * h
    r_wrist = r_elbow + _rotate(r_upper, -a[:, _J["r_elbow"]]) * p["forearm"] * h

    l_thigh = _rotate(down, a[:, _J["l_hip"]])
    r_thigh = _rotate(down, -a[:, _J["r_hip"]])
    l_knee = l_hip + l_thigh * p["thigh"] * h
    r_knee = r_hip + r_thigh * p["thigh"] * h
    l_ankle = l_knee + _rotate(l_thigh, -a[:, _J["l_knee"]]) * p["shin"] * h
    r_ankle = r_knee + _rotate(r_thigh, a[:, _J["r_knee"]]) * p["shin"] * h

    head_up = _rotate(up, a[:, _J["head"]])
    head_side = np.stack([head_up[:, 1], -head_up[:, 0]], axis=-1)
    hl = p["head"] * h
    nose = neck + head_up * hl
    l_eye = neck + head_up * hl * 1.15 + head_side * hl * 0.18
    r_eye = neck + head_up * hl * 1.15 - head_side * hl * 0.18
    l_ear = neck + head_up * hl + head_side * hl * 0.45
    r_ear = neck + head_up * hl - head_side * hl * 0.45

    body_xy = np.stack([
        nose, l_eye, r_eye, l_ear, r_ear, l_shoulder, r_shoulder, l_elbow, r_elbow,
        l_wrist, r_wrist, l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle,
    ], axis=1)
    body_xy[..., 0] *= np.cos(yaw)[:, None]

    leg = (p["thigh"] + p["shin"]) * h
    pelvis_x = body.center_x + values[:, _J["sway"]] * h
    pelvis_y = body.floor_y - leg - values[:, _J["bounce"]] * h
    out = np.empty_like(body_xy)
    out[..., 0] = pelvis_x[:, None] + body_xy[..., 0]
    out[..., 1] = pelvis_y[:, None] - body_xy[..., 1]  # image y points down
    return out


def _curve(value, times: np.ndarray) -> np.ndarray:
    if np.isscalar(value):
        return np.full_like(times, float(value))
    knots = np.asarray(value, dtype=float)
    return np.interp(times, knots[:, 0], knots[:, 1])


def _style(rng: np.random.Generator, times: np.ndarray, degrees: float) -> np.ndarray:
    """A member's own slow, small deviation from the choreography (degrees per angle joint)."""
    out = np.zeros((len(times), len(JOINTS)))
    if degrees <= 0:
        return out
    for j in range(_ANGLE_JOINTS):
        for _ in range(3):
            freq, phase = rng.uniform(0.1, 0.6), rng.uniform(0, 2 * np.pi)
            out[:, j] += degrees / np.sqrt(3) * np.sin(2 * np.pi * freq * times + phase)
    return out


_GLITCH_CHAINS = (([5, 7, 9], [6, 8, 10]), ([11, 13, 15], [12, 14, 16]))


def _add_glitches(kp: np.ndarray, rng: np.random.Generator, rate: float, height: float) -> None:
    """The two pose-model failures seen on real dance footage, in place.

    Half the glitches swap the left and right labels of the arms or the legs for one or two
    frames (typical when a dancer turns side-on). The other half put one wrist up to a third of
    a body height away for one frame. Confidence stays high in both, which is what makes them
    dangerous: nothing upstream filters them out.
    """
    frames = len(kp)
    for t in np.flatnonzero(rng.random(frames) < rate):
        if rng.random() < 0.5:
            left, right = _GLITCH_CHAINS[rng.integers(2)]
            span = slice(t, min(frames, t + rng.integers(1, 3)))
            kp[span, left + right] = kp[span, right + left]
        else:
            wrist = 9 if rng.random() < 0.5 else 10
            kp[t, wrist] += rng.uniform(-1, 1, size=2) * height / 3


def _envelope(times: np.ndarray, t0: float, t1: float, ramp: float) -> np.ndarray:
    if ramp <= 0:
        return ((times >= t0) & (times < t1)).astype(float)
    return np.clip((times - t0) / ramp, 0, 1) * np.clip((t1 - times) / ramp, 0, 1)


def make_group(spec: GroupSpec) -> SyntheticGroup:
    rng = np.random.default_rng(spec.seed)
    m = spec.n_members
    width, height = spec.image_size
    times = np.arange(int(round(spec.duration_s * spec.fps))) / spec.fps
    choreography = make_choreography(spec.duration_s, spec.bpm, rng, loops=spec.loops)
    offsets = spec.offsets_ms if spec.offsets_ms is not None else [0.0] * m
    mirrored = np.array(spec.mirrored if spec.mirrored is not None else [False] * m, dtype=bool)

    heights = 380 * np.linspace(0.82, 1.18, m)
    rng.shuffle(heights)
    bodies = []
    for i in range(m):
        proportions = {k: v * rng.uniform(0.88, 1.12) for k, v in _PROPORTIONS.items()}
        floor_y = height * rng.uniform(0.82, 0.92)
        bodies.append(Body(float(heights[i]), width * (i + 1) / (m + 1), floor_y, proportions))
    if spec.same_body:
        bodies = [Body(bodies[0].height, b.center_x, bodies[0].floor_y, bodies[0].proportions) for b in bodies]

    keypoints = np.empty((m, len(times), 17, 2))
    conf = np.empty((m, len(times), 17))
    offsets_ms = np.empty((m, len(times)))
    for i in range(m):
        offsets_ms[i] = _curve(offsets[i], times)
        local = times - offsets_ms[i] / 1000.0  # a late member is at an earlier point of the choreography
        values = choreography.angles(local) + _style(rng, times, spec.style_deg)
        for error in spec.limb_errors:
            if error.member == i:
                values[:, _J[error.joint]] += error.degrees * _envelope(times, error.t0, error.t1, error.ramp_s)
        yaw = _spin_yaw(local, spec.spins)
        kp = pose_keypoints(values, yaw, bodies[i])
        c = rng.uniform(0.85, 0.95, size=(len(times), 17))
        if mirrored[i]:
            kp = kp[:, FLIP]
            kp[..., 0] = 2 * bodies[i].center_x - kp[..., 0]
        flips = spec.spin_label_flip if isinstance(spec.spin_label_flip, (list, tuple)) else [spec.spin_label_flip] * m
        if flips[i]:
            back = np.cos(yaw) < 0
            kp[back] = kp[back][:, FLIP]
        if spec.noise > 0:
            kp = kp + rng.normal(0.0, spec.noise * bodies[i].height, size=kp.shape)
        if spec.glitches > 0:
            _add_glitches(kp, rng, spec.glitches, bodies[i].height)
        if spec.dropout > 0:
            dropped = rng.random((len(times), 17)) < spec.dropout
            c[dropped] = 0.05
            kp[dropped] += rng.normal(0.0, 0.1 * bodies[i].height, size=(int(dropped.sum()), 2))
        for gap in spec.occlusions:
            if gap.member == i:
                hidden = (times >= gap.t0) & (times < gap.t1)
                kp[hidden] = np.nan
                c[hidden] = 0.0
        for crop in spec.cropped:
            if crop.member == i:
                hidden = np.flatnonzero((times >= crop.t0) & (times < crop.t1))
                kp[np.ix_(hidden, _LOWER_LEGS)] = np.nan
                c[np.ix_(hidden, _LOWER_LEGS)] = 0.0
        keypoints[i], conf[i] = kp, c

    error_bones = []
    for error in spec.limb_errors:
        bone = JOINT_BONE.get(error.joint, error.joint)
        error_bones.append(mirror_name(bone) if mirrored[error.member] else bone)
    return SyntheticGroup(
        fps=spec.fps, times=times, keypoints=keypoints, conf=conf, offsets_ms=offsets_ms,
        mirrored=mirrored, bodies=bodies, limb_error_bones=error_bones,
        names=[f"M{i + 1}" for i in range(m)], spec=spec,
    )

