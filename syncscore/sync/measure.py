"""Keypoints of a whole group -> timing offsets and shape errors for every member.

This is the method end to end. Video and synthetic data both enter here, so the synthetic
benchmark tests exactly the code that runs on real footage.

1. Every member's keypoints become bone directions (and their mirror images).
2. Iterate (up to config `sync.iterations`, stopping once nothing moves by 0.1 ms):
   a. move every member onto the group's clock using their current offset estimate,
   b. build each member's leave-one-out consensus from the others (median direction),
   c. cross-correlate the member's bone velocities against the consensus in sliding windows
      to get their lag per window,
   d. re-centre so the median member is at 0 ms. "Group timing" is the median member,
      then move each estimate halfway toward the new value (damping, see below),
   e. with `mirror: auto`, decide per window whether the member is closer to the consensus
      as they are or mirrored.
   The first pass compares everyone against an unaligned consensus, which blurs the
   reference; later passes compare against a consensus whose members are already lined up,
   which is sharper.
3. Per body part: the same lag search on just that part's bones.
4. Shape: move the whole group's median pose onto the member's own clock and measure the
   angle per bone (see consensus.py for why shape includes the member and timing does not).
5. Hits (hits.py): each member's speed peak at the group's big simultaneous movements. Scored
   by the planted-shift test only; the report doesn't use them (they added no reliable
   readings on real footage).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from ..features.bones import BONE_NAMES, PART_OF_BONE, PARTS, bone_directions, mirror_keypoints
from .consensus import group_median, leave_one_out_median, warp
from .hits import hit_timing
from .shape import shape_errors
from .timing import Windows, estimate_lags, make_windows, velocity

FACING_DEG = 120.0   # shoulder and hip lines both reversed this much = facing a different way
_SHOULDERS, _HIPS = BONE_NAMES.index("shoulders"), BONE_NAMES.index("hips")
DAMPING = 0.5        # share of the way to move toward each new estimate
CONVERGED_MS = 0.1   # stop iterating once no reliable window moves more than this


@dataclass
class SyncResult:
    fps: float
    names: list
    n_frames: int
    window_centers_s: np.ndarray   # (K,)
    window_s: float
    offset_ms: np.ndarray          # (M, K) + = behind the group, - = ahead
    peak: np.ndarray               # (M, K) correlation at the chosen lag
    motion: np.ndarray             # (M, K) how much the group was moving, rad/s
    reliable: np.ndarray           # (M, K) bool
    coverage: np.ndarray           # (M, K) share of the window the member was seen in
    margin: np.ndarray             # (M, K) chosen correlation peak minus the best separate one
    part_offset_ms: np.ndarray     # (M, P, K)
    part_reliable: np.ndarray      # (M, P, K)
    mirrored: np.ndarray           # (M, K) bool, member compared as mirrored in that window
    offset_curve_ms: np.ndarray    # (M, T) per-frame offset, interpolated between reliable windows
    bone_shape_deg: np.ndarray     # (M, T, B) on the member's own (video) clock
    part_shape_deg: np.ndarray     # (M, T, P)
    facing_differently: np.ndarray # (M, T) shoulders and hips reversed vs the group: pose not comparable
    hit_times_s: np.ndarray        # (H,) the group's hits (see hits.py)
    hit_offset_ms: np.ndarray      # (M, H) member's peak minus the others' median peak; NaN = no clean match
    hit_agreement: np.ndarray      # (M, H) how well the member's movement matched the group's direction there
    bone_names: tuple = BONE_NAMES
    part_names: tuple = PARTS
    notes: list = field(default_factory=list)

    def save(self, path) -> None:
        arrays = {k: v for k, v in self.__dict__.items() if isinstance(v, np.ndarray)}
        np.savez_compressed(path, names=np.array(self.names), fps=self.fps, n_frames=self.n_frames, window_s=self.window_s, **arrays)


def _mirror_flags(setting, members: int) -> tuple[np.ndarray, bool]:
    """Config `features.mirror` -> (fixed per-member flags, auto?). Lists use 1-based member numbers."""
    if setting in (None, False, "off"):
        return np.zeros(members, dtype=bool), False
    if setting == "auto":
        return np.zeros(members, dtype=bool), True
    flags = np.zeros(members, dtype=bool)
    for number in setting:
        flags[int(number) - 1] = True
    return flags, False


def _curve(values: np.ndarray, ok: np.ndarray, centers: np.ndarray, n_frames: int) -> np.ndarray:
    """Per-frame curve through the reliable windows (flat beyond the ends, 0 if none)."""
    if not ok.any():
        return np.zeros(n_frames)
    return np.interp(np.arange(n_frames), centers[ok], values[ok])


def _per_frame(flags: np.ndarray, windows: Windows, n_frames: int) -> np.ndarray:
    """Window flags -> per frame, from the nearest window centre."""
    nearest = np.abs(np.arange(n_frames)[:, None] - windows.centers[None, :]).argmin(axis=1)
    return flags[nearest]


def _majority(flags: np.ndarray, span: int) -> np.ndarray:
    if span <= 1:
        return flags
    kernel = np.ones(span) / span
    return np.convolve(flags.astype(float), kernel, mode="same") > 0.5


def measure_sync(xy: np.ndarray, conf: np.ndarray, fps: float, config: Config, names: list | None = None) -> SyncResult:
    """xy (M, T, 17, 2) pixels, conf (M, T, 17) -> SyncResult."""
    members, n_frames = xy.shape[:2]
    names = names or [f"M{i + 1}" for i in range(members)]
    sc = config.sync
    min_conf = config.pose.min_keypoint_conf

    plain = [bone_directions(xy[i], conf[i], min_conf) for i in range(members)]
    flipped = [bone_directions(*mirror_keypoints(xy[i], conf[i]), min_conf) for i in range(members)]
    vel_plain = [velocity(f.u, f.weight, fps, sc.max_bone_speed_deg_s) for f in plain]
    vel_flipped = [velocity(f.u, f.weight, fps, sc.max_bone_speed_deg_s) for f in flipped]

    max_lag = max(1, int(round(sc.max_lag_ms / 1000 * fps)))
    windows = make_windows(n_frames, int(round(sc.window_s * fps)), int(round(sc.hop_s * fps)))
    centers, n_windows = windows.centers, len(windows.starts)
    frame_ms = 1000.0 / fps
    penalty = sc.lag_penalty * frame_ms / 100.0  # config is per 100 ms of jump
    search = dict(max_lag=max_lag, penalty_per_step=penalty, min_motion=sc.min_motion, min_peak=sc.min_peak, min_coverage=sc.min_coverage, min_margin=sc.min_margin)

    fixed, auto = _mirror_flags(config.features.mirror, members)
    mirrored = np.repeat(fixed[:, None], n_windows, axis=1)
    lag = np.zeros((members, n_windows))
    reliable = np.zeros((members, n_windows), dtype=bool)
    peak = np.full((members, n_windows), np.nan)
    motion = np.zeros((members, n_windows))
    coverage = np.zeros((members, n_windows))
    margin = np.full((members, n_windows), np.nan)

    def chosen(i):
        """Member i's directions and velocities in the orientation chosen for each frame."""
        flip = _per_frame(mirrored[i], windows, n_frames)
        u = np.where(flip[:, None, None], flipped[i].u, plain[i].u)
        w = np.where(flip[:, None], flipped[i].weight, plain[i].weight)
        v = np.where(flip[:, None, None], vel_flipped[i][0], vel_plain[i][0])
        vw = np.where(flip[:, None], vel_flipped[i][1], vel_plain[i][1])
        return u, w, v, vw

    def consensus_now():
        curves = np.stack([_curve(lag[i], reliable[i], centers, n_frames) for i in range(members)])
        aligned = [warp(*chosen(i)[:2], curves[i]) for i in range(members)]
        stacked_u, stacked_w = np.stack([a[0] for a in aligned]), np.stack([a[1] for a in aligned])
        c, cw = leave_one_out_median(stacked_u, stacked_w)
        consensus_now.aligned = (stacked_u, stacked_w)
        return curves, c, cw

    for iteration in range(max(1, sc.iterations)):
        curves, cons, cons_w = consensus_now()
        new = np.full_like(lag, np.nan)
        was_reliable = reliable.copy()
        for i in range(members):
            _, _, v, vw = chosen(i)
            cv, cvw = velocity(cons[i], cons_w[i], fps, sc.max_bone_speed_deg_s)
            result = estimate_lags(v, vw, cv, cvw, windows, **search)
            new[i], reliable[i], peak[i], motion[i] = result.lag, result.reliable, result.peak, result.motion
            coverage[i], margin[i] = result.coverage, result.margin
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # windows where nobody is reliable
            center = np.nanmedian(np.where(reliable, new, np.nan), axis=0)
        new -= np.nan_to_num(center)[None, :]
        # Damped update. Each member is measured against the others' *current* estimates, so
        # undamped passes overshoot each other and flip sign every pass (with two members,
        # forever). Moving halfway removes the overshoot and converges in a few passes.
        damp = (iteration > 0) & was_reliable & np.isfinite(lag) & np.isfinite(new)
        previous = lag.copy()
        lag = np.where(damp, previous + DAMPING * (new - previous), new)
        change = np.abs(lag - previous)[damp & reliable]
        if iteration > 0 and (change.size == 0 or change.max() * frame_ms < CONVERGED_MS):
            break
        if auto:
            curves, cons, cons_w = consensus_now()
            for i in range(members):
                errs = []
                for feats in (plain[i], flipped[i]):
                    au, aw = warp(feats.u, feats.weight, curves[i])
                    bone, _ = shape_errors(au, aw, cons[i], cons_w[i])
                    errs.append(np.array([np.nanmean(bone[s:s + windows.length]) if np.isfinite(bone[s:s + windows.length]).any() else np.nan for s in windows.starts]))
                better = np.nan_to_num(errs[1], nan=np.inf) < np.nan_to_num(errs[0], nan=np.inf)
                mirrored[i] = _majority(better, span=5)

    curves, cons, cons_w = consensus_now()
    shape_ref, shape_ref_w = group_median(*consensus_now.aligned)  # whole group: a vote, see consensus.py
    part_lag = np.full((members, len(PARTS), n_windows), np.nan)
    part_ok = np.zeros((members, len(PARTS), n_windows), dtype=bool)
    bone_deg = np.full((members, n_frames, len(BONE_NAMES)), np.nan)
    facing = np.zeros((members, n_frames), dtype=bool)
    part_deg = np.full((members, n_frames, len(PARTS)), np.nan)
    for i in range(members):
        u, w, v, vw = chosen(i)
        cv, cvw = velocity(cons[i], cons_w[i], fps, sc.max_bone_speed_deg_s)
        for p in range(len(PARTS)):
            bones = np.flatnonzero(PART_OF_BONE == p)
            result = estimate_lags(v, vw, cv, cvw, windows, bones=bones, **search)
            part_lag[i, p] = result.lag  # the consensus is on the group clock, so this is the part's own offset
            part_ok[i, p] = result.reliable
        own_u, own_w = warp(shape_ref, shape_ref_w, -curves[i])  # the group's pose on the member's clock
        bone_deg[i], part_deg[i] = shape_errors(u, w, own_u, own_w)
        # If both the shoulder line and the hip line point the opposite way to the group's, the
        # member faces a different way (or the pose model swapped left and right). In 2D that
        # pose can't be compared with the group's: it would read as 180-degree "errors".
        facing[i] = (np.nan_to_num(bone_deg[i, :, _SHOULDERS]) > FACING_DEG) & (np.nan_to_num(bone_deg[i, :, _HIPS]) > FACING_DEG)
        bone_deg[i, facing[i]] = np.nan
        part_deg[i, facing[i]] = np.nan
    # Hits: every member's own peaks on the video clock, so this needs no alignment.
    moving = [chosen(i)[2:] for i in range(members)]
    hits = hit_timing([v for v, _ in moving], [vw for _, vw in moving], fps, config.hits)

    return SyncResult(
        fps=fps, names=list(names), n_frames=n_frames, window_centers_s=centers / fps, window_s=sc.window_s,
        offset_ms=np.where(np.isfinite(lag), lag * frame_ms, np.nan), peak=peak, motion=motion, reliable=reliable,
        coverage=coverage, margin=margin,
        part_offset_ms=part_lag * frame_ms, part_reliable=part_ok, mirrored=mirrored,
        offset_curve_ms=curves * frame_ms, bone_shape_deg=bone_deg, part_shape_deg=part_deg,
        facing_differently=facing,
        hit_times_s=hits.frames / fps, hit_offset_ms=hits.offset_ms, hit_agreement=hits.agreement,
    )

