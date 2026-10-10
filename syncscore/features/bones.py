"""COCO-17 keypoints -> bone direction unit vectors.

Raw pixel positions can't be compared across members: a tall member standing near the camera
and a short one at the back produce very different coordinates for the same move. A bone's
*direction* (the unit vector from one joint to the next) has no position, no scale and no
length, so it only depends on how the joints are bent. Two bodies doing the same move give the
same directions whatever their proportions.

Arrays follow one convention throughout the package:
    xy   (..., T, 17, 2) pixel coordinates, NaN where unknown
    conf (..., T, 17)    keypoint confidence, 0 where unknown
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
# Two virtual points appended after the 17 keypoints.
NECK = 17    # midpoint of the shoulders
PELVIS = 18  # midpoint of the hips

# FLIP[i] is the keypoint that plays i's role on the other side of the body.
FLIP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])


@dataclass(frozen=True)
class Bone:
    name: str
    start: int
    end: int
    part: str


BONES = (
    Bone("left_upper_arm", 5, 7, "arms"),
    Bone("left_forearm", 7, 9, "arms"),
    Bone("right_upper_arm", 6, 8, "arms"),
    Bone("right_forearm", 8, 10, "arms"),
    Bone("left_thigh", 11, 13, "legs"),
    Bone("left_shin", 13, 15, "legs"),
    Bone("right_thigh", 12, 14, "legs"),
    Bone("right_shin", 14, 16, "legs"),
    Bone("spine", PELVIS, NECK, "torso"),
    Bone("shoulders", 6, 5, "torso"),
    Bone("hips", 12, 11, "torso"),
    Bone("head", NECK, 0, "head"),
    Bone("ears", 4, 3, "head"),
)
PARTS = ("arms", "legs", "torso", "head")
BONE_NAMES = tuple(bone.name for bone in BONES)
PART_OF_BONE = np.array([PARTS.index(bone.part) for bone in BONES])
_STARTS = np.array([bone.start for bone in BONES])
_ENDS = np.array([bone.end for bone in BONES])


def mirror_name(name: str) -> str:
    """The same bone on the other side of the body ('left_shin' <-> 'right_shin')."""
    if name.startswith("left_"):
        return "right_" + name[5:]
    if name.startswith("right_"):
        return "left_" + name[6:]
    return name


def mirror_keypoints(xy: np.ndarray, conf: np.ndarray | None = None):
    """Mirror a pose: swap left and right keypoints and reflect x.

    A member dancing the mirrored version of the choreography (their left arm doing what the
    others do with their right) matches the others after this. The reflection is about x = 0,
    which is fine for directions but means the result is not meant to be drawn.
    """
    out = xy[..., FLIP, :].copy()
    out[..., 0] *= -1.0
    return out if conf is None else (out, conf[..., FLIP])


@dataclass
class BoneFeatures:
    u: np.ndarray       # (..., T, B, 2) unit directions, NaN where unknown
    weight: np.ndarray  # (..., T, B) trust in [0, 1], 0 where unknown
    length: np.ndarray  # (..., T, B) 2D length in pixels, NaN where unknown


def _with_virtual_points(xy: np.ndarray, conf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    neck = (xy[..., 5, :] + xy[..., 6, :]) / 2
    pelvis = (xy[..., 11, :] + xy[..., 12, :]) / 2
    points = np.concatenate([xy, neck[..., None, :], pelvis[..., None, :]], axis=-2)
    neck_conf = np.minimum(conf[..., 5], conf[..., 6])
    pelvis_conf = np.minimum(conf[..., 11], conf[..., 12])
    scores = np.concatenate([conf, neck_conf[..., None], pelvis_conf[..., None]], axis=-1)
    return points, scores


def foreshortening_weight(length: np.ndarray, full: float = 0.5) -> np.ndarray:
    """How far to trust a bone's 2D direction given how short it currently looks.

    When a limb points at the camera its 2D length shrinks and its direction becomes noise.
    Each bone is compared with its own typical length for this member (90th percentile over
    time), so body proportions do not matter: weight 1 when at least `full` of the typical
    length is visible, falling linearly to 0.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN bones
        typical = np.nanpercentile(length, 90, axis=-2, keepdims=True)
    ratio = np.nan_to_num(length / typical, nan=0.0)
    return np.clip(ratio / full, 0.0, 1.0)


def bone_directions(xy: np.ndarray, conf: np.ndarray, min_conf: float = 0.3, foreshortening: bool = True) -> BoneFeatures:
    """Unit direction of every bone, with a weight saying how much to trust it."""
    points, scores = _with_virtual_points(xy, conf)
    vectors = points[..., _ENDS, :] - points[..., _STARTS, :]
    length = np.linalg.norm(vectors, axis=-1)
    bone_conf = np.minimum(scores[..., _STARTS], scores[..., _ENDS])
    valid = (bone_conf >= min_conf) & np.isfinite(length) & (length > 1e-6)
    safe = np.where(valid, length, 1.0)
    u = np.where(valid[..., None], vectors / safe[..., None], np.nan)
    length = np.where(valid, length, np.nan)
    weight = np.where(valid, bone_conf, 0.0)
    if foreshortening:
        weight = weight * foreshortening_weight(length)
    return BoneFeatures(u=u, weight=weight, length=length)


def angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unsigned angle in degrees between unit vectors along the last axis (NaN if unknown)."""
    dot = np.sum(a * b, axis=-1)
    cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return np.degrees(np.arctan2(np.abs(cross), dot))
