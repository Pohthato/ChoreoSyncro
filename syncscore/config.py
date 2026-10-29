"""Typed configuration loaded from YAML.

Each section of configs/default.yaml maps to one dataclass, so a typo in a key fails loudly
instead of silently using a default.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


@dataclass
class PoseConfig:
    model: str = "yolo11s-pose.pt"
    device: str = "cpu"
    min_keypoint_conf: float = 0.3
    min_box_conf: float = 0.25


@dataclass
class TrackingConfig:
    tracker: str = "bytetrack.yaml"
    members: int = 0
    stitch_max_gap_s: float = 2.0
    crossing_iou: float = 0.3
    fill_max_gap_s: float = 2.0  # fill member gaps up to this long with real detections (0 = off)
    recover: bool = True        # re-detect frames inside such gaps at a lower threshold
    recover_conf: float = 0.05
    recover_imgsz: int = 1280


@dataclass
class SmoothingConfig:
    min_cutoff: float = 1.0
    beta: float = 1.0
    d_cutoff: float = 1.0
    max_gap_frames: int = 4
    repair_swaps: bool = True


@dataclass
class FeaturesConfig:
    mirror: Any = "off"  # "off" | "auto" | list of member ids


@dataclass
class SyncConfig:
    window_s: float = 2.0
    hop_s: float = 0.25
    max_lag_ms: float = 400.0
    min_motion: float = 0.3
    min_peak: float = 0.5
    min_coverage: float = 0.5
    min_margin: float = 0.15
    lag_penalty: float = 0.1
    iterations: int = 10
    shape_threshold_deg: float = 15.0
    max_bone_speed_deg_s: float = 1800.0


@dataclass
class SectionsConfig:
    explicit: list = field(default_factory=list)
    chunk_s: float = 32.0
    min_peak: float = 0.2      # looser than sync.min_peak: windows are pooled, not trusted alone
    min_windows: int = 8
    min_blocks: int = 5        # independent stretches; fewer makes the bootstrap interval meaningless
    max_ci_ms: float = 50.0


@dataclass
class UnisonConfig:
    max_spread_deg: float = 25.0    # two members dance the same thing: limb angle between them (at least; 2x the clip's typical)
    max_own_deg: float = 30.0       # this member dances what they dance: limb angle to their median (at least; 2x the clip's typical)
    limit_scale: float = 2.0        # both limits are at least this x the clip's typical difference (0: the fixed limits above)
    moving_speed: float = 0.3       # "moving" = average speed over smooth_s above this x the member's usual moving speed
    min_moving_share: float = 0.5   # the others are moving when at least this share of them is
    smooth_s: float = 1.0
    own_smooth_s: float = 2.0
    min_others: int = 2


@dataclass
class BeatConfig:
    landing_peak: float = 2.5       # a clean landing's move peaks at least this x the member's usual moving speed
    landing_hold: float = 0.5       # ...and falls to this or less
    landing_within_s: float = 0.2   # ...within this long
    landing_level: float = 0.25     # landing time: speed fallen this close to the hold (0.25 = three quarters down)


@dataclass
class HitsConfig:
    prominence: float = 1.5      # a group hit rises this far above its surroundings (x a member's typical speed)
    max_offset_ms: float = 400.0 # search range for the member's matching peak
    max_second_peak: float = 0.7 # another peak at least this high (relative) in range = a tie...
    min_agreement: float = 0.5   # ...won only by a peak whose direction agrees this well (cosine) with the others'
    min_others: int = 3          # other members with a matched peak needed for a reference
    min_hits: int = 5            # fewer matched hits in a section = no reading
    max_ci_ms: float = 50.0      # a 90% interval wider than this = no reading


@dataclass
class IdentityConfig:
    model: str = "youtu"         # re-ID model, a key of syncscore.pose.fingerprint.MODELS
    name_margin: float = 0.05    # a piece is named only when its member beats the next best class by this much
    learn_margin: float = 0.10   # ...and learned from (its rows become examples) only by this much
    rounds: int = 3              # learning rounds
    split_distance: float = 0.35 # cosine distance between a tracklet's fingerprints before and after a crossing that splits it
    position_gate: float = 3.0   # a piece further than this (Mahalanobis) from where the member's motion predicts is refused
    free_after_s: float = 3.0    # after this long unseen, a member's position no longer limits where she can come back


@dataclass
class RenderConfig:
    in_sync_ms: float = 35.0
    off_sync_ms: float = 135.0


@dataclass
class Config:
    pose: PoseConfig = field(default_factory=PoseConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    sections: SectionsConfig = field(default_factory=SectionsConfig)
    hits: HitsConfig = field(default_factory=HitsConfig)
    unison: UnisonConfig = field(default_factory=UnisonConfig)
    beat: BeatConfig = field(default_factory=BeatConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)


def _build(cls: type, data: dict[str, Any]) -> Any:
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"Unknown config keys in {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        default = known[name].default_factory() if callable(known[name].default_factory) else None
        kwargs[name] = _build(type(default), value or {}) if is_dataclass(default) else value
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> Config:
    """Load a YAML config. With no path, the packaged default is used."""
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as handle:
        return _build(Config, yaml.safe_load(handle) or {})
