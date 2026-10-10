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


@dataclass
class SmoothingConfig:
    min_cutoff: float = 1.0
    beta: float = 0.05
    d_cutoff: float = 1.0
    max_gap_frames: int = 4


@dataclass
class FeaturesConfig:
    mirror: Any = "off"  # "off" | "auto" | list of member ids


@dataclass
class SyncConfig:
    window_s: float = 2.0
    hop_s: float = 0.25
    max_lag_ms: float = 400.0
    min_motion: float = 0.15
    min_peak: float = 0.5


@dataclass
class SectionsConfig:
    explicit: list = field(default_factory=list)
    chunk_s: float = 8.0


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
    render: RenderConfig = field(default_factory=RenderConfig)


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
