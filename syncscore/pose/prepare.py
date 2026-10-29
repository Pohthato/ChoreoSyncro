"""What following members by name needs, worked out once per set of detections.

- `camera.npz`: the camera path (camera.py).
- `fingerprints.npz`: one re-ID fingerprint per detection (fingerprint.py).
- `pick_frames.json`: what the picking page needs: the clip's size and length, the frames
  suggested for picking, and each frame's real time, which the page seeks to.

The first two are reused only while their stored key matches the detections (and the re-ID
model matches); new detections recompute both.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import Config
from .camera import CameraPath, estimate_camera
from .fingerprint import Fingerprinter, fingerprint_video, load_fingerprints, save_fingerprints
from .picks import suggest_frames
from .tracks import Detections, detections_key
from .video import VideoInfo, probe


def prepare(video, out: Path, det: Detections, config: Config, progress: bool = True,
            info: VideoInfo | None = None) -> tuple[CameraPath, np.ndarray]:
    """(camera path, fingerprints), from `out` when they match `det`, else computed and saved there."""
    out = Path(out)
    key = detections_key(det)
    camera = CameraPath.load(out / "camera.npz", key)
    if camera is None:
        camera = estimate_camera(video, det, progress)
        camera.save(out / "camera.npz", key)
    model = config.identity.model
    vectors = load_fingerprints(out / "fingerprints.npz", model, key)
    if vectors is None:
        vectors = fingerprint_video(video, det, Fingerprinter.load(model), progress)
        save_fingerprints(out / "fingerprints.npz", vectors, model, key)
    info = info or probe(video)
    pick_frames = {"n_frames": det.n_frames, "fps": det.fps, "size": [int(v) for v in det.image_size],
                   "suggested": suggest_frames(det, config.pose.min_keypoint_conf),
                   "times_ms": [round(float(t), 3) for t in info.times_ms]}
    (out / "pick_frames.json").write_text(json.dumps(pick_frames), encoding="utf-8")
    return camera, vectors
