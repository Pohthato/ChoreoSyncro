"""One re-identification fingerprint per detection: a vector that is close for the same person.

Following members by name needs to tell people apart by how they look, across a whole clip,
in matching outfits and under changing light. A colour histogram of the torso is not enough
for that, so each detection's box is cropped and run through a person re-ID network:
OpenCV Zoo's YouTu ReID (Apache-2.0), with `cv2.dnn`, so no new dependency is needed.

- The weights download once to ~/.cache/syncscore/ and are checked against a pinned SHA-256.
- Only numbers are kept: no pixels are saved.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .tracks import Detections
from .video import read_frames

_REVISION = "https://huggingface.co/opencv/person_reid_youtureid/resolve/b7bb1eda63dd4035eb7fa48d13d31b818877fc1a/"
MODELS = {
    "youtu": {"file": "person_reid_youtu_2021nov.onnx", "url": _REVISION + "person_reid_youtu_2021nov.onnx",
              "size": 106878407, "sha256": "0579683334d4b9440221606dcb461656dd0dc64143b18f48faedaced9b4f580d"},
    "youtu-int8": {"file": "person_reid_youtu_2021nov_int8.onnx", "url": _REVISION + "person_reid_youtu_2021nov_int8.onnx",
                   "size": 26763574, "sha256": "4757c4cb759b79030a9870abf29c064c2ee51e079a05700690800c81b16cf245"},
}
CACHE = Path.home() / ".cache" / "syncscore"
INPUT_SIZE = (128, 256)  # width, height
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet, RGB
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MIN_CROP = 8  # px; a smaller crop has nothing to recognise


def fetch(entry: dict, cache: Path = CACHE, opener=urllib.request.urlopen) -> Path:
    """The model file, downloaded on first use. A download whose SHA-256 differs is deleted."""
    cache = Path(cache)
    path = cache / entry["file"]
    if path.exists() and path.stat().st_size == entry["size"]:
        return path
    cache.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    print(f"Downloading {entry['file']} ({entry['size'] / 1e6:.0f} MB, first run only)", file=sys.stderr, flush=True)
    digest = hashlib.sha256()
    try:
        with opener(entry["url"], timeout=60) as response, open(part, "wb") as out:
            while chunk := response.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    if digest.hexdigest() != entry["sha256"]:
        part.unlink()
        raise ValueError(f"{entry['file']}: the download's SHA-256 is {digest.hexdigest()}, expected {entry['sha256']}")
    part.replace(path)
    return path


def crop(frame: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    """The box's pixels, clipped to the frame; None when under MIN_CROP px on either side."""
    h, w = frame.shape[:2]
    x1, x2 = (int(round(min(max(v, 0), w))) for v in (box[0], box[2]))
    y1, y2 = (int(round(min(max(v, 0), h))) for v in (box[1], box[3]))
    if x2 - x1 < MIN_CROP or y2 - y1 < MIN_CROP:
        return None
    return frame[y1:y2, x1:x2]


class Fingerprinter:
    """BGR crops -> unit-length fingerprints. Callers gather `batch` crops per call, but the network
    sees one crop at a time: run as a batch under cv2.dnn the YouTu graph mixes the crops, and a
    fingerprint's cosine to its one-at-a-time self fell as low as -0.16 on the aespa clip."""

    def __init__(self, net, batch: int = 64):
        self.net = net
        self.batch = batch
        self.dim = 768  # replaced by the network's real output width on first use

    @classmethod
    def load(cls, name: str) -> "Fingerprinter":
        return cls(cv2.dnn.readNet(str(fetch(MODELS[name]))))

    def __call__(self, crops: list[np.ndarray]) -> np.ndarray:
        out = []
        for c in crops:
            blob = cv2.dnn.blobFromImage(c, 1 / 255, INPUT_SIZE, swapRB=True)
            blob = (blob - MEAN[None, :, None, None]) / STD[None, :, None, None]
            self.net.setInput(blob.astype(np.float32))
            v = np.asarray(self.net.forward(), dtype=np.float32).reshape(1, -1)
            norm = np.linalg.norm(v, axis=1, keepdims=True)
            out.append(np.divide(v, norm, out=np.zeros_like(v), where=norm > 0))
            self.dim = v.shape[1]
        return np.vstack(out) if out else np.zeros((0, self.dim), np.float32)


def fingerprint_video(video, det: Detections, fingerprinter, progress: bool = True) -> np.ndarray:
    """(N, D) fingerprints, one per detection row; zeros where the box gives no usable crop."""
    by_frame = defaultdict(list)
    for row, f in enumerate(det.frame):
        by_frame[int(f)].append(row)
    vectors = None
    pending_rows, pending_crops = [], []

    def flush():
        nonlocal vectors
        v = fingerprinter(pending_crops)
        if vectors is None:
            vectors = np.zeros((len(det.frame), v.shape[1]), np.float32)
        vectors[pending_rows] = v
        pending_rows.clear()
        pending_crops.clear()

    cap = cv2.VideoCapture(str(video))
    shown = -50
    try:
        for f, image in read_frames(cap, np.array(sorted(by_frame))):
            for row in by_frame[f]:
                c = crop(image, det.box[row])
                if c is not None:
                    pending_rows.append(row)
                    pending_crops.append(c)
            if len(pending_crops) >= fingerprinter.batch:
                flush()
            if progress and f - shown >= 50:
                shown = f
                print(f"\r  fingerprints: frame {f}/{det.n_frames}", end="", file=sys.stderr, flush=True)
    finally:
        cap.release()
    if pending_crops:
        flush()
    if progress:
        print(file=sys.stderr)
    return vectors if vectors is not None else np.zeros((len(det.frame), fingerprinter.dim), np.float32)


def save_fingerprints(path, vectors: np.ndarray, model: str, key: str) -> None:
    np.savez(path, vectors=vectors.astype(np.float16), model=model, key=key)


def load_fingerprints(path, model: str, key: str) -> np.ndarray | None:
    """The saved fingerprints, or None when missing or made by another model or from other detections."""
    if not Path(path).exists():
        return None
    with np.load(path) as data:
        if str(data["model"]) != model or str(data["key"]) != key:
            return None
        return data["vectors"].astype(np.float32)
