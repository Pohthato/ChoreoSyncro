import hashlib
import io

import cv2
import numpy as np
import pytest

from syncscore.pose.fingerprint import Fingerprinter, crop, fetch, fingerprint_video, load_fingerprints, save_fingerprints
from syncscore.pose.tracks import Detections

class MeanNet:  # stands in for cv2.dnn: the "fingerprint" is the input blob's mean per channel
    def setInput(self, blob):
        self.blob = blob

    def forward(self):
        return self.blob.mean(axis=(2, 3))


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def test_fetch_downloads_once_and_checks_the_hash(tmp_path):
    payload = b"model bytes"
    entry = {"file": "m.onnx", "url": "https://example.invalid/m.onnx", "size": len(payload),
             "sha256": hashlib.sha256(payload).hexdigest()}
    calls = []

    def opener(url, timeout=60):
        calls.append(url)
        return io.BytesIO(payload)

    path = fetch(entry, tmp_path, opener)
    assert path.read_bytes() == payload
    assert fetch(entry, tmp_path, opener) == path and len(calls) == 1


def test_a_download_with_the_wrong_hash_is_refused_and_not_kept(tmp_path):
    entry = {"file": "m.onnx", "url": "https://example.invalid/m.onnx", "size": 8, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="SHA-256"):
        fetch(entry, tmp_path, lambda url, timeout=60: io.BytesIO(b"tampered"))
    assert list(tmp_path.iterdir()) == []


def test_crops_are_rgb_imagenet_normalised_and_vectors_unit_length():
    red = np.zeros((40, 20, 3), np.uint8)
    red[..., 2] = 255  # BGR red
    v = Fingerprinter(MeanNet())([red])
    expected = np.array([(1 - 0.485) / 0.229, -0.456 / 0.224, -0.406 / 0.225])
    assert np.allclose(v[0], expected / np.linalg.norm(expected), atol=1e-3)


def test_batches_give_the_same_vectors_as_one_at_a_time():
    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 256, (int(rng.integers(20, 90)), int(rng.integers(10, 50)), 3), dtype=np.uint8) for _ in range(5)]
    together = Fingerprinter(MeanNet(), batch=2)(crops)
    alone = np.vstack([Fingerprinter(MeanNet())([c]) for c in crops])
    assert together.shape == (5, 3) and together.dtype == np.float32
    assert np.allclose(together, alone, atol=1e-6)


def test_crop_is_clipped_to_the_frame_and_none_when_tiny():
    frame = np.zeros((100, 200, 3), np.uint8)
    assert crop(frame, np.array([150.0, 10, 260, 90])).shape == (80, 50, 3)
    assert crop(frame, np.array([10.0, 10, 15, 90])) is None  # under 8 px wide


def _two_colour_video(path, n_frames=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    image = np.zeros((120, 160, 3), np.uint8)
    image[:, :80] = (255, 0, 0)  # BGR blue
    image[:, 80:] = (0, 0, 255)  # BGR red
    for _ in range(n_frames):
        writer.write(image)
    writer.release()


def _detections(boxes, frames, n_frames):
    n = len(frames)
    return Detections(frame=np.array(frames), track_id=np.full(n, -1), box=np.array(boxes, float),
                      box_conf=np.ones(n), xy=np.zeros((n, 17, 2)), kconf=np.zeros((n, 17)), hist=np.zeros((n, 4)),
                      fps=10.0, n_frames=n_frames, image_size=(160, 120))


def test_fingerprint_video_gives_one_vector_per_detection_row(tmp_path):
    video = tmp_path / "two.mp4"
    _two_colour_video(video)
    boxes, frames, kind = [], [], []
    for f in range(10):
        boxes += [[5, 5, 75, 115], [85, 5, 155, 115]]
        frames += [f, f]
        kind += ["blue", "red"]
        if f == 3:
            boxes.append([20, 20, 24, 24])
            frames.append(f)
            kind.append("tiny")
    vectors = fingerprint_video(video, _detections(boxes, frames, 10), Fingerprinter(MeanNet()), progress=False)
    blue = _unit(  # RGB order: blue is the third channel
np.array([-0.485 / 0.229, -0.456 / 0.224, (1 - 0.406) / 0.225]))
    red = _unit(np.array([(1 - 0.485) / 0.229, -0.456 / 0.224, -0.406 / 0.225]))
    assert vectors.shape == (len(frames), 3) and vectors.dtype == np.float32
    for row, k in enumerate(kind):
        if k == "tiny":
            assert not vectors[row].any()
        else:
            assert np.allclose(vectors[row], blue if k == "blue" else red, atol=0.05)


def test_saved_fingerprints_load_only_for_the_same_detections_and_model(tmp_path):
    path = tmp_path / "fingerprints.npz"
    vectors = np.random.default_rng(1).normal(size=(6, 8)).astype(np.float32)
    save_fingerprints(path, vectors, "youtu", "abc")
    loaded = load_fingerprints(path, "youtu", "abc")
    assert loaded.dtype == np.float32 and np.allclose(loaded, vectors, atol=1e-3)
    assert load_fingerprints(path, "youtu-int8", "abc") is None
    assert load_fingerprints(path, "youtu", "other") is None
    assert load_fingerprints(tmp_path / "missing.npz", "youtu", "abc") is None
