"""Re-detecting only the frames where a member is missing inside a short gap (no model needed here)."""
import dataclasses

import numpy as np

from syncscore.config import load_config
from syncscore.pose.recover import _cache_matches, _load_cache, _save_cache, fill_occlusions, frames_to_recover, merge
from syncscore.pose.tracks import Detections, MemberTracks
from test_tracks import _detections


def _tracks_with_box(seen_frames, n_frames=100):
    box = np.full((1, n_frames, 4), np.nan)
    box[0, seen_frames] = [260, 100, 340, 400]
    rows = np.where(np.isfinite(box[..., 0]), 0, -1)
    return MemberTracks(np.full((1, n_frames, 17, 2), np.nan), np.zeros((1, n_frames, 17)), box,
                        np.zeros((1, n_frames, 8)), [[1]], rows)


def test_frames_to_recover_are_the_unseen_frames_inside_short_gaps():
    seen = [f for f in range(5, 100) if not 40 <= f < 50]  # unseen before 5 (edge) and 40-49 (gap)
    frames = frames_to_recover(_tracks_with_box(seen), fps=30.0, max_gap_s=2.0)
    assert frames.tolist() == list(range(40, 50))


def test_merge_keeps_existing_row_indices():
    det = _detections([(1, range(0, 10), lambda f: 300, 0)], n_frames=20)
    extra = _detections([(-1, [3, 15], lambda f: 600, 1)], n_frames=20)
    merged = merge(det, extra)
    assert len(merged.frame) == 12
    assert np.array_equal(merged.frame[:10], det.frame) and np.array_equal(merged.xy[:10], det.xy)
    assert merged.frame[10:].tolist() == [3, 15] and (merged.track_id[10:] == -1).all()


def test_recover_cache_is_rebuilt_when_settings_change(tmp_path):
    config = load_config()
    cache = tmp_path / "recovered.npz"
    frames = np.array([4, 8, 15])
    _save_cache(cache, _detections([(-1, [4], lambda f: 300, 0)], n_frames=20), frames, config)
    assert _cache_matches(cache, frames, config)
    other_conf = dataclasses.replace(config, tracking=dataclasses.replace(config.tracking, recover_conf=0.1))
    assert not _cache_matches(cache, frames, other_conf)
    other_model = dataclasses.replace(config, pose=dataclasses.replace(config.pose, model="yolo11m-pose.pt"))
    assert not _cache_matches(cache, frames, other_model)
    assert not _cache_matches(cache, np.array([4, 8]), config)
    assert not _cache_matches(tmp_path / "missing.npz", frames, config)


def test_merge_keeps_fingerprints_only_when_both_sides_have_them():
    det = _detections([(1, range(0, 10), lambda f: 300, 0)], n_frames=20)
    extra = _detections([(-1, [3, 15], lambda f: 600, 1)], n_frames=20)
    assert merge(det, extra).vectors is None
    det.vectors = np.ones((10, 4), np.float32)
    assert merge(det, extra).vectors is None
    extra.vectors = np.zeros((2, 4), np.float32)
    merged = merge(det, extra)
    assert merged.vectors.shape == (12, 4) and (merged.vectors[:10] == 1).all() and (merged.vectors[10:] == 0).all()


def test_a_recover_cache_without_fingerprints_is_rebuilt_when_they_are_needed(tmp_path):
    config = load_config()
    cache = tmp_path / "recovered.npz"
    frames = np.array([4, 8])
    plain = _detections([(-1, [4], lambda f: 300, 0)], n_frames=20)
    _save_cache(cache, plain, frames, config)
    assert _cache_matches(cache, frames, config)
    assert not _cache_matches(cache, frames, config, fingerprints=True)
    _save_cache(cache, dataclasses.replace(plain, vectors=np.full((1, 4), 0.5, np.float32)), frames, config)
    assert _cache_matches(cache, frames, config, fingerprints=True) and _cache_matches(cache, frames, config)
    assert np.allclose(_load_cache(cache).vectors, 0.5)
    other_model = "youtu" if config.identity.model != "youtu" else "youtu-int8"
    other = dataclasses.replace(config, identity=dataclasses.replace(config.identity, model=other_model))
    assert not _cache_matches(cache, frames, other, fingerprints=True)


def test_fill_occlusions_asks_the_acceptance_test_in_both_passes():
    from syncscore.pose.tracks import assemble, assign_members, build_tracklets

    det = _detections([
        (1, [f for f in range(100) if not 40 <= f < 44], lambda f: 300, 0),
        (7, range(40, 44), lambda f: 305, 0),   # a leftover for the first pass
        (2, [f for f in range(100) if not 60 <= f < 64], lambda f: 900, 1),
    ])
    tracks = assemble(det, assign_members(det, build_tracklets(det, min_frames=9), members=2, max_gap_s=2.0))
    found = _detections([(-1, range(60, 64), lambda f: 905, 1)], n_frames=100)
    redetect = lambda frames: Detections(found.frame, found.track_id, found.box, found.box_conf, found.xy, found.kconf,
                                         found.hist, found.fps, found.n_frames, found.image_size)
    _, report = fill_occlusions(det, tracks, 2.0, 0.3, redetect=redetect)
    assert report["filled_from_leftovers"] == 4 and report["filled_from_redetection"] == 4
    _, report = fill_occlusions(det, tracks, 2.0, 0.3, redetect=redetect, allow=lambda d, m, r: False)
    assert report["filled_from_leftovers"] == 0 and report["filled_from_redetection"] == 0


def test_redetected_duplicates_do_not_change_the_cause_split():
    # Re-detection finds everyone in the frame again. Counting those duplicates as "detected"
    # would make missing frames look like assignment losses instead of detector misses.
    from syncscore.pose.coverage import coverage
    from syncscore.pose.recover import fill_occlusions
    from syncscore.pose.tracks import assemble, assign_members, build_tracklets

    det = _detections([
        (1, range(100), lambda f: 300, 0),
        (2, [f for f in range(100) if not 40 <= f < 50], lambda f: 900, 1),
        (3, range(100), lambda f: 1500, 2),
    ])
    tracks = assemble(det, assign_members(det, build_tracklets(det, min_frames=9), members=3, max_gap_s=2.0))

    def redetect(frames):
        rows = np.flatnonzero(np.isin(det.frame, frames))
        return Detections(det.frame[rows], np.full(len(rows), -1), det.box[rows], det.box_conf[rows], det.xy[rows],
                          det.kconf[rows], det.hist[rows], det.fps, det.n_frames, det.image_size)

    filled, report = fill_occlusions(det, tracks, max_gap_s=2.0, min_keypoint_conf=0.3, redetect=redetect)
    assert report["filled_from_redetection"] == 0  # the duplicates are the present members
    assert report["after"]["missing_detector_share"] == 1.0
    assert report["after"] == coverage(det, filled, 0.3)


class _VariableRateVideo:
    """Stands in for cv2.VideoCapture on a variable-frame-rate file: frames decode in order, but
    seeking by index converts it to a time with the average frame rate and lands one frame late."""

    def __init__(self, n_frames):
        self.n, self.pos = n_frames, 0

    def set(self, prop, value):
        self.pos = int(value) + 1

    def grab(self):
        self.pos += 1
        return self.pos <= self.n

    def retrieve(self):
        return True, np.full((4, 4, 3), self.pos - 1, dtype=np.uint8)  # pixel value = frame index

    def read(self):
        ok = self.grab()
        return self.retrieve() if ok else (False, None)


def test_frames_are_read_in_decode_order_not_by_seeking():
    from syncscore.pose.video import read_frames

    got = {f: int(image[0, 0, 0]) for f, image in read_frames(_VariableRateVideo(20), np.array([3, 4, 11]))}
    assert got == {3: 3, 4: 4, 11: 11}


class _Array:
    """Stands in for a torch tensor in a YOLO result."""

    def __init__(self, a):
        self.a = np.asarray(a, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def __getitem__(self, i):
        return _Array(self.a[i])

    def __float__(self):
        return float(self.a)

    def __len__(self):
        return len(self.a)


class _FakePose:
    """Two people in every frame: one too small to crop, one 40 px square."""

    def __init__(self, name):
        pass

    def predict(self, image, **kwargs):
        import types

        boxes = _Array([[0, 0, 2, 2], [10, 10, 50, 50]])
        boxes.xyxy, boxes.conf = boxes, _Array([0.5, 0.6])
        keypoints = types.SimpleNamespace(xy=_Array(np.full((2, 17, 2), 20.0)), conf=_Array(np.full((2, 17), 0.9)))
        return [types.SimpleNamespace(boxes=boxes, keypoints=keypoints)]


class _Frames:
    def __init__(self, n):
        self.n, self.pos = n, 0

    def grab(self):
        self.pos += 1
        return self.pos <= self.n

    def retrieve(self):
        return True, np.full((64, 64, 3), self.pos, dtype=np.uint8)

    def release(self):
        pass


def test_redetect_fingerprints_what_it_finds_and_caches_it(tmp_path, monkeypatch):
    import sys
    import types

    import cv2

    from syncscore.pose.recover import redetect

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=_FakePose))
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: _Frames(20))
    calls = []

    def fingerprinter(crops):
        calls.append(len(crops))
        return np.ones((len(crops), 4), np.float32)

    fingerprinter.dim, fingerprinter.batch = 4, 64
    config, cache, frames = load_config(), tmp_path / "recovered.npz", np.array([3, 5])
    det = redetect("clip.mp4", frames, config, cache, (64, 64), 30.0, 20, fingerprinter=fingerprinter)
    assert det.vectors.shape == (4, 4)
    assert (det.vectors[0::2] == 0).all() and (det.vectors[1::2] == 1).all()  # the tiny box gets no fingerprint
    assert sum(calls) == 2
    again = redetect("clip.mp4", frames, config, cache, (64, 64), 30.0, 20, fingerprinter=fingerprinter)
    assert sum(calls) == 2 and np.array_equal(again.vectors, det.vectors)  # from the cache
