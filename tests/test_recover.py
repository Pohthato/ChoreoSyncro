"""Re-detecting only the frames where a member is missing inside a short gap (no model needed here)."""
import dataclasses

import numpy as np

from syncscore.config import load_config
from syncscore.pose.recover import _cache_matches, _save_cache, frames_to_recover, merge
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
    from syncscore.pose.recover import _read_frames

    got = {f: int(image[0, 0, 0]) for f, image in _read_frames(_VariableRateVideo(20), np.array([3, 4, 11]))}
    assert got == {3: 3, 4: 4, 11: 11}
