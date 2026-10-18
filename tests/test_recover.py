"""Re-detecting only the frames where a member is missing inside a short gap (no model needed here)."""
import dataclasses

import numpy as np

from syncscore.config import load_config
from syncscore.pose.recover import _cache_matches, _save_cache, frames_to_recover, merge
from syncscore.pose.tracks import MemberTracks
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
