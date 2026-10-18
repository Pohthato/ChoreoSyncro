"""Track stitching and swap reporting on hand-built detections (no video or model needed)."""
import numpy as np
import pytest

from syncscore.pose.tracks import (
    Detections, assemble, assign_members, build_tracklets, colour_distance, count_members, score_audit, suspected_swaps,
)

H = 8


def _hist(k):
    h = np.zeros(H)
    h[k] = 1.0
    return h


def _detections(people, n_frames=100, fps=30.0):
    """people: list of (track_id, frames, x_center(frame) callable, colour bin)."""
    rows = []
    for tid, frames, x_of, colour in people:
        for f in frames:
            x = x_of(f)
            rows.append((f, tid, [x - 40, 100, x + 40, 400], 0.9, colour))
    rows.sort()
    n = len(rows)
    xy = np.zeros((n, 17, 2))
    for r, row in enumerate(rows):
        xy[r, :, 0] = (row[2][0] + row[2][2]) / 2
        xy[r, :, 1] = np.linspace(100, 400, 17)
    return Detections(
        frame=np.array([r[0] for r in rows]), track_id=np.array([r[1] for r in rows]),
        box=np.array([r[2] for r in rows], dtype=float), box_conf=np.array([r[3] for r in rows]),
        xy=xy, kconf=np.full((n, 17), 0.9), hist=np.array([_hist(r[4]) for r in rows]),
        fps=fps, n_frames=n_frames, image_size=(1920, 1080),
    )


def test_colour_distance():
    assert colour_distance(_hist(1), _hist(1)) == pytest.approx(0.0)
    assert colour_distance(_hist(1), _hist(2)) == pytest.approx(1.0)


def test_count_members_is_the_most_common_visible_count():
    det = _detections([(1, range(100), lambda f: 300, 0), (2, range(100), lambda f: 900, 1), (3, range(10), lambda f: 1500, 2)])
    assert count_members(det, 0.3) == 2


def test_a_broken_track_is_rejoined_and_a_passer_by_dropped():
    det = _detections([
        (1, range(0, 40), lambda f: 300, 0),      # member A, first piece
        (5, range(50, 100), lambda f: 310, 0),    # member A comes back with a new ID after a gap
        (2, range(0, 100), lambda f: 900, 1),     # member B, never lost
        (9, range(60, 75), lambda f: 1600, 6),    # a crew member walking through
    ])
    slots = assign_members(det, build_tracklets(det, min_frames=5), members=2, max_gap_s=2.0)
    tracks = assemble(det, slots)
    assert tracks.fragments == [[1, 5], [2]]  # left to right: A then B
    assert np.isfinite(tracks.xy[0, 45]).sum() == 0  # the gap stays empty, it is not invented
    assert np.isfinite(tracks.xy[0, 60]).all()


def test_swap_at_a_crossing_is_flagged():
    # A (colour 0) walks right, B (colour 1) walks left; they cross around frame 50 and the
    # tracker swaps their IDs. So after the crossing, "member A" wears B's colour.
    a_x = lambda f: 300 + 12 * f
    b_x = lambda f: 1500 - 12 * f
    det = _detections([
        (1, range(0, 50), a_x, 0), (2, range(0, 50), b_x, 1),
        (1, range(50, 100), b_x, 1), (2, range(50, 100), a_x, 0),
    ])
    slots = assign_members(det, build_tracklets(det, min_frames=5), members=2, max_gap_s=2.0)
    tracks = assemble(det, slots)
    swaps = suspected_swaps(tracks.hist, tracks.box, det.fps, iou_threshold=0.3)
    assert len(swaps) == 1
    assert 1.3 < swaps[0]["time_s"] < 1.8


def test_crossing_without_swap_is_not_flagged():
    a_x = lambda f: 300 + 12 * f
    b_x = lambda f: 1500 - 12 * f
    det = _detections([(1, range(100), a_x, 0), (2, range(100), b_x, 1)])
    tracks = assemble(det, assign_members(det, build_tracklets(det, 5), members=2, max_gap_s=2.0))
    assert suspected_swaps(tracks.hist, tracks.box, det.fps, iou_threshold=0.3) == []


def test_audit_scoring(tmp_path):
    path = tmp_path / "ids.csv"
    path.write_text(
        "frame,shown_label,true_name\n"
        "0,M1,Ann\n300,M1,Ann\n600,M1,Bo\n900,M1,Bo\n"   # M1 switched person once
        "0,M2,Bo\n300,M2,Bo\n600,M2,Ann\n900,M2,\n",     # one row left unlabelled
        encoding="utf-8",
    )
    result = score_audit(path, fps=30.0)
    assert result["labelled"] == 7
    assert result["switches"] == 2
    # M1 is Ann twice and Bo twice (2 match its majority name), M2 is Bo twice and Ann once (2 match).
    assert result["accuracy"] == pytest.approx(4 / 7)


def test_a_long_track_with_changed_colour_is_kept_when_a_slot_is_free():
    # Member A walks into different light: same position, very different torso colour, for 2 s.
    # In a dance practice nobody is a passer-by, so it must not be thrown away.
    det = _detections([
        (1, range(0, 40), lambda f: 300, 0),
        (5, range(45, 105), lambda f: 300, 7),    # 60 frames = 2 s, colour 7 matches no one
        (2, range(0, 105), lambda f: 900, 1),
    ], n_frames=105)
    tracks = assemble(det, assign_members(det, build_tracklets(det, min_frames=5), members=2, max_gap_s=2.0))
    assert tracks.fragments == [[1, 5], [2]]

