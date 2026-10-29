import json

import numpy as np
import pytest

from syncscore.pose.picks import (
    PickError, dump_picks, load_picks, parse_picks, people_in_frame, resolve_picks, suggest_frames,
)
from syncscore.pose.tracks import Detections


def _frame_people(people, n_frames=10, fps=30.0):
    """people: (frame, box, clear) tuples -> Detections. A clear person has all 17 keypoints confident."""
    n = len(people)
    kconf = np.array([[0.9] * 17 if clear else [0.9] * 5 + [0.1] * 12 for _, _, clear in people]).reshape(n, 17)
    return Detections(frame=np.array([f for f, _, _ in people], int), track_id=np.arange(n),
                      box=np.array([b for _, b, _ in people], float).reshape(n, 4), box_conf=np.ones(n),
                      xy=np.zeros((n, 17, 2)), kconf=kconf, hist=np.zeros((n, 4)), fps=fps, n_frames=n_frames,
                      image_size=(1280, 720))


THREE = [(5, [500, 100, 600, 400], True), (5, [100, 100, 200, 400], True), (5, [300, 100, 400, 400], False),
         (6, [100, 100, 200, 400], True)]


def _picks(*named):
    return {"frame": 5, "members": [{"name": n, "box": b} for n, b in named]}


def test_picks_are_parsed_with_names_trimmed_and_kept_verbatim():
    picks = parse_picks({"frame": 5, "members": [{"name": "  Ning-ning 宁宁 ", "box": [1, 2, 3, 4]},
                                                 {"name": "<b>Winter</b>", "box": [5, 6, 7, 8]}]}, n_frames=10)
    assert picks.frame == 5
    assert [p.name for p in picks.members] == ["Ning-ning 宁宁", "<b>Winter</b>"]
    assert np.array_equal(picks.members[1].box, [5, 6, 7, 8])


@pytest.mark.parametrize("data, message", [
    ({"frame": 5, "members": [{"name": "A", "box": [1, 2, 3, 4]}]}, "at least 2"),
    ({"frame": 5, "members": [{"name": "A", "box": [1, 2, 3, 4]}, {"name": " ", "box": [1, 2, 3, 4]}]}, "Type a name"),
    ({"frame": 5, "members": [{"name": "Karina", "box": [1, 2, 3, 4]}, {"name": "karina", "box": [5, 6, 7, 8]}]}, "already called"),
    ({"frame": 5, "members": [{"name": "A" * 41, "box": [1, 2, 3, 4]}, {"name": "B", "box": [1, 2, 3, 4]}]}, "40 characters"),
    ({"frame": 10, "members": [{"name": "A", "box": [1, 2, 3, 4]}, {"name": "B", "box": [1, 2, 3, 4]}]}, "Frame .* is not in this video"),
    ({"frame": 5, "members": [{"name": "A", "box": [1, 2, 3]}, {"name": "B", "box": [1, 2, 3, 4]}]}, "box"),
    ({"frame": 5, "members": [{"name": "A", "box": [1, 2, "x", 4]}, {"name": "B", "box": [1, 2, 3, 4]}]}, "box"),
    ({"frame": "5", "members": [{"name": "A", "box": [1, 2, 3, 4]}, {"name": "B", "box": [1, 2, 3, 4]}]}, "Frame .* is not in this video"),
    ({"frame": 5}, "at least 2"),
])
def test_bad_picks_are_refused_with_the_reason(data, message):
    with pytest.raises(PickError, match=message):
        parse_picks(data, n_frames=10)


def test_picks_survive_a_round_trip_through_json(tmp_path):
    picks = parse_picks(_picks(("Karina", [1, 2, 3, 4]), ("Winter", [5, 6, 7, 8])), n_frames=10)
    path = tmp_path / "picks.json"
    path.write_text(json.dumps(dump_picks(picks)), encoding="utf-8")
    again = load_picks(path, n_frames=10)
    assert again.frame == 5 and [p.name for p in again.members] == ["Karina", "Winter"]
    assert np.array_equal(again.members[0].box, [1, 2, 3, 4])


def test_each_pick_resolves_to_the_detection_under_it():
    det = _frame_people(THREE)
    picks = parse_picks(_picks(("C", [305, 105, 405, 405]), ("A", [505, 95, 605, 395]), ("B", [95, 100, 195, 400])), 10)
    assert resolve_picks(picks, det) == [2, 0, 1]


def test_a_pick_where_nobody_stands_is_refused_by_name():
    det = _frame_people(THREE)
    picks = parse_picks(_picks(("Winter", [100, 100, 200, 400]), ("Karina", [700, 100, 800, 400])), 10)
    with pytest.raises(PickError, match="No person found where Karina was picked in frame 5"):
        resolve_picks(picks, det)


def test_two_picks_on_the_same_person_are_refused():  # Review Focus 4
    det = _frame_people(THREE)
    picks = parse_picks(_picks(("Karina", [100, 100, 200, 400]), ("Winter", [104, 100, 204, 400])), 10)
    with pytest.raises(PickError, match="Karina and Winter were picked on the same person"):
        resolve_picks(picks, det)


def test_people_in_a_frame_are_listed_left_to_right_with_clear_flags():
    det = _frame_people(THREE)
    people = people_in_frame(det, 5, min_conf=0.3)
    assert people == [{"row": 1, "box": [100.0, 100.0, 200.0, 400.0], "clear": True},
                      {"row": 2, "box": [300.0, 100.0, 400.0, 400.0], "clear": False},
                      {"row": 0, "box": [500.0, 100.0, 600.0, 400.0], "clear": True}]
    assert people_in_frame(det, 7, min_conf=0.3) == []


def _box(x):
    return [x, 100, x + 80, 400]


def test_suggested_frames_favour_clear_separate_people_away_from_the_ends():
    people = []
    for f in range(300):
        if 100 <= f <= 120:
            xs = [100, 300, 500, 700]
        elif 200 <= f <= 210:
            xs = [100, 150, 500, 700]  # two of them overlap
        elif f == 5:
            xs = [100, 300, 500, 700, 900]
        else:
            xs = [100, 500]
        people += [(f, _box(x), True) for x in xs]
    det = _frame_people(people, n_frames=300)
    suggested = suggest_frames(det, min_conf=0.3)
    assert 100 <= suggested[0] <= 120
    assert not any(f < 15 or f >= 285 for f in suggested)
    assert not any(200 <= f <= 210 for f in suggested)
    assert all(abs(a - b) >= 60 for i, a in enumerate(suggested) for b in suggested[i + 1:])
    assert 1 < len(suggested) <= 5
