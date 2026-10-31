import dataclasses
import math

import numpy as np
import pytest

from scenes import near, unit
from syncscore.config import IdentityConfig
from syncscore.pose.recognize import Recognizer, class_score, learn, overlap

CONFIG = IdentityConfig()


def normalise(v):
    return v / np.linalg.norm(v)


class Scene:
    """Four members a..d and two dancers e, f, all seen in the picked frame (frames 0-29)."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.a, self.b, self.c, self.d, self.e, self.f = (unit(self.rng) for _ in range(6))
        self.pieces, self.frames = [], []
        for centre in (self.a, self.b, self.c, self.d, self.e, self.f):
            self.add(centre, start=0)
        self.seeds, self.not_members = [0, 1, 2, 3], [4, 5]
        self.next_start = 100

    def rows(self, centre, n=30):
        rows = centre + 0.05 * self.rng.normal(size=(n, len(centre)))
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)

    def add(self, centre, start=None, n=30) -> int:
        if start is None:
            start, self.next_start = self.next_start, self.next_start + n + 10
        self.pieces.append(self.rows(centre, n))
        self.frames.append(np.arange(start, start + n))
        return len(self.pieces) - 1

    def learn(self, config=CONFIG):
        return learn(self.pieces, self.frames, self.seeds, self.not_members, config)


def test_pieces_take_the_name_of_the_member_they_look_like():
    s = Scene()
    new = [s.add(s.b), s.add(s.d), s.add(s.a)]
    _, names = s.learn()
    assert [names[p].member for p in new] == [1, 3, 0]
    assert all(names[p].margin > 0.3 for p in new)


def test_seeds_come_back_named_with_an_infinite_margin():
    s = Scene()
    _, names = s.learn()
    assert [(n.member, n.margin) for n in names] == [(0, math.inf), (1, math.inf), (2, math.inf), (3, math.inf),
                                                      (None, math.inf), (None, math.inf)]


def test_a_piece_between_two_members_stays_unassigned():
    s = Scene()
    p = s.add(normalise(s.a + s.b))
    _, names = s.learn()
    assert names[p].member is None


def test_a_dancer_not_in_the_picked_frame_is_not_named():
    s = Scene()
    g = s.rng.normal(size=32)  # someone who looks like nobody seen so far
    for known in np.linalg.qr(np.stack([s.a, s.b, s.c, s.d, s.e, s.f]).T)[0].T:
        g -= (g @ known) * known
    p = s.add(normalise(g))
    _, names = s.learn()
    assert names[p].member is None


def test_learning_rounds_teach_a_members_other_side():
    s = Scene()
    a2 = normalise(0.3 * s.a + 0.95 * near(s.a, 0.0, s.rng))
    s.not_members.append(s.add(near(a2, 0.55, s.rng), start=0))  # a dancer in the picked frame who looks like a's other side
    chain = [s.add(normalise((1 - t) * s.a + t * a2)) for t in (0.25, 0.5, 0.75, 1.0)]
    _, before = s.learn(dataclasses.replace(CONFIG, rounds=0))
    _, after = s.learn(dataclasses.replace(CONFIG, rounds=3))
    assert before[chain[-1]].member is None
    assert [after[p].member for p in chain] == [0, 0, 0, 0]


def test_learning_does_not_drift_onto_a_decoy_near_a_member():
    s = Scene()
    decoy = near(s.b, 0.8, s.rng)
    s.not_members.append(s.add(decoy, start=0))
    decoys = [s.add(decoy) for _ in range(5)]
    bs = [s.add(s.b) for _ in range(5)]
    _, names = s.learn()
    assert [names[p].member for p in decoys] == [None] * 5
    assert [names[p].member for p in bs] == [1] * 5


def test_a_piece_seen_at_the_same_time_as_a_member_is_never_learned_as_her():
    s = Scene()
    s.add(near(s.a, 0.95, s.rng), start=0)  # frames 0-29, the same as a's seed
    recognizer, _ = s.learn()
    assert recognizer.examples[0].shape == s.pieces[0].shape and np.allclose(recognizer.examples[0], s.pieces[0], atol=1e-6)


def test_excluding_a_member_rescores_without_it():
    s = Scene()
    recognizer, _ = s.learn()
    rows = s.rows(s.a)
    assert recognizer.name(rows).member == 0
    assert recognizer.name(rows, exclude={0}).member is None


def test_a_pieces_own_rows_do_not_count_toward_how_much_it_looks_like_her():
    rng = np.random.default_rng(0)
    a = unit(rng)
    own = np.stack([near(a, 0.7, rng) for _ in range(10)], dtype=np.float32)
    rest = np.stack([near(a, 0.99, rng) for _ in range(10)], dtype=np.float32)
    recognizer = Recognizer(examples=[np.vstack([rest, own])], others=np.zeros((0, 32), np.float32), margin=0.06,
                            owners=[np.array([1] * 10 + [2] * 10), np.zeros(0, int)])
    assert recognizer.likeness(0, own, skip=[2]) == pytest.approx(class_score(own, rest))
    assert recognizer.likeness(0, own, skip=[]) > recognizer.likeness(0, own, skip=[2]) + 0.02


def test_rows_without_a_fingerprint_are_ignored():
    s = Scene()
    recognizer, _ = s.learn()
    rows = np.vstack([s.rows(s.a), np.zeros((10, 32))])
    assert recognizer.name(rows).member == 0
    blank = recognizer.name(np.zeros((10, 32)))
    assert blank.member is None and math.isnan(blank.margin)


def test_class_score_is_the_mean_of_each_rows_top_k_similarities():
    examples = np.eye(4)  # four orthogonal unit examples
    row = normalise(np.array([1.0, 1.0, 0.0, 0.0]))
    # 4 examples, so top-k takes all of them: (0.707 + 0.707 + 0 + 0) / 4
    assert class_score(row[None], examples) == pytest.approx(math.sqrt(0.5) / 2)
    assert math.isnan(class_score(np.zeros((3, 4)), examples))


def test_overlap_counts_shared_frames():
    assert overlap(np.arange(0, 10), np.arange(8, 20)) == 2
    assert overlap(np.arange(0, 10), np.arange(10, 20)) == 0
