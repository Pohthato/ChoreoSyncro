"""Name pieces of track by fingerprint, starting from the members picked in one frame.

Each class (every member, then "not a member") has example fingerprints. A piece's score for a
class is the mean over its rows of the mean of each row's top-k cosine similarities to the
examples: a k-nearest-neighbour vote, so a member seen from several sides keeps several clusters.

- A piece takes a member's name only when she beats every other class by `name_margin`.
- Learning rounds: pieces that win by the stricter `learn_margin` become examples too, which
  teaches how each member looks from other sides and under other light. Each round re-decides
  who belongs where, and a piece is never scored against its own rows, so an early wrong call
  does not confirm itself.
- A piece seen at the same time as a member's examples can't be her, so it is never learned as
  her (a lookalike standing next to her stays out of her examples).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import IdentityConfig

TOP_K = 5
MAX_ROWS = 64        # rows of a piece used for scoring, evenly spaced
MAX_EXAMPLES = 4000  # examples per class, evenly spaced
MAX_SHARED = 2       # frames a learned piece may share with the member's other examples (tracker jitter)


@dataclass
class Naming:
    member: int | None
    margin: float
    scores: np.ndarray  # one per member, then not-a-member


def _evenly(n: int, cap: int) -> np.ndarray:
    return np.arange(n) if n <= cap else np.unique(np.linspace(0, n - 1, cap).round().astype(int))


def _usable(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    rows = rows[np.any(rows != 0, axis=1)]
    return rows[_evenly(len(rows), MAX_ROWS)]


def _top_k_mean(similarity: np.ndarray) -> float:
    k = min(TOP_K, similarity.shape[1])
    if similarity.shape[0] == 0 or k == 0:
        return math.nan
    return float(np.partition(similarity, -k, axis=1)[:, -k:].mean())


def class_score(rows: np.ndarray, examples: np.ndarray) -> float:
    """Mean over rows of the mean of each row's top-k cosine similarities to `examples`; NaN if either is empty."""
    examples = np.asarray(examples, dtype=np.float32)
    examples = examples[np.any(examples != 0, axis=1)]
    return _top_k_mean(_usable(rows) @ examples.T)


def _naming(scores: np.ndarray, margin: float, exclude=()) -> Naming:
    """Best allowed member against everyone else, excluded members included: a piece that looks
    like an excluded member is evidence it is not anyone else."""
    members = len(scores) - 1
    allowed = [m for m in range(members) if m not in exclude and not math.isnan(scores[m])]
    if not allowed:
        return Naming(None, math.nan, scores)
    best = max(allowed, key=lambda m: scores[m])
    rivals = [s for c, s in enumerate(scores) if c != best and not math.isnan(s)]
    lead = float(scores[best] - max(rivals)) if rivals else math.inf
    return Naming(best if lead >= margin else None, lead, scores)


def overlap(a_frames: np.ndarray, b_frames: np.ndarray) -> int:
    return len(np.intersect1d(a_frames, b_frames))


def _scores_without(rows: np.ndarray, matrices: list[np.ndarray], owners: list[np.ndarray], skip: list[int]) -> np.ndarray:
    """Each class's score for `rows`, leaving out its examples from the pieces in `skip`."""
    rows = _usable(rows)
    return np.array([_top_k_mean(rows @ matrix[~np.isin(owner, skip)].T) for matrix, owner in zip(matrices, owners)])


@dataclass
class Recognizer:
    examples: list[np.ndarray]  # per member
    others: np.ndarray          # not a member
    margin: float               # name_margin
    owners: list[np.ndarray]    # the piece each example came from, per class (members, then not a member)

    def scores(self, rows: np.ndarray) -> np.ndarray:
        return np.array([class_score(rows, e) for e in [*self.examples, self.others]])

    def name(self, rows: np.ndarray, exclude=()) -> Naming:
        return _naming(self.scores(rows), self.margin, exclude)

    def likeness(self, member: int, rows: np.ndarray, skip: list[int]) -> float:
        """Her class score for `rows`, without her examples from the `skip` pieces."""
        return float(_scores_without(rows, [self.examples[member]], [self.owners[member]], skip)[0])


class _Classes:
    """Example rows per class with the piece each came from, so a piece can be scored without its own rows."""

    def __init__(self, pieces: list[np.ndarray], members: list[list[int]]):
        self.matrices, self.owners = [], []
        for chosen in members:
            rows = [np.asarray(pieces[p], np.float32) for p in chosen]
            owner = np.concatenate([np.full(len(r), p) for p, r in zip(chosen, rows)]) if chosen else np.zeros(0, int)
            matrix = np.vstack(rows) if chosen else np.zeros((0, pieces[0].shape[1]), np.float32)
            keep = np.flatnonzero(np.any(matrix != 0, axis=1))
            keep = keep[_evenly(len(keep), MAX_EXAMPLES)]
            self.matrices.append(matrix[keep])
            self.owners.append(owner[keep])

    def scores(self, rows: np.ndarray, piece: int) -> np.ndarray:
        return _scores_without(rows, self.matrices, self.owners, [piece])


def learn(pieces: list[np.ndarray], frames: list[np.ndarray], seeds: list[int], not_members: list[int],
          config: IdentityConfig) -> tuple[Recognizer, list[Naming]]:
    """Name every piece, learning from the confident ones for up to `config.rounds` rounds.

    `seeds[i]` is the piece picked as member i; `not_members` are the other people in the picked frame."""
    n_members = len(seeds)
    fixed = {p: m for m, p in enumerate(seeds)} | {p: None for p in not_members}
    candidates = [p for p in range(len(pieces)) if p not in fixed]
    joined: dict[int, int | None] = {}  # learned piece -> member (None = not a member)

    def classes() -> _Classes:
        members = [[seeds[m]] + [p for p, j in joined.items() if j == m] for m in range(n_members)]
        return _Classes(pieces, members + [list(not_members) + [p for p, j in joined.items() if j is None]])

    def name_all(current: _Classes, margin: float) -> dict[int, Naming]:
        return {p: _naming(current.scores(pieces[p], p), margin) for p in candidates}

    for _ in range(config.rounds):
        names = name_all(classes(), config.name_margin)
        new: dict[int, int | None] = {}
        taken = {m: [frames[seeds[m]]] for m in range(n_members)}
        for p in sorted(candidates, key=lambda p: -names[p].margin if not math.isnan(names[p].margin) else math.inf):
            m = names[p].member
            if m is not None and names[p].margin >= config.learn_margin:
                if all(overlap(frames[p], f) <= MAX_SHARED for f in taken[m]):
                    new[p] = m
                    taken[m].append(frames[p])
            elif m is None:
                scores = names[p].scores
                member_best = np.nanmax(scores[:-1]) if np.any(~np.isnan(scores[:-1])) else math.nan
                if not math.isnan(scores[-1]) and not math.isnan(member_best) and scores[-1] - member_best >= config.learn_margin:
                    new[p] = None
        if new == joined:
            break
        joined = new

    final = classes()
    names = name_all(final, config.name_margin)
    namings = [Naming(fixed[p], math.inf, np.full(n_members + 1, math.nan)) if p in fixed else names[p]
               for p in range(len(pieces))]
    recognizer = Recognizer(examples=final.matrices[:-1], others=final.matrices[-1], margin=config.name_margin,
                            owners=final.owners)
    return recognizer, namings
