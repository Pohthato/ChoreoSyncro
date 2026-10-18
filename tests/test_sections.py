import dataclasses

import numpy as np
import pytest

from syncscore.config import load_config
from syncscore.pipeline import analyze_keypoints
from syncscore.sync.sections import _block_bootstrap_median, section_timing
from syncscore.sync.summary import make_sections
from syncscore.synth.generate import GroupSpec, Interval, make_group


@pytest.fixture(scope="module")
def noisy_group():
    base = load_config()
    config = dataclasses.replace(base, sections=dataclasses.replace(base.sections, chunk_s=10.0))  # 5 blocks each
    offsets = [0, 40, -30, 90, 0]
    spec = GroupSpec(n_members=5, duration_s=40, fps=30, offsets_ms=offsets, noise=0.02, dropout=0.03, style_deg=4.0,
                     occlusions=[Interval(2, 16.0, 26.0)], seed=31)
    group = make_group(spec)
    result, _, _ = analyze_keypoints(group.keypoints, group.conf, group.fps, config)
    timing = section_timing(result, make_sections(config, 40.0), config)
    assert len(timing.names) == 4
    return np.array(offsets, float), timing


def test_sections_recover_known_offsets(noisy_group):
    truth, timing = noisy_group
    ok = timing.reliable
    err = np.abs(timing.offset_ms - (truth - np.median(truth))[:, None])
    assert ok.mean() > 0.7
    assert np.median(err[ok]) < 6.0


def test_section_intervals_are_honest(noisy_group):
    truth, timing = noisy_group
    rel = (truth - np.median(truth))[:, None]
    inside = (timing.low_ms - 3.0 <= rel) & (rel <= timing.high_ms + 3.0)  # 3 ms slack for the reference itself
    assert inside[timing.reliable].mean() > 0.8


def test_a_hidden_member_gets_no_reading(noisy_group):
    _, timing = noisy_group
    # 16-26 s hidden: the 20-30 s section is mostly unseen, the 10-20 s one partly.
    hidden = timing.starts_s == 20.0
    assert hidden.any()
    assert not timing.reliable[2, hidden].any()


def test_block_bootstrap_is_wider_than_pretending_windows_are_independent():
    rng = np.random.default_rng(1)
    # Strongly correlated neighbours, as overlapping windows are: 8 copies of each value.
    values = np.repeat(rng.normal(0, 10, size=6), 8)
    lo_b, hi_b = _block_bootstrap_median(values, block=8, rng=np.random.default_rng(0))
    lo_i, hi_i = _block_bootstrap_median(values, block=1, rng=np.random.default_rng(0))
    assert (hi_b - lo_b) > 1.5 * (hi_i - lo_i)


def test_a_short_tail_is_folded_into_the_last_section():
    from syncscore.config import load_config as load

    config = load()
    sections = make_sections(config, 230.0)  # 32 s chunks: 7 full ones and a 6 s tail
    assert sections[-1] == (192.0, 230.0, "192-230s")
    assert len(make_sections(config, 16.0)) == 1
