"""The deliverables are written and the summary says what was planted."""
import json

import numpy as np
import pytest

from syncscore.config import load_config
from syncscore.outputs import write_outputs
from syncscore.pipeline import analyze_keypoints
from syncscore.synth.generate import GroupSpec, LimbError, make_group
from syncscore.viz.style import signed_ms


def test_signed_ms_never_prints_minus_zero():
    assert signed_ms(-0.3) == "0 ms"
    assert signed_ms(12.4) == "+12 ms"
    assert signed_ms(-8.6) == "-9 ms"


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    config = load_config()
    spec = GroupSpec(n_members=3, duration_s=16, fps=30, offsets_ms=[0, 0, 100], noise=0.005,
                     limb_errors=[LimbError(member=1, joint="r_knee", degrees=30.0, t0=2.0, t1=14.0)], seed=21)
    group = make_group(spec)
    result, xy, conf = analyze_keypoints(group.keypoints, group.conf, group.fps, config, names=group.names)
    out = tmp_path_factory.mktemp("out")
    summary = write_outputs(result, xy, conf, out, config, spec.image_size)
    return out, summary


def test_every_deliverable_exists(written):
    out, _ = written
    for name in ("summary.json", "summary.csv", "heatmap.png", "skeletons.mp4", "sync_result.npz", "keypoints_smoothed.npz"):
        assert (out / name).stat().st_size > 0, name
    assert sorted(p.name for p in (out / "report_cards").iterdir()) == ["M1.png", "M2.png", "M3.png"]


def test_summary_names_the_planted_errors(written):
    out, _ = written
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    members = {m["member"]: m["overall"] for m in summary["members"]}
    assert members["M3"]["mean_offset_ms"] == pytest.approx(100.0, abs=5.0)
    assert members["M3"]["section_typical_offset_ms"] == pytest.approx(100.0, abs=5.0)
    assert members["M3"]["sections_measured_share"] == 1.0
    assert members["M2"]["shape_part_most_often_off"] == "legs"
    assert members["M2"]["worst_shape_moments"][0]["bone"] == "right_shin"
    assert members["M1"]["shape_part_most_often_off"] is None
    assert len(summary["members"][0]["sections"]) == 1  # 16 s clip, 32 s sections
