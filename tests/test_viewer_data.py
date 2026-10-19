"""The viewer's data file: everything the page draws, computed in Python, no pixels."""
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import numpy as np
import pytest

from syncscore.config import load_config
from syncscore.features.bones import BONE_NAMES
from syncscore.outputs import write_outputs
from syncscore.pipeline import analyze_keypoints
from syncscore.synth.generate import GroupSpec, LimbError, make_group
from syncscore.viz.viewer_data import decode_int16, encode_int16, video_url, viewer_data, write_viewer

KEYS = {"version", "fps", "n_frames", "image_size", "names", "in_sync_ms", "off_sync_ms", "shape_threshold_deg", "video",
        "sections", "section_status", "keypoints", "ghost", "conf", "halo", "moments"}


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    config = load_config()
    spec = GroupSpec(n_members=3, duration_s=16, fps=30, offsets_ms=[0, 0, 100], noise=0.005,
                     limb_errors=[LimbError(member=1, joint="r_knee", degrees=30.0, t0=2.0, t1=14.0)], seed=21)
    group = make_group(spec)
    result, xy, conf = analyze_keypoints(group.keypoints, group.conf, group.fps, config, names=group.names)
    out = tmp_path_factory.mktemp("results")
    write_outputs(result, xy, conf, out, config, spec.image_size)
    return out, config, spec


def test_int16_round_trip_keeps_quarter_pixels_and_nan():
    values = np.array([[0.0, 12.3456], [np.nan, -731.9], [1919.99, 3.0]])
    back = decode_int16(encode_int16(values), values.shape)
    assert np.isnan(back[1, 0])
    finite = np.isfinite(values)
    assert np.allclose(back[finite], values[finite], atol=0.125)


def test_viewer_data_has_every_part_the_page_reads(results):
    out, config, spec = results
    data = viewer_data(out, config, spec.image_size, None)
    assert set(data) >= KEYS
    smoothed = np.load(out / "keypoints_smoothed.npz")["xy"]
    keypoints = decode_int16(data["keypoints"], smoothed.shape)
    both = np.isfinite(smoothed)
    assert np.array_equal(both, np.isfinite(keypoints))
    assert np.allclose(keypoints[both], smoothed[both], atol=0.125)
    assert data["n_frames"] == smoothed.shape[1] and data["names"] == ["M1", "M2", "M3"]
    assert len(data["section_status"]) == 3 and len(data["section_status"][0]) == len(data["sections"])


def test_halo_marks_the_planted_limb_error(results):
    out, config, spec = results
    data = viewer_data(out, config, spec.image_size, None)
    import base64

    halo = np.frombuffer(base64.b64decode(data["halo"]), dtype="<u2").reshape(3, data["n_frames"])
    bit = 1 << BONE_NAMES.index("right_shin")
    t = np.arange(data["n_frames"]) / data["fps"]
    during = (t >= 3.0) & (t < 13.0)
    assert ((halo[1, during] & bit) > 0).mean() > 0.8
    assert ((halo[0] & bit) > 0).mean() < 0.05


def test_no_video_gives_null(results):
    out, config, spec = results
    assert viewer_data(out, config, spec.image_size, None)["video"] is None


def test_video_path_becomes_a_file_url(tmp_path):
    path = tmp_path / "BTS (방탄소년단) 'Boy With Luv' Dance Practice.mp4"
    url = video_url(path)
    assert url.startswith("file:///") and " " not in url
    assert Path(url2pathname(urlparse(url).path)) == path.resolve()


def test_write_viewer_writes_the_data_script(results, tmp_path):
    out, config, spec = results
    index = write_viewer(out, tmp_path / "viewer", config, spec.image_size, None)
    text = (index.parent / "viewer_data.js").read_text(encoding="utf-8")
    prefix = "window.SYNCSCORE = "
    assert text.startswith(prefix)
    data = json.loads(text[len(prefix):].rstrip().rstrip(";"))
    assert set(data) >= KEYS


def test_data_size_for_a_full_clip():
    rng = np.random.default_rng(0)
    shape = (7, 6905, 17, 2)
    points = rng.uniform(0, 1920, size=shape)
    size = 2 * len(encode_int16(points))  # skeletons and ghosts
    import base64

    size += len(base64.b64encode(rng.integers(0, 256, size=shape[:3], dtype=np.uint8).tobytes()))
    size += len(base64.b64encode(rng.integers(0, 2**13, size=shape[:2], dtype="<u2").tobytes()))
    assert size < 15e6


def test_page_reads_only_keys_the_data_has(results, tmp_path):
    import re

    out, config, spec = results
    index = write_viewer(out, tmp_path / "viewer", config, spec.image_size, None)
    page = index.parent
    used = set(re.findall(r"\bD\.(\w+)", (page / "viewer.js").read_text(encoding="utf-8")))
    assert used, "viewer.js should read the data through D"
    assert used <= set(viewer_data(out, config, spec.image_size, None))
    html = index.read_text(encoding="utf-8")
    scripts = re.findall(r'<script src="([^"]+)"', html)
    sheets = re.findall(r'<link rel="stylesheet" href="([^"]+)"', html)
    assert scripts == ["viewer_data.js", "viewer_core.js", "viewer.js"] and sheets == ["viewer.css"]
    assert all((page / name).is_file() for name in scripts + sheets)
    assert not re.search(r'(src|href)="(https?:)?//', html), "the page must not load anything from the network"


def test_view_command_writes_a_viewer(results, tmp_path):
    import cv2

    from syncscore.cli import main

    out, _, _ = results
    video = tmp_path / "practice clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for k in range(60):
        writer.write(np.full((48, 64, 3), k * 4, dtype=np.uint8))
    writer.release()
    assert main(["view", str(video), "--results", str(out)]) == 0
    page = out / "viewer"
    assert (page / "index.html").is_file()
    assert video_url(video) in (page / "viewer_data.js").read_text(encoding="utf-8")
