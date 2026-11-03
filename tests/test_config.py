from dataclasses import replace

import pytest

from syncscore.config import load_config


def test_default_config_loads():
    config = load_config()
    assert config.pose.model.endswith("-pose.pt")
    assert config.features.mirror == "off"  # quoted in YAML so it is not parsed as False
    assert config.sync.max_lag_ms == 400


def test_unknown_key_fails(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("sync:\n  windw_s: 2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="windw_s"):
        load_config(path)


def test_cli_survives_a_non_ascii_video_name(tmp_path, capsys):
    from syncscore.cli import main

    missing = tmp_path / "방탄소년단 dance.mp4"
    assert main(["analyze", str(missing), "--out", str(tmp_path / "out")]) == 2


def test_identity_settings_load_and_reject_unknown_keys(tmp_path):
    from syncscore.pose.fingerprint import MODELS

    identity = load_config().identity
    assert identity.rounds == 3 and identity.free_after_s == 3.0 and identity.model in MODELS
    bad = tmp_path / "bad.yaml"
    bad.write_text("identity:\n  margin: 0.1\n")
    with pytest.raises(ValueError, match="IdentityConfig"):
        load_config(bad)


def test_a_config_file_gets_the_packaged_defaults_for_what_it_leaves_out(tmp_path):
    # A clip's config lists only what differs for it; everything else must be what default.yaml says,
    # including values measured since (the identity margins).
    path = tmp_path / "clip.yaml"
    path.write_text("tracking:\n  members: 4\n", encoding="utf-8")
    packaged = load_config()
    assert load_config(path) == replace(packaged, tracking=replace(packaged.tracking, members=4))
