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
