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
