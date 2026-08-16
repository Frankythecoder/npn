import pytest
from ml.config import Config, CONFIG_PATH


def test_config_file_exists():
    assert CONFIG_PATH.exists()


def test_load_returns_expected_values():
    cfg = Config.load()
    assert cfg.get("detectors.contamination") == 0.05
    assert cfg.get("ensemble.threshold") == 0.5
    assert cfg.get("detectors.dbscan.eps") == 3.0
    assert cfg.get("detectors.dbscan.min_samples") == 5
    assert cfg.get("detectors.random_state") == 42
    assert cfg.get("validation.sane_band") == [0.03, 0.07]


def test_get_with_default_for_missing_key():
    cfg = Config.load()
    assert cfg.get("detectors.nonexistent", "fallback") == "fallback"


def test_get_raises_on_missing_key_without_default():
    cfg = Config.load()
    with pytest.raises(KeyError):
        cfg.get("detectors.nonexistent")


def test_missing_required_key_rejected_at_load():
    with pytest.raises(ValueError, match="missing required key"):
        Config({"data": {}})
