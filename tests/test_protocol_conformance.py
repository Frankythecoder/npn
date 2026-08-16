"""Spec 14: Protocol conformance, parametrised across all eight detectors."""
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler, StandardScaler

from ml.config import Config
from ml.data.loader import load_raw
from ml.detectors.base import AnomalyDetector
from ml.detectors.registry import DETECTOR_ORDER, build_detectors
from ml.features.engineer import build_training_frame


@pytest.fixture(scope="module")
def fitted_detectors():
    cfg = Config.load()
    X, artifacts, _ = build_training_frame(load_raw(cfg.get("data.csv_path")))
    frames = {
        "standard": pd.DataFrame(
            StandardScaler().fit_transform(X), columns=X.columns
        ),
        "robust": pd.DataFrame(RobustScaler().fit_transform(X), columns=X.columns),
        "continuous": pd.DataFrame(
            StandardScaler().fit_transform(X[artifacts.continuous_columns]),
            columns=artifacts.continuous_columns,
        ),
    }
    fitted = {}
    for detector in build_detectors(cfg):
        frame = frames[detector.scaler]
        fitted[detector.name] = (detector.fit(frame), frame)
    return fitted


@pytest.mark.parametrize("name", DETECTOR_ORDER)
def test_satisfies_the_protocol(fitted_detectors, name):
    detector, _ = fitted_detectors[name]
    assert isinstance(detector, AnomalyDetector)
    assert detector.name == name
    assert detector.view in {"full", "continuous"}
    assert isinstance(detector.live_scorable, bool)


@pytest.mark.parametrize("name", DETECTOR_ORDER)
def test_fit_returns_self(fitted_detectors, name):
    detector, frame = fitted_detectors[name]
    assert detector.fit(frame) is detector


@pytest.mark.parametrize("name", DETECTOR_ORDER)
def test_score_and_flag_shapes(fitted_detectors, name):
    detector, frame = fitted_detectors[name]
    scores = detector.score(frame)
    flags = detector.flag(frame)
    assert scores.shape == (len(frame),)
    assert flags.shape == (len(frame),)
    assert set(np.unique(flags)).issubset({0, 1})


@pytest.mark.parametrize("name", DETECTOR_ORDER)
def test_flag_rate_matches_contamination_within_one_row(fitted_detectors, name):
    detector, frame = fitted_detectors[name]
    expected = detector.contamination * len(frame)
    assert abs(detector.fit_flags_.sum() - expected) <= 1.0


@pytest.mark.parametrize("name", DETECTOR_ORDER)
def test_flag_agrees_with_thresholding_score(fitted_detectors, name):
    detector, frame = fitted_detectors[name]
    expected = (detector.score(frame) >= detector.live_threshold_).astype(int)
    assert np.array_equal(detector.flag(frame), expected)


@pytest.mark.parametrize("name", DETECTOR_ORDER)
def test_scores_a_single_row(fitted_detectors, name):
    detector, frame = fitted_detectors[name]
    assert detector.score(frame.iloc[[0]]).shape == (1,)


@pytest.mark.parametrize("name", DETECTOR_ORDER)
def test_score_percentile_is_bounded(fitted_detectors, name):
    detector, frame = fitted_detectors[name]
    value = float(detector.score(frame.iloc[[0]])[0])
    assert 0.0 <= detector.score_percentile(value) <= 100.0
