import numpy as np
import pandas as pd
import pytest

from ml.detectors.base import BaseDetector


class FakeDetector(BaseDetector):
    """Scores each row by its first column, so expectations are exact."""

    name = "fake"
    view = "full"
    scaler = "standard"
    live_scorable = True

    def _fit(self, X: np.ndarray) -> None:
        self._fitted = True

    def _score(self, X: np.ndarray) -> np.ndarray:
        return X[:, 0].astype(float)


@pytest.fixture
def frame():
    return pd.DataFrame({"a": np.arange(100.0), "b": np.zeros(100)})


def test_fit_returns_self(frame):
    det = FakeDetector(contamination=0.05)
    assert det.fit(frame) is det


def test_threshold_is_the_contamination_percentile(frame):
    det = FakeDetector(contamination=0.05).fit(frame)
    assert det.threshold_ == pytest.approx(np.percentile(np.arange(100.0), 95))


def test_flag_rate_matches_contamination_on_training_data(frame):
    det = FakeDetector(contamination=0.05).fit(frame)
    flags = det.flag(frame)
    assert flags.shape == (100,)
    assert set(np.unique(flags)).issubset({0, 1})
    assert abs(flags.mean() - 0.05) <= 0.01


def test_fit_flags_matches_flag_on_training_data(frame):
    det = FakeDetector(contamination=0.05).fit(frame)
    assert np.array_equal(det.fit_flags_, det.flag(frame))


def test_flag_agrees_with_thresholding_score(frame):
    det = FakeDetector(contamination=0.05).fit(frame)
    expected = (det.score(frame) >= det.live_threshold_).astype(int)
    assert np.array_equal(det.flag(frame), expected)


def test_train_scores_are_sorted(frame):
    det = FakeDetector(contamination=0.05).fit(frame)
    assert np.array_equal(det.train_scores_, np.sort(det.train_scores_))
    assert len(det.train_scores_) == 100


def test_score_percentile_bounds(frame):
    det = FakeDetector(contamination=0.05).fit(frame)
    assert det.score_percentile(-1.0) == pytest.approx(0.0)
    assert det.score_percentile(999.0) == pytest.approx(100.0)
    assert 45.0 <= det.score_percentile(50.0) <= 55.0


def test_score_before_fit_raises(frame):
    det = FakeDetector(contamination=0.05)
    with pytest.raises(RuntimeError, match="not fitted"):
        det.flag(frame)


def test_column_mismatch_is_rejected(frame):
    det = FakeDetector(contamination=0.05).fit(frame)
    with pytest.raises(ValueError, match="column mismatch"):
        det.score(pd.DataFrame({"a": [1.0], "z": [2.0]}))
