import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from ml.config import Config
from ml.data.loader import load_raw
from ml.detectors.dbscan import DBSCANDetector
from ml.features.engineer import build_training_frame


@pytest.fixture(scope="module")
def unscaled():
    X, _, _ = build_training_frame(load_raw(Config.load().get("data.csv_path")))
    return X


@pytest.fixture(scope="module")
def scaled():
    X, _, _ = build_training_frame(load_raw(Config.load().get("data.csv_path")))
    return pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)


@pytest.fixture(scope="module")
def fitted(scaled):
    return DBSCANDetector(contamination=0.05, eps=3.0, min_samples=5).fit(scaled)


def test_flags_the_contamination_rate(fitted):
    assert fitted.fit_flags_.sum() == 126
    assert fitted.name == "dbscan"
    assert fitted.live_scorable is True


def test_core_samples_are_retained_for_serving(fitted, scaled):
    assert fitted.core_samples_.ndim == 2
    assert fitted.core_samples_.shape[1] == scaled.shape[1]
    assert fitted.core_samples_.shape[0] > 1000


def test_native_noise_rate_is_reported(fitted):
    assert 0.0 < fitted.native_noise_rate_ < 0.20
    assert fitted.n_clusters_ > 1


def test_calibrated_flags_are_a_subset_of_native_noise(fitted, scaled):
    """The calibrated flag tightens DBSCAN's own noise definition (spec 4.6)."""
    native_noise = fitted._model.labels_ == -1
    calibrated = fitted.fit_flags_ == 1
    assert calibrated.sum() <= native_noise.sum()
    assert np.all(native_noise[calibrated]), (
        "every calibrated flag must also be a native DBSCAN noise point"
    )


def test_scores_a_single_unseen_row(fitted, scaled):
    row = scaled.iloc[[0]]
    assert fitted.score(row).shape == (1,)
    assert fitted.flag(row).shape == (1,)


def test_score_is_a_non_negative_distance(fitted, scaled):
    assert (fitted.score(scaled) >= 0).all()


def test_dbscan_score_direction(fitted, unscaled):
    # Verify that _score returns nearest-core-sample distance, not its
    # negation: flagged rows should have higher UtilizationRatio than normal
    # rows. fit_flags_.sum() == 126 alone proves nothing about score
    # direction -- BaseDetector always flags whatever sits above the 95th
    # percentile of the score it is given, so a count check passes regardless
    # of which end of the distribution "anomalous" means. If _score returned
    # negated distances, the flagged set would no longer be the rows farthest
    # from every core sample; this test would fail because flagged rows would
    # not have distinctly higher UtilizationRatio than normal rows.
    flagged_util = unscaled.loc[fitted.fit_flags_ == 1, "UtilizationRatio"].mean()
    normal_util = unscaled.loc[fitted.fit_flags_ == 0, "UtilizationRatio"].mean()
    assert flagged_util > normal_util * 3  # Flagged should be at least 3x higher
