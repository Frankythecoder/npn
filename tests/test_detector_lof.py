import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

from ml.config import Config
from ml.data.loader import load_raw
from ml.detectors.lof import LOFDetector
from ml.features.engineer import build_training_frame


@pytest.fixture(scope="module")
def unscaled():
    X, _, _ = build_training_frame(load_raw(Config.load().get("data.csv_path")))
    return X


@pytest.fixture(scope="module")
def scaled():
    X, _, _ = build_training_frame(load_raw(Config.load().get("data.csv_path")))
    return pd.DataFrame(RobustScaler().fit_transform(X), columns=X.columns)


@pytest.fixture(scope="module")
def fitted(scaled):
    return LOFDetector(contamination=0.05, n_neighbors=20).fit(scaled)


def test_uses_the_robust_scaler(fitted):
    assert fitted.scaler == "robust"
    assert fitted.name == "lof"
    assert fitted.live_scorable is True


def test_flags_the_contamination_rate(fitted):
    assert fitted.fit_flags_.sum() == 126


def test_two_distinct_thresholds_are_persisted(fitted):
    assert fitted.threshold_ is not None
    assert fitted.live_threshold_ is not None
    assert fitted.threshold_ != fitted.live_threshold_, (
        "the novelty=False and novelty=True fits are different quantities "
        "and must carry separate thresholds (spec 4.5)"
    )


def test_both_score_distributions_are_persisted(fitted):
    assert len(fitted.train_scores_) == 2512
    assert len(fitted.live_train_scores_) == 2512


def test_novelty_copy_scores_a_single_unseen_row(fitted, scaled):
    row = scaled.iloc[[0]]
    assert fitted.score(row).shape == (1,)
    assert fitted.flag(row).shape == (1,)


def test_live_flag_rate_on_training_data_is_near_contamination(fitted, scaled):
    rate = fitted.flag(scaled).mean()
    assert 0.03 <= rate <= 0.07


def test_lof_score_direction(fitted, unscaled):
    # Verify that negating negative_outlier_factor_ is correct: flagged rows
    # should have higher UtilizationRatio than normal rows. BaseDetector
    # always flags exactly the top contamination fraction by count, so
    # fit_flags_.sum() == 126 holds even if the negation were dropped (it
    # would just flag the 126 most normal rows instead). This test would fail
    # if the minus sign were removed from _training_scores, because then the
    # flagged rows would be the least anomalous, not the most.
    flagged_util = unscaled.loc[fitted.fit_flags_ == 1, "UtilizationRatio"].mean()
    normal_util = unscaled.loc[fitted.fit_flags_ == 0, "UtilizationRatio"].mean()
    assert flagged_util > normal_util * 3  # Flagged should be at least 3x higher
