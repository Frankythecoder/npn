import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from ml.config import Config
from ml.data.loader import load_raw
from ml.detectors.isolation_forest import IsolationForestDetector
from ml.detectors.one_class_svm import OneClassSVMDetector
from ml.features.engineer import build_training_frame


@pytest.fixture(scope="module")
def scaled():
    X, _, _ = build_training_frame(load_raw(Config.load().get("data.csv_path")))
    return pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)


def test_isolation_forest_flags_the_contamination_rate(scaled):
    det = IsolationForestDetector(contamination=0.05, n_estimators=200, random_state=42)
    det.fit(scaled)
    assert det.fit_flags_.sum() == 126
    assert det.name == "isolation_forest"
    assert det.live_scorable is True
    assert det.scaler == "standard"


def test_one_class_svm_flags_the_contamination_rate(scaled):
    det = OneClassSVMDetector(contamination=0.05, kernel="rbf", gamma="scale", nu=0.05)
    det.fit(scaled)
    assert det.fit_flags_.sum() == 126
    assert det.name == "one_class_svm"
    assert det.live_scorable is True


@pytest.mark.parametrize(
    "factory",
    [
        lambda: IsolationForestDetector(
            contamination=0.05, n_estimators=200, random_state=42
        ),
        lambda: OneClassSVMDetector(
            contamination=0.05, kernel="rbf", gamma="scale", nu=0.05
        ),
    ],
)
def test_scores_a_single_unseen_row(scaled, factory):
    det = factory().fit(scaled)
    row = scaled.iloc[[0]]
    assert det.score(row).shape == (1,)
    assert det.flag(row).shape == (1,)


def test_isolation_forest_is_deterministic(scaled):
    a = IsolationForestDetector(
        contamination=0.05, n_estimators=200, random_state=42
    ).fit(scaled)
    b = IsolationForestDetector(
        contamination=0.05, n_estimators=200, random_state=42
    ).fit(scaled)
    assert np.array_equal(a.fit_flags_, b.fit_flags_)
