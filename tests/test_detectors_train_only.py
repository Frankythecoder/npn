import warnings

import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from ml.config import Config
from ml.data.loader import load_raw
from ml.detectors.gmm import GMMDetector
from ml.detectors.kmeans import KMeansDetector
from ml.detectors.mcd import MCDDetector
from ml.features.engineer import CONTINUOUS_COLUMNS, build_training_frame


@pytest.fixture(scope="module")
def frames():
    X, _, _ = build_training_frame(load_raw(Config.load().get("data.csv_path")))
    full = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    cont_raw = X[CONTINUOUS_COLUMNS]
    cont = pd.DataFrame(
        StandardScaler().fit_transform(cont_raw), columns=CONTINUOUS_COLUMNS
    )
    return full, cont


@pytest.fixture(scope="module")
def unscaled():
    X, _, _ = build_training_frame(load_raw(Config.load().get("data.csv_path")))
    return X


def test_all_three_are_marked_train_only():
    for cls in (MCDDetector, GMMDetector, KMeansDetector):
        assert cls.live_scorable is False, cls.__name__


def test_mcd_uses_the_continuous_view():
    assert MCDDetector.view == "continuous"
    assert MCDDetector.scaler == "continuous"


def test_mcd_fits_cleanly_on_the_continuous_view(frames):
    """On the full 19 columns MCD fails to converge (spec 2.2)."""
    _, cont = frames
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        det = MCDDetector(contamination=0.05, random_state=42).fit(cont)
    convergence = [w for w in caught if "Determinant has increased" in str(w.message)]
    assert convergence == [], "MCD must converge without determinant warnings"
    assert det.fit_flags_.sum() == 126


def test_gmm_converges_and_flags_the_rate(frames):
    full, _ = frames
    det = GMMDetector(
        contamination=0.05, n_components=5, covariance_type="full", random_state=42
    ).fit(full)
    assert det._model.converged_ is True
    assert det.fit_flags_.sum() == 126


def test_kmeans_flags_the_rate(frames):
    full, _ = frames
    det = KMeansDetector(
        contamination=0.05, n_clusters=8, n_init=10, random_state=42
    ).fit(full)
    assert det.fit_flags_.sum() == 126
    assert (det.score(full) >= 0).all()


def test_all_three_score_a_single_row(frames):
    full, cont = frames
    pairs = [
        (MCDDetector(contamination=0.05, random_state=42), cont),
        (
            GMMDetector(
                contamination=0.05,
                n_components=5,
                covariance_type="full",
                random_state=42,
            ),
            full,
        ),
        (
            KMeansDetector(
                contamination=0.05, n_clusters=8, n_init=10, random_state=42
            ),
            full,
        ),
    ]
    for det, frame in pairs:
        det.fit(frame)
        assert det.score(frame.iloc[[0]]).shape == (1,), det.name


@pytest.mark.parametrize(
    "factory,view",
    [
        (
            lambda: MCDDetector(contamination=0.05, random_state=42),
            "cont",
        ),
        (
            lambda: GMMDetector(
                contamination=0.05,
                n_components=5,
                covariance_type="full",
                random_state=42,
            ),
            "full",
        ),
        (
            lambda: KMeansDetector(
                contamination=0.05, n_clusters=8, n_init=10, random_state=42
            ),
            "full",
        ),
    ],
)
def test_score_direction(frames, unscaled, factory, view):
    # Verify each detector's sign convention is correct: flagged rows should have
    # higher UtilizationRatio than normal rows. fit_flags_.sum() == 126 alone proves
    # nothing about direction -- BaseDetector flags whatever is above the 95th
    # percentile regardless of sign, so a negated score would still flag 126 rows,
    # just the 126 most *normal* ones instead. This test would fail if any of the
    # sign conventions (MCD/KMeans/PCA unnegated, GMM negated) were flipped.
    full, cont = frames
    frame = cont if view == "cont" else full
    det = factory().fit(frame)
    flagged_util = unscaled.loc[det.fit_flags_ == 1, "UtilizationRatio"].mean()
    normal_util = unscaled.loc[det.fit_flags_ == 0, "UtilizationRatio"].mean()
    assert flagged_util > normal_util
