import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from ml.config import Config
from ml.data.loader import load_raw
from ml.detectors.isolation_forest import IsolationForestDetector
from ml.features.engineer import build_training_frame
from ml.storage.artifacts import ArtifactBundle, load_bundle, save_bundle


@pytest.fixture(scope="module")
def bundle():
    X, artifacts, profiles = build_training_frame(
        load_raw(Config.load().get("data.csv_path"))
    )
    scaler = StandardScaler().fit(X)
    scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
    det = IsolationForestDetector(
        contamination=0.05, n_estimators=200, random_state=42
    ).fit(scaled)
    return ArtifactBundle(
        manifest={"version": "test", "rate_table": {"isolation_forest": 0.0502}},
        scalers={"standard": scaler},
        feature_artifacts=artifacts,
        profile_store=profiles,
        detectors={"isolation_forest": det},
        surrogate=None,
        explainer_state={"feature_percentiles": {"TransactionAmount": [1.0, 2.0]}},
    )


def test_round_trip_preserves_the_manifest(tmp_path, bundle):
    save_bundle(bundle, tmp_path)
    loaded = load_bundle(tmp_path)
    assert loaded.manifest == bundle.manifest


def test_round_trip_preserves_feature_artifacts(tmp_path, bundle):
    save_bundle(bundle, tmp_path)
    loaded = load_bundle(tmp_path)
    assert loaded.feature_artifacts.feature_columns == (
        bundle.feature_artifacts.feature_columns
    )
    assert loaded.feature_artifacts.location_freq_default == (
        bundle.feature_artifacts.location_freq_default
    )
    assert loaded.feature_artifacts.time_since_last_tx_median == pytest.approx(
        bundle.feature_artifacts.time_since_last_tx_median
    )


def test_round_trip_preserves_the_profile_store(tmp_path, bundle):
    save_bundle(bundle, tmp_path)
    loaded = load_bundle(tmp_path)
    assert loaded.profile_store.account_last_tx == bundle.profile_store.account_last_tx
    assert (
        loaded.profile_store.account_day_counts
        == bundle.profile_store.account_day_counts
    )


def test_round_trip_preserves_detector_thresholds(tmp_path, bundle):
    save_bundle(bundle, tmp_path)
    loaded = load_bundle(tmp_path)
    original = bundle.detectors["isolation_forest"]
    restored = loaded.detectors["isolation_forest"]
    assert restored.threshold_ == pytest.approx(original.threshold_)
    assert np.array_equal(restored.train_scores_, original.train_scores_)
    assert np.array_equal(restored.fit_flags_, original.fit_flags_)


def test_restored_detector_scores_identically(tmp_path, bundle):
    X, _, _ = build_training_frame(load_raw(Config.load().get("data.csv_path")))
    scaled = pd.DataFrame(
        bundle.scalers["standard"].transform(X), columns=X.columns
    )
    save_bundle(bundle, tmp_path)
    loaded = load_bundle(tmp_path)
    row = scaled.iloc[[0]]
    assert loaded.detectors["isolation_forest"].score(row) == pytest.approx(
        bundle.detectors["isolation_forest"].score(row)
    )


def test_manifest_is_human_readable_json(tmp_path, bundle):
    import json

    save_bundle(bundle, tmp_path)
    text = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    assert json.loads(text)["version"] == "test"


def test_load_rejects_a_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path / "does-not-exist")
