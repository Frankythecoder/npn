import numpy as np
import pandas as pd
import pytest

from ml.config import Config
from ml.data.loader import load_raw
from ml.features.engineer import (
    CATEGORICAL_LEVELS,
    CONTINUOUS_COLUMNS,
    FEATURE_COLUMNS,
    build_training_frame,
)


@pytest.fixture(scope="module")
def built():
    return build_training_frame(load_raw(Config.load().get("data.csv_path")))


def test_frame_shape_and_column_order(built):
    X, _, _ = built
    assert X.shape == (2512, 19)
    assert list(X.columns) == FEATURE_COLUMNS


def test_continuous_columns_are_a_subset_of_feature_columns():
    assert len(CONTINUOUS_COLUMNS) == 7
    assert set(CONTINUOUS_COLUMNS).issubset(set(FEATURE_COLUMNS))


def test_all_values_numeric_and_finite(built):
    X, _, _ = built
    assert X.select_dtypes(include=[np.number]).shape[1] == 19
    assert np.isfinite(X.to_numpy()).all()


def test_time_since_last_tx_is_positive_and_sane(built):
    X, artifacts, _ = built
    gaps = X["TimeSinceLastTx_Hours"]
    assert (gaps > 0).all(), "the rebuilt gap must never be negative (spec 2.1)"
    assert 900 < artifacts.time_since_last_tx_median < 980


def test_daily_counts_are_running_not_whole_day(built):
    X, _, _ = built
    assert X["DailyAccountVolume"].value_counts().to_dict() == {1: 2475, 2: 37}
    assert X["DailyDeviceVelocity"].value_counts().to_dict() == {1: 2488, 2: 24}


def test_one_hot_groups_each_sum_to_one(built):
    X, _, _ = built
    for prefix, levels in CATEGORICAL_LEVELS.items():
        cols = [f"{prefix}_{lvl}" for lvl in levels]
        assert (X[cols].sum(axis=1) == 1).all()


def test_utilization_ratio_matches_definition(built):
    X, _, _ = built
    raw = load_raw(Config.load().get("data.csv_path"))
    expected = raw["TransactionAmount"] / raw["AccountBalance"]
    assert np.allclose(X["UtilizationRatio"], expected)


def test_artifacts_capture_location_frequencies(built):
    _, artifacts, _ = built
    assert len(artifacts.location_freq) == 43
    assert artifacts.location_freq_default == min(artifacts.location_freq.values())
    assert artifacts.feature_columns == FEATURE_COLUMNS


def test_profile_store_covers_every_account_and_device(built):
    _, _, profiles = built
    raw = load_raw(Config.load().get("data.csv_path"))
    assert len(profiles.account_last_tx) == raw["AccountID"].nunique()
    assert sum(profiles.account_day_counts.values()) == 2512
    assert sum(profiles.device_day_counts.values()) == 2512


def test_profile_last_tx_is_the_accounts_latest(built):
    _, _, profiles = built
    raw = load_raw(Config.load().get("data.csv_path"))
    expected = raw.groupby("AccountID")["TransactionDate"].max()
    for acct, ts in expected.items():
        assert profiles.account_last_tx[acct] == ts
