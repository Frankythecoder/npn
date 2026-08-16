import numpy as np
import pandas as pd
import pytest

from ml.config import Config
from ml.data.loader import load_raw
from ml.features.engineer import (
    FEATURE_COLUMNS,
    ProfileStore,
    build_training_frame,
    transform_one,
)


@pytest.fixture(scope="module")
def raw():
    return load_raw(Config.load().get("data.csv_path"))


@pytest.fixture(scope="module")
def built(raw):
    return build_training_frame(raw)


def _profiles_up_to(raw: pd.DataFrame, cutoff: pd.Timestamp) -> ProfileStore:
    """A profile store holding only transactions strictly before `cutoff`."""
    store = ProfileStore()
    earlier = raw[raw["TransactionDate"] < cutoff].sort_values("TransactionDate")
    for row in earlier.itertuples(index=False):
        store.observe(row.AccountID, row.DeviceID, row.TransactionDate)
    return store


def test_transform_one_returns_the_frozen_columns(raw, built):
    _, artifacts, profiles = built
    frame, _ = transform_one(raw.iloc[0].to_dict(), artifacts, profiles)
    assert frame.shape == (1, 19)
    assert list(frame.columns) == FEATURE_COLUMNS


def test_train_serve_parity_on_a_repeat_account(raw, built):
    """The critical test: rewind the profile store and reproduce a training row."""
    X, artifacts, _ = built
    repeat_accounts = raw["AccountID"].value_counts()
    account = repeat_accounts[repeat_accounts >= 2].index[0]
    rows = raw[raw["AccountID"] == account].sort_values("TransactionDate")
    target = rows.iloc[1]

    profiles = _profiles_up_to(raw, target["TransactionDate"])
    frame, warnings = transform_one(target.to_dict(), artifacts, profiles)

    expected = X.loc[target.name]
    for col in FEATURE_COLUMNS:
        assert frame.iloc[0][col] == pytest.approx(expected[col], rel=1e-9), col
    assert warnings == []


def test_train_serve_parity_across_many_rows(raw, built):
    X, artifacts, _ = built
    sample = raw.sample(n=25, random_state=42)
    for _, target in sample.iterrows():
        profiles = _profiles_up_to(raw, target["TransactionDate"])
        frame, _ = transform_one(target.to_dict(), artifacts, profiles)
        expected = X.loc[target.name]
        assert np.allclose(
            frame.iloc[0].to_numpy(dtype=float),
            expected.to_numpy(dtype=float),
            rtol=1e-9,
        ), f"parity failed for row {target.name}"


def test_unseen_account_uses_the_median_gap_and_warns(raw, built):
    _, artifacts, profiles = built
    txn = raw.iloc[0].to_dict()
    txn["AccountID"] = "AC99999"
    frame, warnings = transform_one(txn, artifacts, profiles)
    assert frame.iloc[0]["TimeSinceLastTx_Hours"] == pytest.approx(
        artifacts.time_since_last_tx_median
    )
    assert any("unseen account" in w for w in warnings)


def test_negative_gap_is_clamped_to_the_median_and_warns(raw, built):
    """A transaction predating the account's last known activity must not
    produce a negative gap never seen in training (spec 2.1) - it gets the
    same median-fill treatment as an unseen account."""
    _, artifacts, profiles = built
    txn = raw.iloc[0].to_dict()
    last_tx = profiles.account_last_tx[txn["AccountID"]]
    txn["TransactionDate"] = last_tx - pd.Timedelta(days=10)
    frame, warnings = transform_one(txn, artifacts, profiles)
    assert frame.iloc[0]["TimeSinceLastTx_Hours"] == pytest.approx(
        artifacts.time_since_last_tx_median
    )
    assert any("predates" in w for w in warnings)


def test_unseen_city_uses_the_default_frequency_and_warns(raw, built):
    _, artifacts, profiles = built
    txn = raw.iloc[0].to_dict()
    txn["Location"] = "Atlantis"
    frame, warnings = transform_one(txn, artifacts, profiles)
    assert frame.iloc[0]["Location_Freq"] == artifacts.location_freq_default
    assert any("unseen location" in w for w in warnings)


def test_unseen_categorical_level_yields_all_zeros_and_warns(raw, built):
    _, artifacts, profiles = built
    txn = raw.iloc[0].to_dict()
    txn["Channel"] = "Carrier Pigeon"
    frame, warnings = transform_one(txn, artifacts, profiles)
    assert frame.iloc[0][["Channel_ATM", "Channel_Branch", "Channel_Online"]].sum() == 0
    assert any("unseen Channel" in w for w in warnings)


def test_daily_counts_are_self_inclusive(raw, built):
    """A lone transaction gets 1, matching how a lone training row gets 1."""
    _, artifacts, _ = built
    txn = raw.iloc[0].to_dict()
    txn["AccountID"] = "AC99999"
    txn["DeviceID"] = "D99999"
    frame, _ = transform_one(txn, artifacts, ProfileStore())
    assert frame.iloc[0]["DailyAccountVolume"] == 1
    assert frame.iloc[0]["DailyDeviceVelocity"] == 1


def test_missing_required_field_is_rejected(built):
    _, artifacts, profiles = built
    with pytest.raises(ValueError, match="missing required fields"):
        transform_one({"TransactionAmount": 10.0}, artifacts, profiles)


def test_zero_account_balance_is_rejected(raw, built):
    _, artifacts, profiles = built
    txn = raw.iloc[0].to_dict()
    txn["AccountBalance"] = 0
    with pytest.raises(ValueError, match="AccountBalance must be non-zero"):
        transform_one(txn, artifacts, profiles)
