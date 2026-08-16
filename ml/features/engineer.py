"""Feature engineering shared by training and serving.

Both entry points emit the same 19 columns in the same frozen order. The column
order is persisted in FeatureArtifacts so any drift between training and serving
fails loudly instead of silently misaligning values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

CATEGORICAL_LEVELS: dict[str, list[str]] = {
    "TransactionType": ["Credit", "Debit"],
    "Channel": ["ATM", "Branch", "Online"],
    "CustomerOccupation": ["Doctor", "Engineer", "Retired", "Student"],
}

NUMERIC_COLUMNS = [
    "TransactionAmount",
    "CustomerAge",
    "TransactionDuration",
    "LoginAttempts",
    "AccountBalance",
    "TimeSinceLastTx_Hours",
    "DailyAccountVolume",
    "UtilizationRatio",
    "DailyDeviceVelocity",
    "Location_Freq",
]

ONE_HOT_COLUMNS = [
    f"{prefix}_{level}"
    for prefix, levels in CATEGORICAL_LEVELS.items()
    for level in levels
]

# The frozen 19, built explicitly rather than relying on get_dummies ordering.
FEATURE_COLUMNS = NUMERIC_COLUMNS + ONE_HOT_COLUMNS

# MCD is a Gaussian elliptical estimator and cannot be fitted on binary dummies
# or on the 95-98% single-valued counters (spec 2.2). It is the only consumer.
CONTINUOUS_COLUMNS = [
    "TransactionAmount",
    "CustomerAge",
    "TransactionDuration",
    "AccountBalance",
    "TimeSinceLastTx_Hours",
    "UtilizationRatio",
    "Location_Freq",
]


@dataclass
class FeatureArtifacts:
    """Training-derived state a single incoming row cannot supply."""

    location_freq: dict[str, int]
    location_freq_default: int
    feature_columns: list[str]
    continuous_columns: list[str]
    categorical_levels: dict[str, list[str]]
    time_since_last_tx_median: float


@dataclass
class ProfileStore:
    """Per-account and per-device history for the three history-dependent features."""

    account_last_tx: dict[str, pd.Timestamp] = field(default_factory=dict)
    account_day_counts: dict[tuple[str, str], int] = field(default_factory=dict)
    device_day_counts: dict[tuple[str, str], int] = field(default_factory=dict)

    @staticmethod
    def _day_key(txn_date: pd.Timestamp) -> str:
        return pd.Timestamp(txn_date).date().isoformat()

    def account_day_count(self, account_id: str, txn_date: pd.Timestamp) -> int:
        return self.account_day_counts.get((account_id, self._day_key(txn_date)), 0)

    def device_day_count(self, device_id: str, txn_date: pd.Timestamp) -> int:
        return self.device_day_counts.get((device_id, self._day_key(txn_date)), 0)

    def gap_hours(
        self, account_id: str, txn_date: pd.Timestamp, default: float
    ) -> tuple[float, bool]:
        """Return (gap in hours, seen_before). Falls back to `default` if unseen."""
        previous = self.account_last_tx.get(account_id)
        if previous is None:
            return default, False
        delta = (pd.Timestamp(txn_date) - previous).total_seconds() / 3600.0
        return delta, True

    def observe(
        self, account_id: str, device_id: str, txn_date: pd.Timestamp
    ) -> None:
        """Record a transaction so subsequent scores see it as history."""
        txn_date = pd.Timestamp(txn_date)
        day = self._day_key(txn_date)
        previous = self.account_last_tx.get(account_id)
        if previous is None or txn_date > previous:
            self.account_last_tx[account_id] = txn_date
        self.account_day_counts[(account_id, day)] = (
            self.account_day_counts.get((account_id, day), 0) + 1
        )
        self.device_day_counts[(device_id, day)] = (
            self.device_day_counts.get((device_id, day), 0) + 1
        )


def _chronological_running_count(
    df: pd.DataFrame, key: str, day: pd.Series
) -> pd.Series:
    """Count of prior same-day transactions for `key`, inclusive of this row.

    A running count, not a whole-day total: at the first transaction of a day it
    is not knowable that a second will follow. This also makes the training value
    identical to the serving rule in spec 3.3.
    """
    ordering = df.sort_values([key, "TransactionDate"]).index
    counts = (
        df.loc[ordering]
        .assign(_day=day.loc[ordering])
        .groupby([key, "_day"])
        .cumcount()
        + 1
    )
    return counts.reindex(df.index)


def build_training_frame(
    df_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, FeatureArtifacts, ProfileStore]:
    """Engineer the full 19-column training matrix and the artifacts serving needs."""
    df = df_raw.copy()
    day = df["TransactionDate"].dt.date

    # Gap to this account's own previous transaction. PreviousTransactionDate is an
    # ingest timestamp and is deliberately unused (spec 2.1).
    ordering = df.sort_values(["AccountID", "TransactionDate"]).index
    gap = (
        df.loc[ordering]
        .groupby("AccountID")["TransactionDate"]
        .diff()
        .dt.total_seconds()
        / 3600.0
    ).reindex(df.index)
    gap_median = float(gap.median())
    df["TimeSinceLastTx_Hours"] = gap.fillna(gap_median)

    df["DailyAccountVolume"] = _chronological_running_count(df, "AccountID", day)
    df["DailyDeviceVelocity"] = _chronological_running_count(df, "DeviceID", day)
    df["UtilizationRatio"] = df["TransactionAmount"] / df["AccountBalance"]

    location_freq = df["Location"].value_counts().to_dict()
    df["Location_Freq"] = df["Location"].map(location_freq)

    frame = pd.DataFrame(index=df.index)
    for col in NUMERIC_COLUMNS:
        frame[col] = df[col].astype(float)
    for prefix, levels in CATEGORICAL_LEVELS.items():
        for level in levels:
            frame[f"{prefix}_{level}"] = (df[prefix] == level).astype(int)
    frame = frame[FEATURE_COLUMNS]

    artifacts = FeatureArtifacts(
        location_freq={str(k): int(v) for k, v in location_freq.items()},
        location_freq_default=int(min(location_freq.values())),
        feature_columns=list(FEATURE_COLUMNS),
        continuous_columns=list(CONTINUOUS_COLUMNS),
        categorical_levels={k: list(v) for k, v in CATEGORICAL_LEVELS.items()},
        time_since_last_tx_median=gap_median,
    )

    profiles = ProfileStore()
    for row in df.sort_values("TransactionDate").itertuples(index=False):
        profiles.observe(row.AccountID, row.DeviceID, row.TransactionDate)

    return frame, artifacts, profiles
