"""Loads and validates the raw transaction CSV."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_COLUMNS = [
    "TransactionID",
    "AccountID",
    "TransactionAmount",
    "TransactionDate",
    "TransactionType",
    "Location",
    "DeviceID",
    "IP Address",
    "MerchantID",
    "Channel",
    "CustomerAge",
    "CustomerOccupation",
    "TransactionDuration",
    "LoginAttempts",
    "AccountBalance",
    "PreviousTransactionDate",
]

DATE_COLUMNS = ["TransactionDate", "PreviousTransactionDate"]


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    """Read the raw CSV, verify its schema, and parse the date columns."""
    df = pd.read_csv(csv_path)

    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}")

    df = df[RAW_COLUMNS].copy()
    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col], errors="raise")

    return df.reset_index(drop=True)
