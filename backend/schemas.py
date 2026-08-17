"""Request models.

Field names mirror the raw CSV columns exactly, because ml.features.engineer's
transform_one() reads them by those names. Renaming here would mean translating
in two places.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TransactionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    TransactionID: str = Field(default="")
    AccountID: str
    DeviceID: str
    Location: str
    TransactionDate: str
    TransactionAmount: float
    AccountBalance: float
    CustomerAge: int
    TransactionDuration: int
    LoginAttempts: int
    TransactionType: str
    Channel: str
    CustomerOccupation: str


class BatchIn(BaseModel):
    transactions: list[TransactionIn] = Field(min_length=1, max_length=500)


class BatchOut(BaseModel):
    results: list[dict]


class CsvBatchIn(BaseModel):
    """One chunk of an uploaded CSV, already parsed into cells by the dashboard.

    Rows are positional rather than keyed by column, so a 500-row chunk carries
    the header once instead of once per row. Unlike BatchIn this deliberately
    does not validate cells: a partial column set and a bad row are both normal
    here, and both are reported per row rather than failing the whole request.
    """

    model_config = ConfigDict(extra="forbid")

    columns: list[str] = Field(min_length=1)
    rows: list[list[str | None]] = Field(min_length=1, max_length=500)
    # The line number of the first row within the original file. The dashboard
    # uploads in chunks, so without it a rejection in chunk five would name a
    # line number that does not exist in the file the user is looking at.
    start_row: int = Field(default=2, ge=1)
    # Identifies the file a chunk belongs to. Chunks sharing an id share one
    # profile store, so an account's history within an upload is the history of
    # that file rather than of the long-running service. Empty means an unshared
    # store for this request alone.
    upload_id: str = Field(default="", max_length=64)
