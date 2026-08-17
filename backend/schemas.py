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
