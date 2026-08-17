"""Scoring endpoints.

None of these compute anything. Each builds a plain dict and hands it to the ml
package's score_transaction(), so the API cannot drift from the training path.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend import deps
from backend.config import Settings
from backend.schemas import BatchIn, BatchOut, TransactionIn

router = APIRouter(tags=["scoring"])


def _score_one(payload: TransactionIn) -> dict:
    try:
        result = deps.get_scorer().score_transaction(payload.model_dump())
    except ValueError as exc:
        # The ml layer rejects impossible inputs (a zero balance, a missing field).
        # That is a client error, not a server fault.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    deps.get_log().append(result)
    return result


@router.post("/score")
def score(payload: TransactionIn) -> dict:
    return _score_one(payload)


@router.post("/batch-score", response_model=BatchOut)
def batch_score(payload: BatchIn) -> BatchOut:
    return BatchOut(results=[_score_one(item) for item in payload.transactions])


@router.get("/transactions/recent")
def recent(limit: int | None = Query(default=None, ge=1)) -> list[dict]:
    settings = Settings.from_env()
    effective = limit if limit is not None else settings.recent_limit_default
    if effective > settings.recent_limit_max:
        raise HTTPException(
            status_code=422,
            detail=f"limit exceeds maximum of {settings.recent_limit_max}",
        )
    return deps.get_log().recent(effective)
