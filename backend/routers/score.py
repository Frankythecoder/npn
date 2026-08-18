"""Scoring endpoints.

None of these compute anything. Each builds a plain dict and hands it to the ml
package's score_transaction(), so the API cannot drift from the training path.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend import deps
from backend.config import Settings
from backend.csvingest import UnusableUpload, normalise_rows
from backend.schemas import BatchIn, BatchOut, CsvBatchIn, TransactionIn

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


@router.post("/score-csv")
def score_csv(payload: CsvBatchIn) -> dict:
    """Score one chunk of an uploaded CSV.

    Distinct from /batch-score because the tolerances are opposite. There, a
    transaction is a complete, validated object and anything else is a client
    error. Here a partial column set is expected and a bad row is routine, so
    only a file carrying no crucial column at all is refused outright -- every
    other failure is reported against its line number and the rest still scores.

    Keeping the two apart is what lets TransactionIn go on forbidding unknown
    fields, which is the guarantee /score depends on.
    """
    try:
        accepted, rejected = normalise_rows(
            payload.columns,
            payload.rows,
            payload.start_row,
            # Resolved only when asked for: deriving the table touches the
            # bundle, and a complete upload should never pay for it.
            defaults=deps.get_csv_defaults() if payload.fill_missing else None,
            # Shared across this upload's chunks, so a transaction repeated
            # after a chunk boundary is still caught.
            seen_ids=deps.get_batch_seen_ids(payload.upload_id),
        )
    except UnusableUpload as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    scorer = deps.get_scorer()
    # Score against the upload's own history rather than the service's. Restored
    # in the finally so an upload cannot disturb the live feed's profile state --
    # the same swap seed.py makes, for the same reason.
    original_profiles = scorer.profiles
    scorer.profiles = deps.get_batch_profiles(payload.upload_id)

    results = []
    try:
        for row in accepted:
            try:
                result = scorer.score_transaction(row.payload)
            except ValueError as exc:
                # The ml layer rejects impossible inputs (a zero balance). One
                # such row must not take the other 499 down with it.
                rejected.append({"row": row.row, "reason": str(exc)})
                continue
            # Fills first: they explain the engineered warnings that follow.
            result["warnings"] = [*row.warnings, *result["warnings"]]
            deps.get_log().append(result)
            results.append(result)
    finally:
        scorer.profiles = original_profiles

    rejected.sort(key=lambda item: item["row"])
    return {"results": results, "rejected": rejected}


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
