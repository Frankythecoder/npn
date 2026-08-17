"""Demo endpoints.

/demo/inject is a convenience wrapper, not a second scoring path: it fills absent
fields from a named preset and then delegates to exactly the same code as /score.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from backend import deps
from backend.presets import PRESETS
from backend.schemas import TransactionIn

router = APIRouter(prefix="/demo", tags=["demo"])


class InjectIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preset: str = "normal"
    overrides: dict = Field(default_factory=dict)


@router.get("/presets")
def presets() -> dict:
    return {
        "presets": [
            {
                "name": name,
                "label": preset["label"],
                "description": preset["description"],
                "fields": preset["fields"],
            }
            for name, preset in PRESETS.items()
        ]
    }


@router.post("/inject")
def inject(payload: InjectIn) -> dict:
    preset = PRESETS.get(payload.preset)
    if preset is None:
        raise HTTPException(
            status_code=422,
            detail=f"unknown preset {payload.preset!r}; choose one of {list(PRESETS)}",
        )

    fields = {**preset["fields"], **payload.overrides}
    fields.setdefault("TransactionID", f"DEMO-{uuid.uuid4().hex[:8].upper()}")

    try:
        transaction = TransactionIn(**fields)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        result = deps.get_scorer().score_transaction(transaction.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    deps.get_log().append(result)
    return result
