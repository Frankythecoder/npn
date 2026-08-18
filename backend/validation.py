"""Row validation applied before a transaction reaches the model.

THE LINE THIS DRAWS
-------------------
The validator rejects the impossible. The model judges the unusual. Every bound
below is a physical or logical limit, never the training range -- a validator
tuned to the training data would delete exactly the fraud the ensemble exists to
find, and it would do it silently.

Two concrete cases, both measured on original.csv:

  * 119 of 2,512 training rows spend more than the account balance. That is not
    a broken row, it is the UtilizationRatio signal. Not validated.
  * LoginAttempts tops out at 5 in training, and "credential stuffing" is a
    built-in demo preset. The cap here is 100, far above anything real, so an
    actual attack survives to be scored.

So the numbers are deliberately loose. A check earns its place only by catching
input the model cannot meaningfully score at all.

WHAT WAS ALREADY COVERED
------------------------
Nulls, unparseable numbers, short rows, a zero AccountBalance and an unparseable
TransactionDate are all rejected before this runs -- see normalise_rows and
transform_one. The null check is restated here anyway so the whole contract can
be read in one place rather than inferred from three.
"""
from __future__ import annotations

from typing import Any

from ml.features.engineer import CATEGORICAL_LEVELS

# --- bounds ---------------------------------------------------------------
# Left column is the floor, right the ceiling; both inclusive unless the check
# below says otherwise. Training ranges are quoted only to show the headroom.

MIN_AMOUNT = 0.0            # exclusive. training min 0.26
MIN_BALANCE = 0.0           # exclusive. engineer.py already refuses exactly 0
MIN_AGE, MAX_AGE = 18, 120  # training 18-80; 120 is the human limit
MIN_DURATION, MAX_DURATION = 0, 86_400   # training 10-300s; ceiling is one day
MIN_LOGINS, MAX_LOGINS = 0, 100          # training 1-5; see the note above

# Identifier hygiene. Deliberately permissive: TransactionID is a label that is
# never scored, so rejecting an otherwise-valid transaction over a cosmetic id
# costs a real row and buys nothing. This catches empty, whitespace-bearing and
# absurd ids, not unfamiliar formats.
MIN_ID_LENGTH, MAX_ID_LENGTH = 4, 64
MAX_IDENTITY_LENGTH = 64

NUMERIC_FIELDS = (
    "TransactionAmount",
    "AccountBalance",
    "CustomerAge",
    "TransactionDuration",
    "LoginAttempts",
)


def _bad_identifier(value: Any, label: str, minimum: int) -> str | None:
    """Shared shape check for the id-ish columns."""
    text = "" if value is None else str(value)
    if not text.strip():
        return f"{label} is empty"
    if any(ch.isspace() for ch in text):
        return f"{label} contains whitespace: {text!r}"
    if len(text) < minimum:
        return f"{label} is too short at {len(text)} characters (minimum {minimum})"
    if len(text) > MAX_ID_LENGTH:
        return f"{label} is too long at {len(text)} characters (maximum {MAX_ID_LENGTH})"
    return None


def validate_payload(payload: dict) -> str | None:
    """Return a rejection reason, or None if the row can be scored.

    Plain if/else on a finished payload: every scoring column is present and
    already coerced to its type by this point, so nothing here has to re-parse
    or guess.
    """
    # --- nothing may be null ------------------------------------------------
    for field in NUMERIC_FIELDS:
        if payload.get(field) is None:
            return f"{field} is null"

    # --- numeric bounds -----------------------------------------------------
    amount = float(payload["TransactionAmount"])
    if amount <= MIN_AMOUNT:
        return f"TransactionAmount must be positive, got {amount:g}"

    balance = float(payload["AccountBalance"])
    if balance <= MIN_BALANCE:
        return f"AccountBalance must be positive, got {balance:g}"

    age = float(payload["CustomerAge"])
    if not (MIN_AGE <= age <= MAX_AGE):
        return f"CustomerAge {age:g} is outside {MIN_AGE}-{MAX_AGE}"

    duration = float(payload["TransactionDuration"])
    if not (MIN_DURATION <= duration <= MAX_DURATION):
        return (
            f"TransactionDuration {duration:g} is outside "
            f"{MIN_DURATION}-{MAX_DURATION} seconds"
        )

    logins = float(payload["LoginAttempts"])
    if not (MIN_LOGINS <= logins <= MAX_LOGINS):
        return f"LoginAttempts {logins:g} is outside {MIN_LOGINS}-{MAX_LOGINS}"

    # --- categorical levels the model actually has an indicator for ---------
    # An unrecognised level is not merely unusual: transform_one sets every
    # indicator in that group to 0, which is a feature vector no training row
    # ever had. Scoring it produces a number with nothing behind it.
    for column, levels in CATEGORICAL_LEVELS.items():
        value = payload.get(column)
        if value not in levels:
            return f"{column} {value!r} is not one of {', '.join(levels)}"

    # --- identifier hygiene -------------------------------------------------
    reason = _bad_identifier(payload.get("TransactionID"), "TransactionID", MIN_ID_LENGTH)
    if reason is not None:
        return reason

    for column in ("AccountID", "DeviceID"):
        reason = _bad_identifier(payload.get(column), column, 1)
        if reason is not None:
            return reason

    return None
