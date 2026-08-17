"""Turns uploaded CSV rows into scorer payloads.

An uploaded file is allowed to be a subset of original.csv rather than a copy of
it, so this module answers two questions the scorer itself cannot: whether a file
is a transaction file at all, and what to put in the columns it did not supply.

The fill values come from the artifact bundle already loaded for scoring -- the
sorted per-column training values persisted for SHAP percentile lookups double as
a training distribution, so no second data source and no retrain is needed.

Identity columns are synthesised rather than filled. A fabricated AccountID would
not approximate an account's history, it would invent one, and the three
history-derived features would then describe a customer who does not exist. A
synthetic id instead routes the row through transform_one's documented
unseen-account path, which says so in the row's warnings.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ml.features.engineer import RAW_INPUT_FIELDS

# The quantitative signal the detectors ride on. A file carrying none of these is
# not a transaction file, whatever else it happens to contain.
CRUCIAL_COLUMNS = (
    "TransactionAmount",
    "AccountBalance",
    "CustomerAge",
    "TransactionDuration",
    "LoginAttempts",
)

# The crucial five are exactly the numeric inputs, so parsing splits on the same
# boundary the gate does.
NUMERIC_COLUMNS = CRUCIAL_COLUMNS
CATEGORICAL_COLUMNS = ("TransactionType", "Channel", "CustomerOccupation")
IDENTITY_COLUMNS = ("AccountID", "DeviceID", "TransactionDate")
FILLABLE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + ("Location",)

# Carried through as a label when present; never scored, so a blank one is not a
# reason to reject a row.
LABEL_COLUMN = "TransactionID"

# Columns of original.csv the scorer has never read. Named explicitly so the
# frontend can report them as deliberately ignored rather than unrecognised.
IGNORED_COLUMNS = ("IP Address", "MerchantID", "PreviousTransactionDate")

NULL_TOKENS = frozenset({"", "na", "n/a", "nan", "null", "none"})

# Every scoring column is either fillable or synthesisable. If a future feature
# adds a raw input, this fails at import rather than silently leaving the new
# column absent from uploaded rows.
assert set(FILLABLE_COLUMNS) | set(IDENTITY_COLUMNS) == set(RAW_INPUT_FIELDS), (
    "csvingest column groups have drifted from RAW_INPUT_FIELDS"
)


class MissingCrucialColumns(ValueError):
    """Raised when a file carries none of CRUCIAL_COLUMNS."""


@dataclass
class NormalisedRow:
    """One accepted row: the scorer payload, plus what had to be invented for it."""

    row: int
    payload: dict[str, Any]
    warnings: list[str]


def _median(values: list[float]) -> float:
    ordered = [float(v) for v in values]
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _mean(values: list[float]) -> float:
    return sum(float(v) for v in values) / len(values)


def defaults_from_bundle(bundle: Any) -> dict[str, Any]:
    """Derive a fill value for every fillable column from the loaded bundle.

    Numerics take the training median. A categorical's level is one-hot encoded,
    so the column's mean over the training set is that level's prevalence and the
    highest mean is the mode. Location is not a feature -- Location_Freq is -- so
    its default comes from the frequency table instead.
    """
    percentiles = bundle.explainer_state["feature_percentiles"]
    artifacts = bundle.feature_artifacts

    defaults: dict[str, Any] = {
        column: _median(percentiles[column]) for column in NUMERIC_COLUMNS
    }

    for prefix, levels in artifacts.categorical_levels.items():
        defaults[prefix] = max(
            levels, key=lambda level: _mean(percentiles[f"{prefix}_{level}"])
        )

    freq = artifacts.location_freq
    defaults["Location"] = max(freq, key=freq.get) if freq else ""

    return defaults


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in NULL_TOKENS


def _fill_warning(column: str, value: Any) -> str:
    return (
        f"{column} missing from the file: filled with the training default "
        f"({value!r})"
    )


def normalise_rows(
    columns: list[str],
    rows: list[list[Any]],
    defaults: dict[str, Any],
    start_row: int = 2,
) -> tuple[list[NormalisedRow], list[dict]]:
    """Convert raw CSV cells into scorer payloads.

    `start_row` is the line number of the first row in `rows` within the original
    file -- the caller passes the real offset when uploading in chunks, so a
    rejection always names the line the user can go and look at.

    Returns (accepted, rejected). A bad row is rejected on its own; only a file
    with no crucial column at all raises.
    """
    if not any(column in CRUCIAL_COLUMNS for column in columns):
        raise MissingCrucialColumns(
            "the file must contain at least one of "
            + ", ".join(CRUCIAL_COLUMNS)
            + f"; found {', '.join(columns) or 'no columns'}"
        )

    absent_fillable = [c for c in FILLABLE_COLUMNS if c not in columns]
    absent_identity = [c for c in IDENTITY_COLUMNS if c not in columns]
    supplied_scoring = [c for c in RAW_INPUT_FIELDS if c in columns]

    accepted: list[NormalisedRow] = []
    rejected: list[dict] = []

    for offset, raw_row in enumerate(rows):
        line = start_row + offset

        if len(raw_row) != len(columns):
            rejected.append(
                {
                    "row": line,
                    "reason": (
                        f"expected {len(columns)} columns, found {len(raw_row)}"
                    ),
                }
            )
            continue

        cells = dict(zip(columns, raw_row))
        payload: dict[str, Any] = {}
        reason: str | None = None

        for column in supplied_scoring:
            value = cells[column]
            if _is_null(value):
                reason = f"{column} is empty"
                break
            text = str(value).strip()
            if column in NUMERIC_COLUMNS:
                try:
                    payload[column] = float(text)
                except ValueError:
                    reason = f"{column} is not a number: {text!r}"
                    break
            else:
                payload[column] = text

        if reason is not None:
            rejected.append({"row": line, "reason": reason})
            continue

        warnings: list[str] = []

        for column in absent_fillable:
            payload[column] = defaults[column]
            warnings.append(_fill_warning(column, defaults[column]))

        for column in absent_identity:
            if column == "TransactionDate":
                payload[column] = datetime.now().isoformat(sep=" ", timespec="seconds")
                warnings.append(
                    "TransactionDate missing from the file: scored against the "
                    "current time"
                )
            else:
                prefix = "A" if column == "AccountID" else "D"
                payload[column] = f"CSV-{prefix}{line}"
                warnings.append(
                    f"{column} missing from the file: this row was treated as a "
                    "new one, so its history-derived features fall back to "
                    "training defaults"
                )

        label = cells.get(LABEL_COLUMN)
        payload[LABEL_COLUMN] = (
            str(label).strip() if not _is_null(label) else f"row-{line}"
        )

        accepted.append(NormalisedRow(row=line, payload=payload, warnings=warnings))

    return accepted, rejected
