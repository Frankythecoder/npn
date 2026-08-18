"""Turns uploaded CSV rows into scorer payloads.

An upload must supply every column the scorer reads -- under its own name or
under a recognised synonym. Nothing is substituted from the training set.

That is a deliberate constraint, not an oversight. transform_one needs a value
for all nineteen engineered features, so a column an upload omits has to come
from somewhere; the only honest options are to invent it or to refuse the file,
and inventing it means scoring a transaction that is part real and part training
median. The resulting verdict looks like a judgement about the uploaded row when
it is partly a judgement about the training data. So the file is refused, and the
message names exactly which columns are missing.

Identity columns are the one exception, and they are not filled from training
data either. A fabricated AccountID would not approximate an account's history,
it would invent one, so a missing id is synthesised instead: that routes the row
through transform_one's documented unseen-account path, which says so in the
row's warnings rather than quietly borrowing another customer's behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from backend.validation import validate_payload
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
# Every column an upload has to supply itself. Nothing here has a fallback:
# the scorer needs a real value and there is nowhere honest to get one from.
REQUIRED_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + ("Location",)

# Carried through as a label when present; never scored, so a blank one is not a
# reason to reject a row.
LABEL_COLUMN = "TransactionID"

# Columns of original.csv the scorer has never read. Named explicitly so the
# frontend can report them as deliberately ignored rather than unrecognised.
IGNORED_COLUMNS = ("IP Address", "MerchantID", "PreviousTransactionDate")

NULL_TOKENS = frozenset({"", "na", "n/a", "nan", "null", "none"})

# Every header this module recognises by its real name. A name in here is never
# reinterpreted -- synonym matching is a fallback for headers the file does not
# already spell correctly, never an override of one that does.
CANONICAL_COLUMNS = frozenset(RAW_INPUT_FIELDS) | {LABEL_COLUMN} | set(IGNORED_COLUMNS)


def _match_key(name: Any) -> str:
    """Fold a header to its comparison form: case and punctuation removed.

    "Session Length (Minutes)", "session_length_minutes" and
    "SESSION-LENGTH-MINUTES" are one spelling as far as matching is concerned,
    so the table below needs one entry rather than a dozen.
    """
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


# Header synonyms, keyed by _match_key.
#
# The float is a multiplier applied to the parsed value. A plain rename is 1.0.
# A header that names a *different unit* to the one the model was trained on
# converts instead: TransactionDuration is seconds throughout training data, so
# a column calling itself minutes is scaled by 60. Reading a five-minute session
# as `5` would not be a near-miss -- it would place an ordinary row deep in the
# tail of the distribution and flag it.
#
# Deliberately conservative. A synonym earns its place by being unambiguous in a
# transaction file; anything that could plausibly mean two different columns
# ("value", "time", "id") is left out, because a wrong match is worse than no
# match -- an unmatched column now costs the upload outright, and a wrong one
# would score a real transaction against somebody else's number.
COLUMN_ALIASES: dict[str, tuple[str, float]] = {
    # TransactionAmount
    "amount": ("TransactionAmount", 1.0),
    "txnamount": ("TransactionAmount", 1.0),
    "trxamount": ("TransactionAmount", 1.0),
    "transactionvalue": ("TransactionAmount", 1.0),
    "transactionamt": ("TransactionAmount", 1.0),
    "amountusd": ("TransactionAmount", 1.0),
    # AccountBalance
    "balance": ("AccountBalance", 1.0),
    "accountbal": ("AccountBalance", 1.0),
    "acctbalance": ("AccountBalance", 1.0),
    "availablebalance": ("AccountBalance", 1.0),
    "currentbalance": ("AccountBalance", 1.0),
    # CustomerAge
    "age": ("CustomerAge", 1.0),
    "clientage": ("CustomerAge", 1.0),
    "userage": ("CustomerAge", 1.0),
    "accountholderage": ("CustomerAge", 1.0),
    # TransactionDuration -- seconds in training.
    "duration": ("TransactionDuration", 1.0),
    "durationseconds": ("TransactionDuration", 1.0),
    "durationsecs": ("TransactionDuration", 1.0),
    "durationsec": ("TransactionDuration", 1.0),
    "sessionlength": ("TransactionDuration", 1.0),
    "sessionduration": ("TransactionDuration", 1.0),
    "sessionlengthseconds": ("TransactionDuration", 1.0),
    "sessionlengthinseconds": ("TransactionDuration", 1.0),
    "elapsedseconds": ("TransactionDuration", 1.0),
    "durationminutes": ("TransactionDuration", 60.0),
    "durationmins": ("TransactionDuration", 60.0),
    "durationmin": ("TransactionDuration", 60.0),
    "sessionlengthminutes": ("TransactionDuration", 60.0),
    "sessionlengthinminutes": ("TransactionDuration", 60.0),
    "sessionlengthmins": ("TransactionDuration", 60.0),
    "elapsedminutes": ("TransactionDuration", 60.0),
    # LoginAttempts
    "logins": ("LoginAttempts", 1.0),
    "loginattemptcount": ("LoginAttempts", 1.0),
    "numloginattempts": ("LoginAttempts", 1.0),
    "signinattempts": ("LoginAttempts", 1.0),
    "authattempts": ("LoginAttempts", 1.0),
    # TransactionType
    "txntype": ("TransactionType", 1.0),
    "trxtype": ("TransactionType", 1.0),
    "debitcredit": ("TransactionType", 1.0),
    "creditdebit": ("TransactionType", 1.0),
    # Channel
    "transactionchannel": ("Channel", 1.0),
    "accesschannel": ("Channel", 1.0),
    "paymentchannel": ("Channel", 1.0),
    # CustomerOccupation
    "occupation": ("CustomerOccupation", 1.0),
    "profession": ("CustomerOccupation", 1.0),
    "customerjob": ("CustomerOccupation", 1.0),
    "jobtitle": ("CustomerOccupation", 1.0),
    # Location
    "city": ("Location", 1.0),
    "transactionlocation": ("Location", 1.0),
    "customerlocation": ("Location", 1.0),
    "transactioncity": ("Location", 1.0),
    # AccountID
    "accountnumber": ("AccountID", 1.0),
    "accountno": ("AccountID", 1.0),
    "acctid": ("AccountID", 1.0),
    "acctno": ("AccountID", 1.0),
    # DeviceID
    "device": ("DeviceID", 1.0),
    "deviceidentifier": ("DeviceID", 1.0),
    "terminalid": ("DeviceID", 1.0),
    # TransactionDate
    "transactiontimestamp": ("TransactionDate", 1.0),
    "txndate": ("TransactionDate", 1.0),
    "transactiondatetime": ("TransactionDate", 1.0),
    "bookingdate": ("TransactionDate", 1.0),
    # TransactionID -- a label, never scored.
    "txnid": ("TransactionID", 1.0),
    "transactionref": ("TransactionID", 1.0),
    "transactionreference": ("TransactionID", 1.0),
    "referenceid": ("TransactionID", 1.0),
}

# A synonym that shadows a real header would be unreachable, and one pointing at
# a column that does not exist would silently never match. Both fail at import
# rather than at upload time.
assert not (
    {alias for alias in COLUMN_ALIASES}
    & {_match_key(column) for column in CANONICAL_COLUMNS}
), "a synonym shadows a canonical column name"
assert {target for target, _ in COLUMN_ALIASES.values()} <= CANONICAL_COLUMNS, (
    "a synonym points at a column csvingest does not know"
)

# Every scoring column is either fillable or synthesisable. If a future feature
# adds a raw input, this fails at import rather than silently leaving the new
# column absent from uploaded rows.
assert set(REQUIRED_COLUMNS) | set(IDENTITY_COLUMNS) == set(RAW_INPUT_FIELDS), (
    "csvingest column groups have drifted from RAW_INPUT_FIELDS"
)


class UnusableUpload(ValueError):
    """Base for a file the ingest cannot score at all, whatever the reason."""


class MissingCrucialColumns(UnusableUpload):
    """Raised when a file carries none of CRUCIAL_COLUMNS."""


class MissingScoringColumns(UnusableUpload):
    """Raised when a file omits a column the scorer needs.

    Separate from MissingCrucialColumns because the two are different problems
    with different fixes: one file is not a transaction file at all, the other
    is one but is incomplete.
    """


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
    """Derive a fill value for every required column from the loaded bundle.

    Only reached when an upload explicitly opts into filling. Measured on the
    training set, substituting a column group moves 4-10% of ensemble verdicts,
    so this is never the default path -- see normalise_rows.

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


def _alias_warning(original: str, column: str, scale: float) -> str:
    if scale == 1.0:
        return f"{original!r} read as {column}"
    return (
        f"{original!r} read as {column}, converted to the unit the model was "
        f"trained on (x{scale:g})"
    )


def resolve_columns(
    columns: list[str],
) -> tuple[list[str], dict[str, float], list[tuple[str, str]]]:
    """Rewrite an uploaded header so recognised synonyms use their real names.

    Returns (resolved, scales, matches):

    * `resolved` is `columns` with each matched synonym replaced by the column
      it means, so everything downstream keeps working on canonical names and
      needs no knowledge of aliasing at all.
    * `scales` maps a resolved column to the multiplier that converts it into
      the unit the model was trained on. Absent means 1.0.
    * `matches` is the (header, column) pairs that were rewritten, so a row can
      report what was assumed about it.

    A canonical header always wins over a synonym for the same column, and the
    first synonym wins over any later one, so the result never depends on
    column order beyond "leftmost claims it".
    """
    resolved = list(columns)
    scales: dict[str, float] = {}
    matches: list[tuple[str, str]] = []

    claimed = {column for column in columns if column in CANONICAL_COLUMNS}

    for index, name in enumerate(columns):
        if name in CANONICAL_COLUMNS:
            continue
        entry = COLUMN_ALIASES.get(_match_key(name))
        if entry is None:
            continue
        column, scale = entry
        if column in claimed:
            # The file also carries the real column, or an earlier synonym got
            # there first. Either way this one is surplus, and guessing between
            # two candidates is worse than ignoring the second.
            continue
        claimed.add(column)
        resolved[index] = column
        if scale != 1.0:
            scales[column] = scale
        matches.append((name, column))

    return resolved, scales, matches


def normalise_rows(
    columns: list[str],
    rows: list[list[Any]],
    start_row: int = 2,
    defaults: dict[str, Any] | None = None,
    seen_ids: set[str] | None = None,
) -> tuple[list[NormalisedRow], list[dict]]:
    """Convert raw CSV cells into scorer payloads.

    `start_row` is the line number of the first row in `rows` within the original
    file -- the caller passes the real offset when uploading in chunks, so a
    rejection always names the line the user can go and look at.

    `defaults` decides what an incomplete file gets. Left as None -- the default,
    and what every caller gets unless it asks otherwise -- a file omitting a
    scoring column is refused and nothing is substituted. Passed a fill table,
    the gap is filled from the training distribution and every affected row says
    so in its warnings.

    Filling is opt-in rather than automatic because it is not cheap: measured
    over the training set, substituting the four categoricals moves 10% of
    ensemble verdicts and the five numerics 4%. A verdict on a filled row is
    partly a verdict about the training data, so the caller has to ask for it.

    `seen_ids` carries the transaction ids already accepted, so a duplicate is
    dropped rather than scored twice. Pass the set that belongs to the upload to
    dedupe across its chunks; leave it None and a fresh set dedupes within this
    request alone.

    Returns (accepted, rejected). A bad row is rejected on its own; a file that
    is not a transaction file, or is one but incomplete with no fill table,
    raises instead.
    """
    # Recognised synonyms become their real names before anything else looks at
    # the header, so the completeness check and the parser both go on seeing
    # canonical columns and neither needs to know aliasing exists. A column
    # supplied only under a synonym therefore counts as supplied.
    columns, scales, aliased = resolve_columns(columns)
    alias_warnings = [
        _alias_warning(original, column, scales.get(column, 1.0))
        for original, column in aliased
    ]

    if not any(column in CRUCIAL_COLUMNS for column in columns):
        raise MissingCrucialColumns(
            "the file must contain at least one of "
            + ", ".join(CRUCIAL_COLUMNS)
            + f"; found {', '.join(columns) or 'no columns'}"
        )

    # Checked after the crucial gate so a file of unrelated data is told it is
    # not a transaction file, rather than handed a list of nine column names.
    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing and defaults is None:
        raise MissingScoringColumns(
            "the file is missing "
            + ", ".join(missing)
            + ". Every scoring column must be supplied, under its own name or a "
            "recognised synonym -- nothing is substituted from the training data."
        )

    absent_identity = [c for c in IDENTITY_COLUMNS if c not in columns]
    supplied_scoring = [c for c in RAW_INPUT_FIELDS if c in columns]

    # A local set still dedupes within the request when the caller has no
    # upload-scoped one to offer.
    if seen_ids is None:
        seen_ids = set()

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
                    payload[column] = float(text) * scales.get(column, 1.0)
                except ValueError:
                    reason = f"{column} is not a number: {text!r}"
                    break
            else:
                payload[column] = text

        if reason is not None:
            rejected.append({"row": line, "reason": reason})
            continue

        # Renames first: they say what the rest of the row's warnings are about.
        warnings: list[str] = list(alias_warnings)

        for column in missing:
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

        # Validation last: it reads a finished payload, so every column is
        # present and already coerced and nothing here has to re-parse.
        reason = validate_payload(payload)
        if reason is not None:
            rejected.append({"row": line, "reason": reason})
            continue

        # Deduped on the id the row will actually be scored under, which is the
        # synthesised row-N when the file supplied none -- unique by
        # construction, so a file without ids never self-collides.
        identifier = payload[LABEL_COLUMN]
        if identifier in seen_ids:
            rejected.append(
                {"row": line, "reason": f"duplicate TransactionID {identifier!r}"}
            )
            continue
        seen_ids.add(identifier)

        accepted.append(NormalisedRow(row=line, payload=payload, warnings=warnings))

    return accepted, rejected
