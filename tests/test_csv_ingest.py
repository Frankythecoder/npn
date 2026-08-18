"""CSV ingest.

An upload must supply every scoring column itself, under its own name or a
recognised synonym. Nothing is substituted from the training set, so most tests
here start from a complete header and take something away, rather than starting
from one column and watching the rest appear.
"""
import pytest

from backend.csvingest import (
    CRUCIAL_COLUMNS,
    REQUIRED_COLUMNS,
    MissingCrucialColumns,
    MissingScoringColumns,
    UnusableUpload,
    normalise_rows,
    resolve_columns,
)

# A header that satisfies the contract, and one row to go with it. Tests that
# are not about completeness build from these, so the thing under test is the
# only thing varying.
COMPLETE = list(REQUIRED_COLUMNS)
ROW = {
    "TransactionAmount": "120.50",
    "AccountBalance": "5000",
    "CustomerAge": "40",
    "TransactionDuration": "90",
    "LoginAttempts": "1",
    "TransactionType": "Debit",
    "Channel": "ATM",
    "CustomerOccupation": "Student",
    "Location": "Houston",
}


def row(**overrides) -> list[str]:
    """One data row for the COMPLETE header, with cells optionally replaced."""
    values = {**ROW, **overrides}
    return [values[column] for column in COMPLETE]


def complete(extra_columns=(), extra_cells=(), rows=None, **kwargs):
    """normalise_rows over the complete header plus any extra columns."""
    columns = COMPLETE + list(extra_columns)
    data = rows if rows is not None else [row() + list(extra_cells)]
    return normalise_rows(columns, data, **kwargs)


def without(column: str):
    """The complete header and row, minus one column."""
    columns = [c for c in COMPLETE if c != column]
    return columns, [ROW[c] for c in columns]


# ---------- the completeness contract ----------


def test_a_complete_file_is_accepted():
    accepted, rejected = complete()
    assert rejected == []
    assert accepted[0].payload["TransactionAmount"] == 120.50
    assert accepted[0].payload["Channel"] == "ATM"


def test_a_file_missing_one_required_column_is_refused():
    columns, cells = without("Channel")
    with pytest.raises(MissingScoringColumns) as exc:
        normalise_rows(columns, [cells])
    assert "Channel" in str(exc.value)


def test_the_refusal_names_every_missing_column_not_just_the_first():
    with pytest.raises(MissingScoringColumns) as exc:
        normalise_rows(["TransactionAmount"], [["120.50"]])
    message = str(exc.value)
    for column in REQUIRED_COLUMNS:
        if column != "TransactionAmount":
            assert column in message, column


def test_nothing_is_substituted_from_the_training_data():
    """The point of the contract: a payload carries only what the file sent."""
    accepted, _ = complete()
    payload = accepted[0].payload
    for column in REQUIRED_COLUMNS:
        expected = float(ROW[column]) if column in CRUCIAL_COLUMNS else ROW[column]
        assert payload[column] == expected, column


def test_no_column_is_filled_from_the_training_distribution():
    accepted, _ = complete()
    assert "filled with the training default" not in " ".join(accepted[0].warnings)


def test_omitting_an_account_id_still_costs_the_history_features(complete_row=None):
    """The one residual, documented rather than hidden.

    Column values are never substituted. But a row with no AccountID is scored
    as a new account, and the three history-derived features -- gap since last
    transaction, daily account volume, daily device velocity -- have nothing to
    derive from, so transform_one falls back for those. Supplying AccountID,
    DeviceID and TransactionDate removes it entirely.
    """
    accepted, _ = complete()
    assert "fall back to training defaults" in " ".join(accepted[0].warnings)

    supplied, _ = complete(
        extra_columns=["AccountID", "DeviceID", "TransactionDate"],
        extra_cells=["AC00128", "D000380", "2023-04-11 16:29:14"],
    )
    assert "fall back to training defaults" not in " ".join(supplied[0].warnings)


def test_a_file_with_no_crucial_column_is_told_it_is_not_a_transaction_file():
    # Distinct from the completeness error: this file is not one of ours at all,
    # and a list of nine column names would be the wrong thing to show.
    with pytest.raises(MissingCrucialColumns) as exc:
        normalise_rows(["AccountID", "DeviceID", "Location"], [["A", "D", "Houston"]])
    for column in CRUCIAL_COLUMNS:
        assert column in str(exc.value)


def test_unrecognised_columns_alone_do_not_satisfy_the_gate():
    with pytest.raises(MissingCrucialColumns):
        normalise_rows(["IP Address", "MerchantID"], [["1.2.3.4", "M01"]])


def test_both_refusals_share_a_base_the_router_can_catch():
    assert issubclass(MissingCrucialColumns, UnusableUpload)
    assert issubclass(MissingScoringColumns, UnusableUpload)


# ---------- identity columns are synthesised, not filled ----------


def test_absent_identity_columns_get_distinct_synthetic_ids():
    accepted, _ = complete(rows=[row(), row()])
    first, second = accepted
    assert first.payload["AccountID"] != second.payload["AccountID"]
    assert first.payload["AccountID"].startswith("CSV-A")
    assert first.payload["DeviceID"].startswith("CSV-D")


def test_a_synthesised_identity_says_so_in_the_warnings():
    accepted, _ = complete()
    joined = " ".join(accepted[0].warnings)
    assert "AccountID" in joined
    assert "new one" in joined


def test_supplied_identity_columns_are_kept_as_given():
    accepted, _ = complete(
        extra_columns=["AccountID", "DeviceID"], extra_cells=["AC00128", "D000380"]
    )
    assert accepted[0].payload["AccountID"] == "AC00128"
    assert accepted[0].payload["DeviceID"] == "D000380"


# ---------- row-level rejection ----------


@pytest.mark.parametrize("blank", ["", "   ", "NA", "n/a", "null", "NaN", "None"])
def test_a_null_cell_in_a_supplied_column_rejects_that_row(blank):
    accepted, rejected = complete(rows=[row(), row(Channel=blank)])
    assert len(accepted) == 1
    assert len(rejected) == 1
    assert rejected[0]["row"] == 3  # header is row 1, so the second data row is 3
    assert "Channel" in rejected[0]["reason"]


def test_an_unparseable_number_rejects_only_that_row():
    accepted, rejected = complete(
        rows=[row(), row(TransactionAmount="not-a-number"), row(TransactionAmount="7")]
    )
    assert [a.payload["TransactionAmount"] for a in accepted] == [120.50, 7.0]
    assert len(rejected) == 1
    assert rejected[0]["row"] == 3
    assert "TransactionAmount" in rejected[0]["reason"]


def test_a_null_in_an_ignored_column_does_not_reject_the_row():
    accepted, rejected = complete(extra_columns=["IP Address"], extra_cells=[""])
    assert len(accepted) == 1
    assert rejected == []


def test_a_short_row_is_rejected_rather_than_silently_padded():
    accepted, rejected = complete(rows=[row()[:-1]])
    assert accepted == []
    assert len(rejected) == 1
    assert "column" in rejected[0]["reason"].lower()


# ---------- row numbering ----------


def test_row_numbers_are_csv_line_numbers():
    accepted, _ = complete(rows=[row(), row()])
    assert [a.row for a in accepted] == [2, 3]


def test_start_row_offsets_numbering_for_a_later_chunk():
    """Chunked uploads must keep reporting the line number in the original file."""
    accepted, rejected = complete(
        rows=[row(), row(TransactionAmount="bad")], start_row=502
    )
    assert accepted[0].row == 502
    assert rejected[0]["row"] == 503


# ---------- the transaction id passthrough ----------


def test_transaction_id_is_carried_through_when_supplied():
    accepted, _ = complete(extra_columns=["TransactionID"], extra_cells=["TX000001"])
    assert accepted[0].payload["TransactionID"] == "TX000001"


def test_a_missing_transaction_id_falls_back_to_the_row_number():
    accepted, _ = complete()
    assert accepted[0].payload["TransactionID"] == "row-2"


# ---------- synonym column headers ----------


def _under_synonym(column: str, header: str, value: str):
    """The complete file, with one column renamed to `header`."""
    rest = [c for c in COMPLETE if c != column]
    return [header] + rest, [value] + [ROW[c] for c in rest]


def test_a_synonym_is_read_as_the_column_it_means():
    columns, cells = _under_synonym("TransactionAmount", "txn_amount", "120.50")
    accepted, _ = normalise_rows(columns, [cells])
    assert accepted[0].payload["TransactionAmount"] == 120.50


@pytest.mark.parametrize(
    "header", ["Session Length", "session_length", "SESSION-LENGTH", "sessionlength"]
)
def test_synonym_matching_ignores_case_spacing_and_punctuation(header):
    columns, cells = _under_synonym("TransactionDuration", header, "90")
    accepted, _ = normalise_rows(columns, [cells])
    assert accepted[0].payload["TransactionDuration"] == 90.0


def test_a_synonym_naming_minutes_is_converted_to_the_trained_unit():
    # TransactionDuration is seconds in training, so a five-minute session is
    # 300 -- reading it as 5 would put an ordinary row in the tail.
    columns, cells = _under_synonym(
        "TransactionDuration", "session-length-in-minutes", "5"
    )
    accepted, _ = normalise_rows(columns, [cells])
    assert accepted[0].payload["TransactionDuration"] == 300.0


def test_a_column_supplied_only_under_a_synonym_counts_as_supplied():
    """Aliasing runs before the completeness check, or a file spelling every
    column differently would be refused for missing all of them."""
    columns = [
        "amount", "balance", "age", "duration", "logins",
        "txntype", "paymentchannel", "occupation", "city",
    ]
    cells = ["120.50", "5000", "40", "90", "1", "Debit", "ATM", "Student", "Houston"]
    accepted, rejected = normalise_rows(columns, [cells])
    assert rejected == []
    assert accepted[0].payload["TransactionAmount"] == 120.50
    assert accepted[0].payload["Location"] == "Houston"


def test_the_canonical_name_wins_when_both_are_present():
    accepted, _ = complete(extra_columns=["session_length"], extra_cells=["999"])
    assert accepted[0].payload["TransactionDuration"] == 90.0


def test_a_matched_synonym_warns_naming_both_the_header_and_the_column():
    columns, cells = _under_synonym("TransactionAmount", "txn_amount", "120.50")
    accepted, _ = normalise_rows(columns, [cells])
    joined = " ".join(accepted[0].warnings)
    assert "txn_amount" in joined
    assert "TransactionAmount" in joined


def test_a_converted_synonym_says_so_in_its_warning():
    columns, cells = _under_synonym("TransactionDuration", "duration_minutes", "5")
    accepted, _ = normalise_rows(columns, [cells])
    assert "60" in " ".join(accepted[0].warnings)


def test_an_unrecognised_header_is_still_ignored():
    accepted, _ = complete(extra_columns=["sensor_reading_7"], extra_cells=["42"])
    assert "sensor_reading_7" not in accepted[0].payload


def test_a_synonym_that_is_not_a_number_is_rejected_by_its_real_name():
    columns, cells = _under_synonym("TransactionAmount", "txn_amount", "abc")
    accepted, rejected = normalise_rows(columns, [cells])
    assert accepted == []
    assert "TransactionAmount" in rejected[0]["reason"]


def test_resolve_columns_reports_what_it_matched():
    resolved, scales, matches = resolve_columns(["txn_amount", "IP Address"])
    assert resolved == ["TransactionAmount", "IP Address"]
    assert matches == [("txn_amount", "TransactionAmount")]
    assert scales == {}


def test_resolve_columns_leaves_a_canonical_header_untouched():
    resolved, scales, matches = resolve_columns(["TransactionAmount"])
    assert resolved == ["TransactionAmount"]
    assert matches == []
    assert scales == {}


# ---------- the opt-in fill path ----------

FILL = {
    "TransactionAmount": 20.0,
    "AccountBalance": 200.0,
    "CustomerAge": 40.0,
    "TransactionDuration": 60.0,
    "LoginAttempts": 1.0,
    "TransactionType": "Debit",
    "Channel": "Online",
    "CustomerOccupation": "Student",
    "Location": "Houston",
}


def test_an_incomplete_file_is_accepted_when_a_fill_table_is_given():
    columns, cells = without("Channel")
    accepted, rejected = normalise_rows(columns, [cells], defaults=FILL)
    assert rejected == []
    assert accepted[0].payload["Channel"] == "Online"


def test_the_same_file_is_still_refused_without_one():
    """Strict is the default: the caller must ask, every time."""
    columns, cells = without("Channel")
    with pytest.raises(MissingScoringColumns):
        normalise_rows(columns, [cells])


def test_each_filled_column_warns_naming_itself_and_the_value():
    columns, cells = without("Channel")
    accepted, _ = normalise_rows(columns, [cells], defaults=FILL)
    joined = " ".join(accepted[0].warnings)
    assert "Channel missing from the file" in joined
    assert "Online" in joined


def test_a_supplied_column_is_never_overwritten_by_the_fill_table():
    accepted, _ = complete(defaults=FILL)
    # ROW says ATM; the fill table says Online. The file wins.
    assert accepted[0].payload["Channel"] == "ATM"
    assert "filled with the training default" not in " ".join(accepted[0].warnings)


def test_a_column_supplied_under_a_synonym_is_not_filled_over():
    columns, cells = _under_synonym("Channel", "paymentchannel", "Branch")
    accepted, _ = normalise_rows(columns, [cells], defaults=FILL)
    assert accepted[0].payload["Channel"] == "Branch"


def test_filling_cannot_rescue_a_file_that_is_not_a_transaction_file():
    """The crucial gate is not negotiable -- filling every column would score
    the training distribution, not the upload."""
    with pytest.raises(MissingCrucialColumns):
        normalise_rows(["IP Address"], [["1.2.3.4"]], defaults=FILL)


# ---------- validation and duplicate removal ----------


def test_an_invalid_row_is_rejected_with_a_reason_naming_the_field():
    accepted, rejected = complete(rows=[row(), row(CustomerAge="900")])
    assert len(accepted) == 1
    assert rejected[0]["row"] == 3
    assert "CustomerAge" in rejected[0]["reason"]


def test_an_unknown_categorical_level_is_rejected_rather_than_scored_on_zeros():
    accepted, rejected = complete(rows=[row(Channel="Pigeon")])
    assert accepted == []
    assert "Channel" in rejected[0]["reason"]


def test_a_valid_but_unusual_row_still_scores():
    """The validator must not eat the anomaly: this row drains the account and
    has more login attempts than anything in training."""
    accepted, rejected = complete(
        rows=[row(TransactionAmount="9000", AccountBalance="100", LoginAttempts="9")]
    )
    assert rejected == []
    assert len(accepted) == 1


def test_a_duplicate_transaction_id_is_dropped_within_one_request():
    accepted, rejected = complete(
        extra_columns=["TransactionID"],
        rows=[row() + ["TX000001"], row() + ["TX000001"], row() + ["TX000002"]],
    )
    assert [a.payload["TransactionID"] for a in accepted] == ["TX000001", "TX000002"]
    assert len(rejected) == 1
    assert "duplicate" in rejected[0]["reason"]
    assert rejected[0]["row"] == 3


def test_a_shared_seen_set_dedupes_across_chunks():
    """Uploads arrive in 200-row chunks, so a set scoped to one call would miss
    most duplicates in a file of any size."""
    seen: set[str] = set()
    first, _ = complete(
        extra_columns=["TransactionID"], rows=[row() + ["TX000001"]], seen_ids=seen
    )
    second, rejected = complete(
        extra_columns=["TransactionID"],
        rows=[row() + ["TX000001"]],
        start_row=502,
        seen_ids=seen,
    )
    assert len(first) == 1
    assert second == []
    assert rejected[0]["row"] == 502
    assert "duplicate" in rejected[0]["reason"]


def test_rows_without_a_transaction_id_never_collide_with_each_other():
    """The synthesised row-N fallback is unique by construction, so a file with
    no id column must not have every row after the first dropped."""
    accepted, rejected = complete(rows=[row(), row(), row()])
    assert len(accepted) == 3
    assert rejected == []
