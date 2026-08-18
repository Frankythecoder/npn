from types import SimpleNamespace

import pytest

from backend.csvingest import (
    CRUCIAL_COLUMNS,
    MissingCrucialColumns,
    defaults_from_bundle,
    normalise_rows,
    resolve_columns,
)
from ml.features.engineer import CATEGORICAL_LEVELS, FEATURE_COLUMNS, FeatureArtifacts

# Three sorted training values per column is enough to pin a median, and enough
# 1.0s to pin a mode -- the derivation only ever reads the middle element or the
# mean, so a realistic column length would test nothing extra.
PERCENTILES = {
    "TransactionAmount": [10.0, 20.0, 30.0],
    "AccountBalance": [100.0, 200.0, 300.0],
    "CustomerAge": [30.0, 40.0, 50.0],
    "TransactionDuration": [10.0, 60.0, 110.0],
    "LoginAttempts": [1.0, 1.0, 4.0],
    "TransactionType_Credit": [0.0, 0.0, 1.0],
    "TransactionType_Debit": [0.0, 1.0, 1.0],
    "Channel_ATM": [0.0, 0.0, 1.0],
    "Channel_Branch": [0.0, 0.0, 0.0],
    "Channel_Online": [0.0, 1.0, 1.0],
    "CustomerOccupation_Doctor": [0.0, 0.0, 0.0],
    "CustomerOccupation_Engineer": [0.0, 0.0, 1.0],
    "CustomerOccupation_Retired": [0.0, 0.0, 0.0],
    "CustomerOccupation_Student": [0.0, 1.0, 1.0],
}


@pytest.fixture
def bundle():
    artifacts = FeatureArtifacts(
        location_freq={"San Diego": 5, "Houston": 12},
        location_freq_default=1,
        feature_columns=list(FEATURE_COLUMNS),
        continuous_columns=[],
        categorical_levels={k: list(v) for k, v in CATEGORICAL_LEVELS.items()},
        time_since_last_tx_median=10.0,
        time_since_last_tx_max=100.0,
    )
    return SimpleNamespace(
        explainer_state={"feature_percentiles": PERCENTILES},
        feature_artifacts=artifacts,
    )


@pytest.fixture
def defaults(bundle):
    return defaults_from_bundle(bundle)


# ---------- defaults derived from the bundle ----------


def test_numeric_defaults_are_training_medians(defaults):
    assert defaults["TransactionAmount"] == 20.0
    assert defaults["AccountBalance"] == 200.0
    assert defaults["CustomerAge"] == 40.0
    assert defaults["TransactionDuration"] == 60.0
    assert defaults["LoginAttempts"] == 1.0


def test_categorical_defaults_are_the_most_prevalent_level(defaults):
    assert defaults["TransactionType"] == "Debit"
    assert defaults["Channel"] == "Online"
    assert defaults["CustomerOccupation"] == "Student"


def test_location_default_is_the_most_frequent_training_city(defaults):
    assert defaults["Location"] == "Houston"


def test_defaults_cover_every_fillable_column(defaults):
    for column in (
        "TransactionAmount",
        "AccountBalance",
        "CustomerAge",
        "TransactionDuration",
        "LoginAttempts",
        "TransactionType",
        "Channel",
        "CustomerOccupation",
        "Location",
    ):
        assert column in defaults


# ---------- the file-level crucial-column gate ----------


def test_a_file_with_no_crucial_column_is_rejected(defaults):
    with pytest.raises(MissingCrucialColumns) as exc:
        normalise_rows(["AccountID", "DeviceID", "Location"], [["A", "D", "Houston"]], defaults)
    # The message must name what was wanted, not just say "invalid".
    for column in CRUCIAL_COLUMNS:
        assert column in str(exc.value)


def test_a_single_crucial_column_is_enough(defaults):
    accepted, rejected = normalise_rows(["TransactionAmount"], [["120.50"]], defaults)
    assert rejected == []
    assert len(accepted) == 1
    assert accepted[0].payload["TransactionAmount"] == 120.50


def test_unrecognised_columns_alone_do_not_satisfy_the_gate(defaults):
    with pytest.raises(MissingCrucialColumns):
        normalise_rows(["IP Address", "MerchantID"], [["1.2.3.4", "M01"]], defaults)


# ---------- filling absent columns ----------


def test_absent_columns_are_filled_from_the_defaults(defaults):
    accepted, _ = normalise_rows(["TransactionAmount"], [["120.50"]], defaults)
    payload = accepted[0].payload
    assert payload["AccountBalance"] == 200.0
    assert payload["Channel"] == "Online"
    assert payload["Location"] == "Houston"


def test_each_filled_column_raises_a_warning_naming_it(defaults):
    accepted, _ = normalise_rows(["TransactionAmount"], [["120.50"]], defaults)
    joined = " ".join(accepted[0].warnings)
    assert "AccountBalance" in joined
    assert "CustomerOccupation" in joined
    # The column that WAS supplied must not be reported as filled.
    assert "TransactionAmount" not in joined


def test_supplied_columns_are_never_overwritten_by_defaults(defaults):
    accepted, _ = normalise_rows(
        ["TransactionAmount", "Channel"], [["120.50", "ATM"]], defaults
    )
    assert accepted[0].payload["Channel"] == "ATM"


def test_absent_identity_columns_get_distinct_synthetic_ids(defaults):
    accepted, _ = normalise_rows(
        ["TransactionAmount"], [["10"], ["20"]], defaults
    )
    first, second = accepted
    assert first.payload["AccountID"] != second.payload["AccountID"]
    assert first.payload["DeviceID"] != second.payload["DeviceID"]
    assert first.payload["TransactionDate"]


def test_supplied_identity_columns_are_kept_as_given(defaults):
    accepted, _ = normalise_rows(
        ["TransactionAmount", "AccountID", "DeviceID"],
        [["10", "AC00128", "D000380"]],
        defaults,
    )
    assert accepted[0].payload["AccountID"] == "AC00128"
    assert accepted[0].payload["DeviceID"] == "D000380"


# ---------- row-level rejection ----------


@pytest.mark.parametrize("blank", ["", "   ", "NA", "n/a", "null", "NaN", "None"])
def test_a_null_cell_in_a_supplied_column_rejects_that_row(defaults, blank):
    accepted, rejected = normalise_rows(
        ["TransactionAmount", "Channel"],
        [["120.50", "ATM"], ["99.00", blank]],
        defaults,
    )
    assert len(accepted) == 1
    assert len(rejected) == 1
    assert rejected[0]["row"] == 3  # header is row 1, so the second data row is 3
    assert "Channel" in rejected[0]["reason"]


def test_an_unparseable_number_rejects_only_that_row(defaults):
    accepted, rejected = normalise_rows(
        ["TransactionAmount"], [["120.50"], ["not-a-number"], ["7"]], defaults
    )
    assert [a.payload["TransactionAmount"] for a in accepted] == [120.50, 7.0]
    assert len(rejected) == 1
    assert rejected[0]["row"] == 3
    assert "TransactionAmount" in rejected[0]["reason"]


def test_a_null_in_an_ignored_column_does_not_reject_the_row(defaults):
    accepted, rejected = normalise_rows(
        ["TransactionAmount", "IP Address"], [["120.50", ""]], defaults
    )
    assert len(accepted) == 1
    assert rejected == []


def test_a_short_row_is_rejected_rather_than_silently_padded(defaults):
    accepted, rejected = normalise_rows(
        ["TransactionAmount", "Channel"], [["120.50"]], defaults
    )
    assert accepted == []
    assert len(rejected) == 1
    assert "column" in rejected[0]["reason"].lower()


# ---------- row numbering ----------


def test_row_numbers_are_csv_line_numbers(defaults):
    accepted, _ = normalise_rows(["TransactionAmount"], [["1"], ["2"]], defaults)
    assert [a.row for a in accepted] == [2, 3]


def test_start_row_offsets_numbering_for_a_later_chunk(defaults):
    """Chunked uploads must keep reporting the line number in the original file."""
    accepted, rejected = normalise_rows(
        ["TransactionAmount"], [["1"], ["bad"]], defaults, start_row=502
    )
    assert accepted[0].row == 502
    assert rejected[0]["row"] == 503


# ---------- the transaction id passthrough ----------


def test_transaction_id_is_carried_through_when_supplied(defaults):
    accepted, _ = normalise_rows(
        ["TransactionID", "TransactionAmount"], [["TX000001", "10"]], defaults
    )
    assert accepted[0].payload["TransactionID"] == "TX000001"


def test_a_missing_transaction_id_falls_back_to_the_row_number(defaults):
    accepted, _ = normalise_rows(["TransactionAmount"], [["10"]], defaults)
    assert accepted[0].payload["TransactionID"] == "row-2"


# ---------- synonym column headers ----------


def test_a_synonym_is_read_as_the_column_it_means(defaults):
    accepted, _ = normalise_rows(["txn_amount"], [["120.50"]], defaults)
    assert accepted[0].payload["TransactionAmount"] == 120.50


def test_synonym_matching_ignores_case_spacing_and_punctuation(defaults):
    for header in ("Session Length", "session_length", "SESSION-LENGTH", "sessionlength"):
        accepted, _ = normalise_rows([header], [["90"]], defaults)
        assert accepted[0].payload["TransactionDuration"] == 90.0, header


def test_a_synonym_naming_minutes_is_converted_to_the_trained_unit(defaults):
    # TransactionDuration is seconds in training, so a five-minute session is
    # 300 -- reading it as 5 would put an ordinary row in the tail of the
    # distribution.
    accepted, _ = normalise_rows(
        ["session-length-in-minutes"], [["5"]], defaults
    )
    assert accepted[0].payload["TransactionDuration"] == 300.0


def test_a_synonym_alone_satisfies_the_crucial_column_gate(defaults):
    accepted, rejected = normalise_rows(["session_length_minutes"], [["5"]], defaults)
    assert rejected == []
    assert len(accepted) == 1


def test_the_canonical_name_wins_when_both_are_present(defaults):
    accepted, _ = normalise_rows(
        ["TransactionDuration", "session_length"], [["90", "999"]], defaults
    )
    assert accepted[0].payload["TransactionDuration"] == 90.0


def test_only_the_first_of_two_synonyms_for_one_column_is_used(defaults):
    accepted, _ = normalise_rows(
        ["session_length", "duration_secs"], [["90", "999"]], defaults
    )
    assert accepted[0].payload["TransactionDuration"] == 90.0


def test_a_matched_synonym_warns_naming_both_the_header_and_the_column(defaults):
    accepted, _ = normalise_rows(["txn_amount"], [["120.50"]], defaults)
    joined = " ".join(accepted[0].warnings)
    assert "txn_amount" in joined
    assert "TransactionAmount" in joined


def test_a_converted_synonym_says_so_in_its_warning(defaults):
    accepted, _ = normalise_rows(["duration_minutes"], [["5"]], defaults)
    joined = " ".join(accepted[0].warnings)
    assert "60" in joined


def test_a_matched_synonym_is_not_also_reported_as_filled(defaults):
    accepted, _ = normalise_rows(["txn_amount"], [["120.50"]], defaults)
    filled = [w for w in accepted[0].warnings if "filled with the training default" in w]
    assert not any("TransactionAmount" in w for w in filled)


def test_an_unrecognised_header_is_still_ignored(defaults):
    # Aliasing must not turn every stray column into a guess.
    accepted, _ = normalise_rows(
        ["TransactionAmount", "sensor_reading_7"], [["10", "42"]], defaults
    )
    assert "sensor_reading_7" not in accepted[0].payload


def test_a_synonym_that_is_not_a_number_is_rejected_by_its_real_name(defaults):
    accepted, rejected = normalise_rows(["txn_amount"], [["abc"]], defaults)
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
