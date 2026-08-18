"""Row validation.

Two things are being tested, and the second matters more than the first: that
impossible input is rejected, and that merely *unusual* input is not. A
validator tuned to the training range would quietly delete the fraud the
ensemble exists to find, so the headroom is asserted as deliberately as the
bounds are.
"""
import pytest

from backend.validation import (
    MAX_AGE,
    MAX_ID_LENGTH,
    MAX_LOGINS,
    MIN_AGE,
    validate_payload,
)

VALID = {
    "TransactionID": "TX000001",
    "AccountID": "AC00128",
    "DeviceID": "D000380",
    "Location": "Houston",
    "TransactionDate": "2023-04-11 16:29:14",
    "TransactionAmount": 120.50,
    "AccountBalance": 5000.0,
    "CustomerAge": 40,
    "TransactionDuration": 90,
    "LoginAttempts": 1,
    "TransactionType": "Debit",
    "Channel": "ATM",
    "CustomerOccupation": "Student",
}


def payload(**overrides):
    return {**VALID, **overrides}


def test_a_valid_row_passes():
    assert validate_payload(payload()) is None


# ---------- what must be rejected ----------


@pytest.mark.parametrize(
    "field,value",
    [
        ("TransactionAmount", 0.0),
        ("TransactionAmount", -50.0),
        ("AccountBalance", 0.0),
        ("AccountBalance", -1.0),
        ("CustomerAge", 5),
        ("CustomerAge", 900),
        ("TransactionDuration", -1),
        ("TransactionDuration", 90_000),
        ("LoginAttempts", -1),
        ("LoginAttempts", 9999),
    ],
)
def test_impossible_numeric_values_are_rejected(field, value):
    reason = validate_payload(payload(**{field: value}))
    assert reason is not None
    assert field in reason


@pytest.mark.parametrize("field", ["TransactionAmount", "CustomerAge", "LoginAttempts"])
def test_a_null_in_a_crucial_column_is_rejected(field):
    reason = validate_payload(payload(**{field: None}))
    assert reason == f"{field} is null"


@pytest.mark.parametrize(
    "column,value",
    [
        ("Channel", "Pigeon"),
        ("TransactionType", "Barter"),
        ("CustomerOccupation", "Wizard"),
        ("Channel", ""),
        ("Channel", None),
    ],
)
def test_an_unknown_categorical_level_is_rejected(column, value):
    """Not merely unusual: transform_one sets every indicator in the group to
    0, which is a feature vector no training row ever had."""
    reason = validate_payload(payload(**{column: value}))
    assert reason is not None
    assert column in reason


@pytest.mark.parametrize(
    "value",
    ["", "   ", "TX 000001", "TX\t01", "ab", "X" * (MAX_ID_LENGTH + 1)],
)
def test_a_malformed_transaction_id_is_rejected(value):
    reason = validate_payload(payload(TransactionID=value))
    assert reason is not None
    assert "TransactionID" in reason


@pytest.mark.parametrize("column", ["AccountID", "DeviceID"])
def test_a_whitespace_or_empty_identity_is_rejected(column):
    """A whitespace id would otherwise create a phantom account."""
    assert validate_payload(payload(**{column: "  "})) is not None
    assert validate_payload(payload(**{column: "AC 001"})) is not None


# ---------- what must NOT be rejected ----------


def test_spending_more_than_the_balance_is_allowed():
    """119 of 2,512 training rows do this. It is the UtilizationRatio signal,
    not a broken row -- rejecting it would delete the account-drain case."""
    assert validate_payload(payload(TransactionAmount=9000.0, AccountBalance=100.0)) is None


def test_login_attempts_above_the_training_maximum_are_allowed():
    """Training tops out at 5 and credential stuffing is a demo preset. The cap
    exists to catch nonsense, not to catch the attack."""
    for attempts in (6, 12, 50, MAX_LOGINS):
        assert validate_payload(payload(LoginAttempts=attempts)) is None, attempts


def test_ages_beyond_the_training_range_but_within_human_limits_are_allowed():
    """Training is 18-80. A 95-year-old is a customer, not a broken row."""
    for age in (MIN_AGE, 81, 95, MAX_AGE):
        assert validate_payload(payload(CustomerAge=age)) is None, age


def test_a_tiny_amount_is_allowed():
    # Training min is 0.26; anything above zero is a real transaction.
    assert validate_payload(payload(TransactionAmount=0.01)) is None


def test_an_unfamiliar_location_is_allowed():
    """Location is not one-hot encoded -- Location_Freq is, and it has a
    documented default for unseen cities."""
    assert validate_payload(payload(Location="Reykjavik")) is None


def test_an_unfamiliar_transaction_id_format_is_allowed():
    """The id is a label and is never scored, so format is not policed."""
    assert validate_payload(payload(TransactionID="order-99881-b")) is None
