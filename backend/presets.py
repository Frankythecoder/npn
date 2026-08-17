"""Demo scenarios, served from the backend so they can be tuned without a rebuild.

The 'normal' preset is not optional. A demonstration where every input is flagged
proves nothing; the clean result is what makes the flagged ones credible.

**Why the presets do not share an identity, and why most of them advance the date.**

Scoring records the transaction in the profile store, so the next score sees it as
history. That is correct for production and it is what makes the rapid-fire scenario
work — but it means repeated demo injections accumulate. Measured on a running
server: with every preset sharing one account and device, four injections drove
`DailyDeviceVelocity` to 4 against a training maximum of 2, and the 'normal' preset
went from 0/4 clear to 4/4 flagged, explained as "an unusually high number of
transactions from this device today". A presenter who rehearses and then demonstrates
would watch the clean case quietly stop being clean.

Two changes prevent that. Each preset owns a distinct account and device, so firing
one cannot contaminate another. And every preset except rapid-fire advances its
transaction date by a day per injection, so repeats land on a fresh day and the
same-day counters stay at 1 — while the account itself stays one the model has seen,
which keeps the gap feature real and avoids an "unseen account" warning on stage.

Rapid-fire deliberately opts out: accumulation on a fixed date is the whole point of
that scenario.
"""
from __future__ import annotations

from datetime import datetime, timedelta

BASE = {
    "Location": "San Diego",
    "TransactionDate": "2023-12-01 10:15:00",
    "TransactionAmount": 120.00,
    "AccountBalance": 8000.00,
    "CustomerAge": 45,
    "TransactionDuration": 90,
    "LoginAttempts": 1,
    "TransactionType": "Debit",
    "Channel": "ATM",
    "CustomerOccupation": "Engineer",
}

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

PRESETS: dict[str, dict] = {
    "normal": {
        "label": "Routine transaction",
        "description": "An ordinary debit well within the account's normal behaviour.",
        "accumulates": False,
        "fields": {**BASE, "AccountID": "AC00128", "DeviceID": "D000380"},
    },
    "account_drain": {
        "label": "Account drain",
        "description": "96% of the available balance moved in a single transaction.",
        "accumulates": False,
        "fields": {
            **BASE,
            "AccountID": "AC00455",
            "DeviceID": "D000051",
            "TransactionAmount": 4800.00,
            "AccountBalance": 5000.00,
        },
    },
    "credential_stuffing": {
        "label": "Credential stuffing",
        "description": "Five login attempts before an online transfer.",
        "accumulates": False,
        "fields": {
            **BASE,
            "AccountID": "AC00019",
            "DeviceID": "D000235",
            "LoginAttempts": 5,
            "Channel": "Online",
            "TransactionAmount": 900.00,
        },
    },
    "rapid_fire": {
        "label": "Rapid-fire activity",
        "description": (
            "Repeated transactions on one account the same day. "
            "Send it twice to watch the daily count climb."
        ),
        "accumulates": True,
        "fields": {
            **BASE,
            "AccountID": "AC00447",
            "DeviceID": "D000187",
            "TransactionAmount": 300.00,
        },
    },
}

# How many times each preset has been injected this process. Only used to move
# non-accumulating presets onto a fresh day; rapid-fire ignores it.
_injections: dict[str, int] = {}


def reset_injection_counts() -> None:
    """Forget the per-preset injection tally. Used by tests."""
    _injections.clear()


def materialise(name: str) -> dict:
    """Return a preset's fields, dated so repeats behave as the scenario intends.

    Non-accumulating presets step forward a day per injection so their same-day
    counters stay at 1. Rapid-fire keeps its date fixed, because its counters
    climbing is the thing it demonstrates.
    """
    preset = PRESETS[name]
    fields = dict(preset["fields"])

    count = _injections.get(name, 0)
    _injections[name] = count + 1

    if not preset["accumulates"] and count:
        base = datetime.strptime(fields["TransactionDate"], DATE_FORMAT)
        fields["TransactionDate"] = (base + timedelta(days=count)).strftime(DATE_FORMAT)

    return fields
