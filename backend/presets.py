"""Demo scenarios, served from the backend so they can be tuned without a rebuild.

The 'normal' preset is not optional. A demonstration where every input is flagged
proves nothing; the clean result is what makes the flagged ones credible.
"""
from __future__ import annotations

BASE = {
    "AccountID": "AC00128",
    "DeviceID": "D000380",
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

PRESETS: dict[str, dict] = {
    "normal": {
        "label": "Routine transaction",
        "description": "An ordinary debit well within the account's normal behaviour.",
        "fields": dict(BASE),
    },
    "account_drain": {
        "label": "Account drain",
        "description": "96% of the available balance moved in a single transaction.",
        "fields": {**BASE, "TransactionAmount": 4800.00, "AccountBalance": 5000.00},
    },
    "credential_stuffing": {
        "label": "Credential stuffing",
        "description": "Five login attempts before an online transfer.",
        "fields": {**BASE, "LoginAttempts": 5, "Channel": "Online", "TransactionAmount": 900.00},
    },
    "rapid_fire": {
        "label": "Rapid-fire activity",
        "description": "Repeated transactions on one account the same day. Inject twice to see the daily count climb.",
        "fields": {**BASE, "AccountID": "AC00455", "TransactionAmount": 300.00},
    },
}


def preset_names() -> list[str]:
    return list(PRESETS)
