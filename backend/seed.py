"""Populates the transaction log so the dashboard feed is not empty on first load.

Deliberately separate from ml/pipeline/train.py: putting this in the training
pipeline would make offline training depend on Firestore, and would break the ml
package's tests on any machine without the google-cloud libraries.
"""
from __future__ import annotations

import argparse

import pandas as pd

from backend.config import Settings
from backend.storage import TransactionLog, build_artifact_source, build_transaction_log
from ml.config import Config
from ml.data.loader import load_raw
from ml.features.engineer import RAW_INPUT_FIELDS, ProfileStore
from ml.pipeline.score import Scorer
from ml.storage.artifacts import load_bundle

DEFAULT_LIMIT = 200


def seed_log(
    scorer: Scorer,
    log: TransactionLog,
    raw_df: pd.DataFrame,
    limit: int = DEFAULT_LIMIT,
) -> int:
    """Score the most recent `limit` historical rows into `log`. Returns the count."""
    ordered = raw_df.sort_values("TransactionDate").tail(limit)

    # scorer.profiles came from training, which observes every row of raw_df --
    # including the ones about to be replayed here. account_last_tx already holds
    # each account's true final transaction date, so replaying against it would
    # make every repeat account's TimeSinceLastTx_Hours collapse to exactly 0h (the
    # single row that happens to be that account's actual last) or the training
    # median (every other occurrence, via the "predates known activity" clamp),
    # flooding the feed with false anomalies instead of the intended mix. A fresh,
    # empty store scoped to just this replay makes each row's history-derived
    # features reflect only what the replay itself has seen so far, then the
    # scorer's original store is restored so seeding leaves it exactly as the
    # caller passed it in.
    original_profiles = scorer.profiles
    scorer.profiles = ProfileStore()
    try:
        written = 0
        for row in ordered.to_dict(orient="records"):
            payload = {field: row[field] for field in RAW_INPUT_FIELDS}
            payload["TransactionID"] = str(row["TransactionID"])
            payload["TransactionDate"] = str(row["TransactionDate"])
            log.append(scorer.score_transaction(payload))
            written += 1
    finally:
        scorer.profiles = original_profiles
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the scored-transaction log.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    args = parser.parse_args()

    settings = Settings.from_env()
    directory = build_artifact_source(settings).ensure_local()
    scorer = Scorer(load_bundle(directory), Config.load().get("ensemble.threshold"))
    log = build_transaction_log(settings)
    raw = load_raw(Config.load().get("data.csv_path"))

    written = seed_log(scorer, log, raw, limit=args.limit)
    print(f"seeded {written} transactions into {settings.transaction_log}")


if __name__ == "__main__":
    main()
