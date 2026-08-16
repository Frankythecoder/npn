"""The single scoring path.

Feature engineering, the live detectors, the ensemble vote, the profile update and
the SHAP explanation all happen here, so training and serving never duplicate
logic. Artifacts load once into a module-level singleton, not per call.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ml.config import Config
from ml.detectors.registry import DETECTOR_ORDER
from ml.ensemble.voting import combine_one
from ml.explain.shap_explainer import ShapExplainer
from ml.features.engineer import transform_one
from ml.storage.artifacts import ArtifactBundle, load_bundle


class Scorer:
    """Holds the loaded bundle and scores one transaction at a time."""

    def __init__(self, bundle: ArtifactBundle, threshold: float) -> None:
        self.bundle = bundle
        self.threshold = float(threshold)
        self.artifacts = bundle.feature_artifacts
        self.profiles = bundle.profile_store

        # Only detectors that can score an unseen row take part. load_bundle
        # returns them alphabetically, so impose the registry's canonical
        # order here rather than re-declaring it as a second literal that
        # could silently drift from DETECTOR_ORDER.
        self.live = sorted(
            (d for d in bundle.detectors.values() if d.live_scorable),
            key=lambda d: DETECTOR_ORDER.index(d.name),
        )

        self.explainer = ShapExplainer(
            bundle.surrogate,
            self.artifacts.feature_columns,
            bundle.explainer_state,
        )

    def score_transaction(self, raw_txn: dict) -> dict:
        """Engineer, score, vote, explain, and record the transaction."""
        frame, warnings = transform_one(raw_txn, self.artifacts, self.profiles)

        detector_entries = []
        flags: dict[str, int] = {}
        for detector in self.live:
            scaled = pd.DataFrame(
                self.bundle.scalers[detector.scaler].transform(frame),
                columns=frame.columns,
            )
            score = float(detector.score(scaled)[0])
            flag = int(detector.flag(scaled)[0])
            flags[detector.name] = flag
            detector_entries.append(
                {
                    "name": detector.name,
                    "flag": flag,
                    "score": score,
                    "score_percentile": float(detector.score_percentile(score)),
                    "live_scored": True,
                }
            )

        ensemble = combine_one(flags, self.threshold)
        explanation = self.explainer.explain(frame, ensemble.is_anomaly)

        # Record the transaction so subsequent scores see it as history.
        self.profiles.observe(
            str(raw_txn["AccountID"]),
            str(raw_txn["DeviceID"]),
            pd.Timestamp(raw_txn["TransactionDate"]),
        )

        return {
            "transaction_id": str(raw_txn.get("TransactionID", "")),
            "scored_at": datetime.now(timezone.utc).isoformat(),
            "ensemble": ensemble.as_dict(),
            "detectors": detector_entries,
            "explanation": explanation,
            "features": {
                col: float(frame.iloc[0][col]) for col in frame.columns
            },
            "raw": dict(raw_txn),
            "warnings": warnings,
        }


_SCORER: Scorer | None = None


def get_scorer(artifact_dir: str | Path | None = None) -> Scorer:
    """Return the process-wide scorer, loading the bundle on first use."""
    global _SCORER
    if _SCORER is None:
        cfg = Config.load()
        source = Path(artifact_dir or cfg.get("storage.local_dir"))
        _SCORER = Scorer(load_bundle(source), cfg.get("ensemble.threshold"))
    return _SCORER


def reset_scorer() -> None:
    """Drop the singleton. Used by tests."""
    global _SCORER
    _SCORER = None


def score_transaction(raw_txn: dict) -> dict:
    """Score one transaction using the process-wide scorer."""
    return get_scorer().score_transaction(raw_txn)
