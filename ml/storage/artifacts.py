"""Artifact bundle persistence.

Takes a filesystem path and writes flat into it -- there is no versioned
sub-directory layout here; train.py reuses the same `dest` (e.g. `artifacts/`)
across retrains. Because the directory is reused rather than written once and
treated as immutable, save_bundle clears stale detector pickles and any stale
surrogate file before writing, so retraining with a shrunk detector roster
cannot leave orphan files behind for load_bundle's glob to pick back up.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from ml.features.engineer import FeatureArtifacts, ProfileStore

MANIFEST_NAME = "manifest.json"
DETECTOR_DIR = "detectors"


@dataclass
class ArtifactBundle:
    """Everything the scorer needs, loaded once."""

    manifest: dict
    scalers: dict[str, Any]
    feature_artifacts: FeatureArtifacts
    profile_store: ProfileStore
    detectors: dict[str, Any]
    surrogate: Any
    explainer_state: dict


def save_bundle(bundle: ArtifactBundle, dest: str | Path) -> None:
    """Write the bundle to `dest`, creating it if needed.

    `dest` is reused across retrains rather than versioned, so any detector
    pickles and surrogate file already there are cleared first -- otherwise a
    retrain with a shrunk roster (or one that drops the surrogate) would
    leave orphans that load_bundle's glob picks back up alongside the new
    ones.
    """
    dest = Path(dest)
    detector_dir = dest / DETECTOR_DIR
    if detector_dir.exists():
        for stale in detector_dir.glob("*.pkl"):
            stale.unlink()
    detector_dir.mkdir(parents=True, exist_ok=True)

    surrogate_path = dest / "surrogate_xgb.json"
    if surrogate_path.exists():
        surrogate_path.unlink()

    with open(dest / MANIFEST_NAME, "w", encoding="utf-8") as fh:
        json.dump(bundle.manifest, fh, indent=2, default=str)

    joblib.dump(bundle.scalers, dest / "scalers.pkl")
    joblib.dump(bundle.feature_artifacts, dest / "feature_artifacts.pkl")
    joblib.dump(bundle.profile_store, dest / "profile_store.pkl")
    joblib.dump(bundle.explainer_state, dest / "explainer_state.pkl")

    for name, detector in bundle.detectors.items():
        joblib.dump(detector, detector_dir / f"{name}.pkl")

    if bundle.surrogate is not None:
        bundle.surrogate.save_model(str(surrogate_path))


def load_bundle(src: str | Path) -> ArtifactBundle:
    """Read a bundle written by save_bundle."""
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"artifact bundle not found: {src}")

    with open(src / MANIFEST_NAME, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    detectors = {
        path.stem: joblib.load(path)
        for path in sorted((src / DETECTOR_DIR).glob("*.pkl"))
    }

    surrogate = None
    surrogate_path = src / "surrogate_xgb.json"
    if surrogate_path.exists():
        from xgboost import XGBClassifier

        surrogate = XGBClassifier()
        surrogate.load_model(str(surrogate_path))

    return ArtifactBundle(
        manifest=manifest,
        scalers=joblib.load(src / "scalers.pkl"),
        feature_artifacts=joblib.load(src / "feature_artifacts.pkl"),
        profile_store=joblib.load(src / "profile_store.pkl"),
        detectors=detectors,
        surrogate=surrogate,
        explainer_state=joblib.load(src / "explainer_state.pkl"),
    )
