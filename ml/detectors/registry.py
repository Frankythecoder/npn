"""Builds the detector roster from config, in one place.

train.py and score.py both call this, so they cannot disagree about which
detectors exist or which of them vote.
"""
from __future__ import annotations

from ml.config import Config
from ml.detectors.base import BaseDetector
from ml.detectors.dbscan import DBSCANDetector
from ml.detectors.gmm import GMMDetector
from ml.detectors.isolation_forest import IsolationForestDetector
from ml.detectors.kmeans import KMeansDetector
from ml.detectors.lof import LOFDetector
from ml.detectors.mcd import MCDDetector
from ml.detectors.one_class_svm import OneClassSVMDetector
from ml.detectors.pca_reconstruction import PCAReconstructionDetector

DETECTOR_ORDER = [
    "isolation_forest",
    "lof",
    "one_class_svm",
    "dbscan",
    "mcd",
    "gmm",
    "kmeans",
    "pca_reconstruction",
]


def build_detectors(cfg: Config) -> list[BaseDetector]:
    """Instantiate all eight detectors with their configured hyperparameters."""
    contamination = cfg.get("detectors.contamination")
    random_state = cfg.get("detectors.random_state")

    return [
        IsolationForestDetector(
            contamination=contamination,
            n_estimators=cfg.get("detectors.isolation_forest.n_estimators"),
            random_state=random_state,
        ),
        LOFDetector(
            contamination=contamination,
            n_neighbors=cfg.get("detectors.lof.n_neighbors"),
        ),
        OneClassSVMDetector(
            contamination=contamination,
            kernel=cfg.get("detectors.one_class_svm.kernel"),
            gamma=cfg.get("detectors.one_class_svm.gamma"),
            nu=cfg.get("detectors.one_class_svm.nu"),
        ),
        DBSCANDetector(
            contamination=contamination,
            eps=cfg.get("detectors.dbscan.eps"),
            min_samples=cfg.get("detectors.dbscan.min_samples"),
        ),
        MCDDetector(contamination=contamination, random_state=random_state),
        GMMDetector(
            contamination=contamination,
            n_components=cfg.get("detectors.gmm.n_components"),
            covariance_type=cfg.get("detectors.gmm.covariance_type"),
            random_state=random_state,
        ),
        KMeansDetector(
            contamination=contamination,
            n_clusters=cfg.get("detectors.kmeans.n_clusters"),
            n_init=cfg.get("detectors.kmeans.n_init"),
            random_state=random_state,
        ),
        PCAReconstructionDetector(
            contamination=contamination,
            n_components=cfg.get("detectors.pca_reconstruction.n_components"),
            random_state=random_state,
        ),
    ]


def live_detectors(detectors: list[BaseDetector]) -> list[BaseDetector]:
    """The subset that can score an unseen row, and therefore the voting roster."""
    return [d for d in detectors if d.live_scorable]
