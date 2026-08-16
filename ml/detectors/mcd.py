"""Minimum Covariance Determinant detector.

Fitted on the continuous view only. On the full 19-column matrix the covariance is
rank-deficient and, more fundamentally, LoginAttempts, DailyAccountVolume and
DailyDeviceVelocity are 95-98% single-valued and become exactly constant inside
MCD's central support subset, making that subset's covariance singular (spec 2.2).

Not live-scorable: it is fitted and reported, but takes no part in the vote.
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import EllipticEnvelope

from ml.detectors.base import BaseDetector


class MCDDetector(BaseDetector):
    name = "mcd"
    view = "continuous"
    scaler = "continuous"
    live_scorable = False

    def __init__(self, contamination: float = 0.05, random_state: int = 42) -> None:
        super().__init__(contamination=contamination, random_state=random_state)

    def _fit(self, X: np.ndarray) -> None:
        self._model = EllipticEnvelope(
            contamination=self.contamination,
            random_state=self.params["random_state"],
        ).fit(X)

    def _score(self, X: np.ndarray) -> np.ndarray:
        # Mahalanobis distance is already higher for anomalies.
        return self._model.mahalanobis(X)
