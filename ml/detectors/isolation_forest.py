"""Isolation Forest detector."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from ml.detectors.base import BaseDetector


class IsolationForestDetector(BaseDetector):
    name = "isolation_forest"
    view = "full"
    scaler = "standard"
    live_scorable = True

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        random_state: int = 42,
    ) -> None:
        super().__init__(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
        )

    def _fit(self, X: np.ndarray) -> None:
        self._model = IsolationForest(
            n_estimators=self.params["n_estimators"],
            contamination=self.contamination,
            max_samples="auto",
            random_state=self.params["random_state"],
            n_jobs=-1,
        ).fit(X)

    def _score(self, X: np.ndarray) -> np.ndarray:
        # decision_function is higher for inliers, so negate it.
        return -self._model.decision_function(X)
