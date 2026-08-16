"""One-Class SVM detector."""
from __future__ import annotations

import numpy as np
from sklearn.svm import OneClassSVM

from ml.detectors.base import BaseDetector


class OneClassSVMDetector(BaseDetector):
    name = "one_class_svm"
    view = "full"
    scaler = "standard"
    live_scorable = True

    def __init__(
        self,
        contamination: float = 0.05,
        kernel: str = "rbf",
        gamma: str = "scale",
        nu: float = 0.05,
    ) -> None:
        super().__init__(
            contamination=contamination, kernel=kernel, gamma=gamma, nu=nu
        )

    def _fit(self, X: np.ndarray) -> None:
        self._model = OneClassSVM(
            kernel=self.params["kernel"],
            gamma=self.params["gamma"],
            nu=self.params["nu"],
        ).fit(X)

    def _score(self, X: np.ndarray) -> np.ndarray:
        # decision_function is higher for inliers, so negate it.
        return -self._model.decision_function(X)
