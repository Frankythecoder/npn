"""Gaussian Mixture detector: flags the lowest-log-likelihood rows.

Not live-scorable: fitted and reported, but takes no part in the vote.
"""
from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture

from ml.detectors.base import BaseDetector


class GMMDetector(BaseDetector):
    name = "gmm"
    view = "full"
    scaler = "standard"
    live_scorable = False

    def __init__(
        self,
        contamination: float = 0.05,
        n_components: int = 5,
        covariance_type: str = "full",
        random_state: int = 42,
    ) -> None:
        super().__init__(
            contamination=contamination,
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
        )

    def _fit(self, X: np.ndarray) -> None:
        self._model = GaussianMixture(
            n_components=self.params["n_components"],
            covariance_type=self.params["covariance_type"],
            random_state=self.params["random_state"],
        ).fit(X)

    def _score(self, X: np.ndarray) -> np.ndarray:
        # score_samples is a log-likelihood, higher for typical rows, so negate.
        return -self._model.score_samples(X)
