"""PCA reconstruction-error detector.

This PCA is fitted solely for reconstruction error and is persisted independently
of any PCA used for visualisation (spec 4.7).

Not live-scorable: fitted and reported, but takes no part in the vote.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from ml.detectors.base import BaseDetector


class PCAReconstructionDetector(BaseDetector):
    name = "pca_reconstruction"
    view = "full"
    scaler = "standard"
    live_scorable = False

    def __init__(
        self,
        contamination: float = 0.05,
        n_components: float = 0.95,
        random_state: int = 42,
    ) -> None:
        super().__init__(
            contamination=contamination,
            n_components=n_components,
            random_state=random_state,
        )

    def _fit(self, X: np.ndarray) -> None:
        self._model = PCA(
            n_components=self.params["n_components"],
            random_state=self.params["random_state"],
        ).fit(X)

    def _score(self, X: np.ndarray) -> np.ndarray:
        reconstructed = self._model.inverse_transform(self._model.transform(X))
        return ((X - reconstructed) ** 2).mean(axis=1)
