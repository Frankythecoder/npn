"""K-Means detector: flags the largest distance to the nearest centroid.

Not live-scorable: fitted and reported, but takes no part in the vote.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from ml.detectors.base import BaseDetector


class KMeansDetector(BaseDetector):
    name = "kmeans"
    view = "full"
    scaler = "standard"
    live_scorable = False

    def __init__(
        self,
        contamination: float = 0.05,
        n_clusters: int = 8,
        n_init: int = 10,
        random_state: int = 42,
    ) -> None:
        super().__init__(
            contamination=contamination,
            n_clusters=n_clusters,
            n_init=n_init,
            random_state=random_state,
        )

    def _fit(self, X: np.ndarray) -> None:
        self._model = KMeans(
            n_clusters=self.params["n_clusters"],
            n_init=self.params["n_init"],
            random_state=self.params["random_state"],
        ).fit(X)

    def _score(self, X: np.ndarray) -> np.ndarray:
        # transform() returns distances to every centroid; take the nearest.
        return self._model.transform(X).min(axis=1)
