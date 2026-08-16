"""DBSCAN with nearest-core-sample scoring for unseen rows.

scikit-learn's DBSCAN exposes only fit_predict and cannot classify new points. The
core samples are retained at fit time and a new row is scored by its distance to
the nearest one, which is the natural extension of DBSCAN's own rule: a point
within reach of a core sample joins that cluster, a point far from every core
sample is noise (spec 4.6).

eps shapes the cluster structure; the anomaly rate is set by threshold transfer,
not by eps. native_noise_rate_ is reported so a badly chosen eps stays visible.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from ml.detectors.base import BaseDetector


class DBSCANDetector(BaseDetector):
    name = "dbscan"
    view = "full"
    scaler = "standard"
    live_scorable = True

    def __init__(
        self,
        contamination: float = 0.05,
        eps: float = 3.0,
        min_samples: int = 5,
    ) -> None:
        super().__init__(
            contamination=contamination, eps=eps, min_samples=min_samples
        )
        self.core_samples_: np.ndarray | None = None
        self.native_noise_rate_: float | None = None
        self.n_clusters_: int | None = None

    def _fit(self, X: np.ndarray) -> None:
        self._model = DBSCAN(
            eps=self.params["eps"],
            min_samples=self.params["min_samples"],
            n_jobs=-1,
        ).fit(X)

        labels = self._model.labels_
        if len(self._model.core_sample_indices_) == 0:
            raise ValueError(
                f"dbscan: eps={self.params['eps']} produced no core samples; "
                "increase eps or lower min_samples"
            )

        self.core_samples_ = X[self._model.core_sample_indices_]
        self.native_noise_rate_ = float((labels == -1).mean())
        self.n_clusters_ = int(len(set(labels)) - (1 if -1 in labels else 0))
        self._nn = NearestNeighbors(n_neighbors=1).fit(self.core_samples_)

    def _score(self, X: np.ndarray) -> np.ndarray:
        distances, _ = self._nn.kneighbors(X)
        return distances.ravel()
