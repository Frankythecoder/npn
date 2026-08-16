"""Local Outlier Factor, fitted twice.

scikit-learn's novelty=False and novelty=True modes are not interchangeable. The
former exposes negative_outlier_factor_ but has no predict(); the latter has
predict() but its training-data scores are a different quantity. Both are fitted
on the same data and both thresholds are persisted (spec 4.5):

  - the novelty=False fit produces training labels and the rate-table entry
  - the novelty=True fit serves live scoring, calibrated on its own scale
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from ml.detectors.base import BaseDetector


class LOFDetector(BaseDetector):
    name = "lof"
    view = "full"
    scaler = "robust"
    live_scorable = True

    def __init__(self, contamination: float = 0.05, n_neighbors: int = 20) -> None:
        super().__init__(contamination=contamination, n_neighbors=n_neighbors)
        self.live_train_scores_: np.ndarray | None = None

    def _fit(self, X: np.ndarray) -> None:
        # Training-time fit: produces negative_outlier_factor_ for the labels.
        self._train_model = LocalOutlierFactor(
            n_neighbors=self.params["n_neighbors"],
            contamination=self.contamination,
            metric="euclidean",
            n_jobs=-1,
        )
        self._train_model.fit_predict(X)

        # Serving fit: the only copy that can score unseen rows.
        self._live_model = LocalOutlierFactor(
            n_neighbors=self.params["n_neighbors"],
            contamination=self.contamination,
            metric="euclidean",
            novelty=True,
            n_jobs=-1,
        ).fit(X)

    def _training_scores(self, X: np.ndarray) -> np.ndarray:
        # negative_outlier_factor_ is lower for outliers, so negate it. This
        # attribute exists only for the data the model was fitted on.
        return -self._train_model.negative_outlier_factor_

    def _score(self, X: np.ndarray) -> np.ndarray:
        # score_samples is higher for inliers, so negate it.
        return -self._live_model.score_samples(X)

    def fit(self, X) -> "LOFDetector":
        super().fit(X)
        # The novelty copy lives on a different scale, so calibrate it separately.
        live_scores = np.asarray(
            self._score(X.to_numpy(dtype=float)), dtype=float
        )
        self.live_threshold_ = float(
            np.percentile(live_scores, 100.0 * (1.0 - self.contamination))
        )
        self.live_train_scores_ = np.sort(live_scores)
        return self

    def score_percentile(self, value: float) -> float:
        """Percentile against the live score distribution, which is what score() emits."""
        if self.live_train_scores_ is None:
            raise RuntimeError("lof is not fitted")
        position = float(
            np.searchsorted(self.live_train_scores_, value, side="right")
        )
        return 100.0 * position / len(self.live_train_scores_)
