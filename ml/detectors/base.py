"""Shared detector contract and the threshold-transfer base class.

Every detector derives its binary flag by comparing a continuous score against a
threshold frozen at fit time. That erases the difference between detectors that
expose a native predict() and those that do not, guarantees each one flags the
contamination rate on its training data, and judges new rows against a fixed
boundary, which is the correct live-scoring semantics.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class AnomalyDetector(Protocol):
    name: str
    view: Literal["full", "continuous"]
    scaler: Literal["standard", "robust", "continuous"]
    live_scorable: bool

    def fit(self, X: pd.DataFrame) -> "AnomalyDetector": ...
    def score(self, X: pd.DataFrame) -> np.ndarray: ...
    def flag(self, X: pd.DataFrame) -> np.ndarray: ...


class BaseDetector(ABC):
    """Implements fit/score/flag on top of subclass _fit and _score hooks."""

    name: str = "base"
    view: str = "full"
    scaler: str = "standard"
    live_scorable: bool = True

    def __init__(self, contamination: float = 0.05, **params) -> None:
        self.contamination = float(contamination)
        self.params = params
        self.threshold_: float | None = None
        self.live_threshold_: float | None = None
        self.train_scores_: np.ndarray | None = None
        self.fit_flags_: np.ndarray | None = None
        self.feature_names_: list[str] | None = None

    # --- subclass hooks -------------------------------------------------

    @abstractmethod
    def _fit(self, X: np.ndarray) -> None:
        """Fit the underlying estimator."""

    @abstractmethod
    def _score(self, X: np.ndarray) -> np.ndarray:
        """Score arbitrary rows. Higher must mean more anomalous."""

    def _training_scores(self, X: np.ndarray) -> np.ndarray:
        """Scores used to set the threshold. LOF overrides this (spec 4.5)."""
        return self._score(X)

    # --- public API -----------------------------------------------------

    def fit(self, X: pd.DataFrame) -> "BaseDetector":
        self.feature_names_ = list(X.columns)
        values = X.to_numpy(dtype=float)
        self._fit(values)

        scores = np.asarray(self._training_scores(values), dtype=float)
        self.threshold_ = float(
            np.percentile(scores, 100.0 * (1.0 - self.contamination))
        )
        self.live_threshold_ = self.threshold_
        self.train_scores_ = np.sort(scores)
        self.fit_flags_ = (scores >= self.threshold_).astype(int)
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        self._check_ready(X)
        return np.asarray(self._score(X.to_numpy(dtype=float)), dtype=float)

    def flag(self, X: pd.DataFrame) -> np.ndarray:
        return (self.score(X) >= self.live_threshold_).astype(int)

    def score_percentile(self, value: float) -> float:
        """Where `value` falls in the training score distribution, as 0-100."""
        if self.train_scores_ is None:
            raise RuntimeError(f"{self.name} is not fitted")
        position = float(np.searchsorted(self.train_scores_, value, side="right"))
        return 100.0 * position / len(self.train_scores_)

    # --- internals ------------------------------------------------------

    def _check_ready(self, X: pd.DataFrame) -> None:
        if self.threshold_ is None:
            raise RuntimeError(f"{self.name} is not fitted")
        if list(X.columns) != self.feature_names_:
            raise ValueError(
                f"{self.name}: column mismatch. "
                f"expected {self.feature_names_}, got {list(X.columns)}"
            )
