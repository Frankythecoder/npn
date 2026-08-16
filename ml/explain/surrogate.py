"""XGBoost surrogate trained on the ensemble's decision.

The surrogate explains; it never decides. Its purpose is to make a single
transaction explainable via SHAP without attributing across four detectors.

It is therefore scored on FIDELITY to the ensemble - held-out AUC and agreement
against the ensemble label - not on detection accuracy. The labels are themselves
model output, so reporting "accuracy" against them would be a category error.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


@dataclass
class SurrogateResult:
    model: Any
    auc: float
    agreement: float
    scale_pos_weight: float


def train_surrogate(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    test_size: float = 0.25,
    random_state: int = 42,
) -> SurrogateResult:
    """Fit the surrogate and measure how faithfully it reproduces the ensemble."""
    y = np.asarray(y, dtype=int)
    positives = int(y.sum())
    if positives == 0 or positives == len(y):
        raise ValueError(
            "train_surrogate needs both classes present in the ensemble label"
        )

    scale_pos_weight = float((len(y) - positives) / positives)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, probabilities))
    agreement = float((model.predict(X_test) == y_test).mean())

    # Refit on the full dataset so the shipped model has seen every row.
    final = XGBClassifier(**model.get_params())
    final.fit(X, y)

    return SurrogateResult(
        model=final,
        auc=auc,
        agreement=agreement,
        scale_pos_weight=scale_pos_weight,
    )
