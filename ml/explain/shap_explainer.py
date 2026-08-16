"""SHAP wrapper producing top features and a plain-English sentence.

"Unusually high" is a claim about the training distribution, so every magnitude
phrase is generated from a measured percentile rather than asserted.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap

FEATURE_PHRASES: dict[str, str] = {
    "TransactionAmount": "transaction amount",
    "CustomerAge": "customer age",
    "TransactionDuration": "transaction duration",
    "LoginAttempts": "login attempts",
    "AccountBalance": "account balance",
    "TimeSinceLastTx_Hours": "gap since the account's last transaction",
    "DailyAccountVolume": "number of transactions on this account today",
    "UtilizationRatio": "share of the account balance drained",
    "DailyDeviceVelocity": "number of transactions from this device today",
    "Location_Freq": "how common this location is",
}

# One-hot features are facts, not magnitudes: "unusually high Channel_Online"
# would be meaningless.
ONE_HOT_PHRASES: dict[str, str] = {
    "TransactionType_Credit": "the transaction was a credit",
    "TransactionType_Debit": "the transaction was a debit",
    "Channel_ATM": "the transaction was made at an ATM",
    "Channel_Branch": "the transaction was made at a branch",
    "Channel_Online": "the transaction was made online",
    "CustomerOccupation_Doctor": "the customer is a doctor",
    "CustomerOccupation_Engineer": "the customer is an engineer",
    "CustomerOccupation_Retired": "the customer is retired",
    "CustomerOccupation_Student": "the customer is a student",
}

HIGH_PERCENTILE = 90.0
LOW_PERCENTILE = 10.0


def build_explainer_state(X_train: pd.DataFrame) -> dict:
    """Persist the sorted training values per column, for percentile lookups."""
    return {
        "feature_percentiles": {
            col: np.sort(X_train[col].to_numpy(dtype=float)).tolist()
            for col in X_train.columns
        }
    }


def _join(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


class ShapExplainer:
    """Explains a single row against the surrogate model."""

    def __init__(
        self,
        model: Any,
        feature_columns: list[str],
        explainer_state: dict,
    ) -> None:
        self.model = model
        self.feature_columns = list(feature_columns)
        self._percentiles = {
            col: np.asarray(values, dtype=float)
            for col, values in explainer_state["feature_percentiles"].items()
        }
        self._explainer = shap.TreeExplainer(model)

    def _percentile_of(self, column: str, value: float) -> float:
        reference = self._percentiles.get(column)
        if reference is None or len(reference) == 0:
            return 50.0
        position = float(np.searchsorted(reference, value, side="right"))
        return 100.0 * position / len(reference)

    def _phrase_for(self, column: str, value: float, percentile: float) -> str:
        if column in ONE_HOT_PHRASES:
            return ONE_HOT_PHRASES[column]
        noun = FEATURE_PHRASES.get(column, column)
        if percentile >= HIGH_PERCENTILE:
            return f"an unusually high {noun} ({percentile:.0f}th percentile)"
        if percentile <= LOW_PERCENTILE:
            return f"an unusually low {noun} ({percentile:.0f}th percentile)"
        return f"the {noun} ({percentile:.0f}th percentile)"

    def explain(
        self, row: pd.DataFrame, is_anomaly: bool, top_n: int = 3
    ) -> dict:
        """Return top contributing features and a rendered sentence."""
        frame = row[self.feature_columns]
        shap_values = np.asarray(self._explainer.shap_values(frame))
        if shap_values.ndim == 3:
            # Some SHAP versions return one matrix per class.
            shap_values = shap_values[:, :, -1]
        contributions = shap_values[0]

        order = np.argsort(np.abs(contributions))[::-1][:top_n]

        top_features = []
        phrases = []
        for idx in order:
            column = self.feature_columns[idx]
            value = float(frame.iloc[0][column])
            percentile = self._percentile_of(column, value)
            top_features.append(
                {
                    "feature": column,
                    "value": value,
                    "shap_value": float(contributions[idx]),
                    "direction": (
                        "increases" if contributions[idx] > 0 else "decreases"
                    ),
                    "percentile": percentile,
                }
            )
            phrases.append(self._phrase_for(column, value, percentile))

        if is_anomaly:
            sentence = f"Flagged primarily due to {_join(phrases[:2])}."
        else:
            sentence = (
                "No strong anomaly indicators. The closest contributors were "
                f"{_join(phrases[:2])}."
            )

        return {
            "top_features": top_features,
            "plain_english": sentence,
            "surrogate_probability": float(
                self.model.predict_proba(frame)[0, 1]
            ),
        }
