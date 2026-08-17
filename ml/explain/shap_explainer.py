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
    "LoginAttempts": "number of login attempts",
    "AccountBalance": "account balance",
    "TimeSinceLastTx_Hours": "gap since the account's last transaction",
    "DailyAccountVolume": "number of transactions on this account today",
    "UtilizationRatio": "share of the account balance drained",
    "DailyDeviceVelocity": "number of transactions from this device today",
    "Location_Freq": "location familiarity",
}

# One-hot features are facts, not magnitudes: "unusually high Channel_Online"
# would be meaningless. Phrased as noun phrases (not clauses) so they compose
# with the sentence templates' "due to {...}" / "were {...}" slots. Each entry
# is (set_phrase, unset_phrase): the feature's value (0.0 or 1.0) picks which
# one renders, so the sentence never asserts a level the row does not have.
ONE_HOT_PHRASES: dict[str, tuple[str, str]] = {
    "TransactionType_Credit": ("a credit transaction", "not a credit transaction"),
    "TransactionType_Debit": ("a debit transaction", "not a debit transaction"),
    "Channel_ATM": ("an ATM transaction", "not an ATM transaction"),
    "Channel_Branch": ("an in-branch transaction", "not an in-branch transaction"),
    "Channel_Online": ("an online transaction", "not an online transaction"),
    "CustomerOccupation_Doctor": ("a doctor's account", "not a doctor's account"),
    "CustomerOccupation_Engineer": (
        "an engineer's account",
        "not an engineer's account",
    ),
    "CustomerOccupation_Retired": (
        "a retired customer's account",
        "not a retired customer's account",
    ),
    "CustomerOccupation_Student": (
        "a student's account",
        "not a student's account",
    ),
}

HIGH_PERCENTILE = 90.0
LOW_PERCENTILE = 10.0


def _ordinal(n: int) -> str:
    """Render an integer with its English ordinal suffix (1st, 2nd, 3rd, 4th...).

    Handles the 11th/12th/13th exception to the usual 1/2/3 -> st/nd/rd rule.
    """
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


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
            set_phrase, unset_phrase = ONE_HOT_PHRASES[column]
            # The one-hot's own value picks the phrase: rendering the "set"
            # phrase regardless of value would state a false fact whenever
            # SHAP surfaces a column that is actually 0 for this row.
            return set_phrase if value >= 0.5 else unset_phrase
        noun = FEATURE_PHRASES.get(column, column)
        rank = _ordinal(round(percentile))
        if percentile >= HIGH_PERCENTILE:
            return f"an unusually high {noun} ({rank} percentile)"
        if percentile <= LOW_PERCENTILE:
            return f"an unusually low {noun} ({rank} percentile)"
        return f"the {noun} ({rank} percentile)"

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

        # The sentence should headline features that are actually remarkable,
        # not whichever ranked highest by raw SHAP magnitude -- a mid-band
        # value in the #2 slot reads as a false claim of severity. One-hot
        # phrases are statements of fact rather than magnitude claims, so
        # they always qualify. Order among qualifying phrases is preserved
        # from SHAP rank; if none qualify, fall back to the raw top two so
        # the sentence still says something. top_features itself is
        # untouched -- the dashboard's bar chart needs the true attribution.
        qualifying = [
            phrase
            for feature, phrase in zip(top_features, phrases)
            if feature["feature"] in ONE_HOT_PHRASES
            or feature["percentile"] >= HIGH_PERCENTILE
            or feature["percentile"] <= LOW_PERCENTILE
        ]
        headline_phrases = qualifying[:2] if qualifying else phrases[:2]

        if is_anomaly:
            sentence = f"Flagged primarily due to {_join(headline_phrases)}."
        else:
            sentence = (
                "No strong anomaly indicators. The closest contributors were "
                f"{_join(headline_phrases)}."
            )

        return {
            "top_features": top_features,
            "plain_english": sentence,
            "surrogate_probability": float(
                self.model.predict_proba(frame)[0, 1]
            ),
        }
