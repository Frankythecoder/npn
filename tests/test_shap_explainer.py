import numpy as np
import pandas as pd
import pytest

from ml.explain.shap_explainer import (
    FEATURE_PHRASES,
    ONE_HOT_PHRASES,
    ShapExplainer,
    _ordinal,
    build_explainer_state,
)
from ml.explain.surrogate import train_surrogate
from ml.features.engineer import FEATURE_COLUMNS


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(42)
    n = 1500
    X = pd.DataFrame(
        {col: rng.normal(size=n) for col in FEATURE_COLUMNS}
    )
    X["UtilizationRatio"] = rng.uniform(0, 0.4, size=n)
    X.loc[: n // 20, "UtilizationRatio"] = rng.uniform(0.9, 1.5, size=n // 20 + 1)
    y = (X["UtilizationRatio"] > 0.8).astype(int).to_numpy()
    result = train_surrogate(X, y, test_size=0.25, random_state=42)
    state = build_explainer_state(X)
    return ShapExplainer(result.model, FEATURE_COLUMNS, state), X, y


def test_every_feature_column_has_a_phrase():
    for col in FEATURE_COLUMNS:
        assert col in FEATURE_PHRASES or col in ONE_HOT_PHRASES, col


def test_explain_returns_the_documented_keys(fitted):
    explainer, X, _ = fitted
    result = explainer.explain(X.iloc[[0]], is_anomaly=False)
    assert set(result) == {"top_features", "plain_english", "surrogate_probability"}


def test_top_features_have_the_documented_shape(fitted):
    explainer, X, _ = fitted
    result = explainer.explain(X.iloc[[0]], is_anomaly=False)
    assert 1 <= len(result["top_features"]) <= 3
    for item in result["top_features"]:
        assert set(item) == {
            "feature",
            "value",
            "shap_value",
            "direction",
            "percentile",
        }
        assert item["direction"] in {"increases", "decreases"}
        assert 0.0 <= item["percentile"] <= 100.0


def test_top_features_are_sorted_by_absolute_shap(fitted):
    explainer, X, y = fitted
    idx = int(np.argmax(y))
    result = explainer.explain(X.iloc[[idx]], is_anomaly=True)
    magnitudes = [abs(f["shap_value"]) for f in result["top_features"]]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_the_driving_feature_is_surfaced(fitted):
    explainer, X, y = fitted
    idx = int(np.argmax(y))
    result = explainer.explain(X.iloc[[idx]], is_anomaly=True)
    names = [f["feature"] for f in result["top_features"]]
    assert "UtilizationRatio" in names


def test_flagged_sentence_reads_as_a_flag(fitted):
    explainer, X, y = fitted
    idx = int(np.argmax(y))
    text = explainer.explain(X.iloc[[idx]], is_anomaly=True)["plain_english"]
    assert text.startswith("Flagged primarily due to")
    assert text.endswith(".")


def test_clean_sentence_uses_its_own_template(fitted):
    explainer, X, y = fitted
    idx = int(np.argmin(y))
    text = explainer.explain(X.iloc[[idx]], is_anomaly=False)["plain_english"]
    assert "No strong anomaly indicators" in text
    assert "Flagged" not in text


def test_surrogate_probability_is_a_probability(fitted):
    explainer, X, _ = fitted
    p = explainer.explain(X.iloc[[0]], is_anomaly=False)["surrogate_probability"]
    assert 0.0 <= p <= 1.0


def test_ordinal_suffix_handles_the_teens_exception():
    assert _ordinal(1) == "1st"
    assert _ordinal(2) == "2nd"
    assert _ordinal(3) == "3rd"
    assert _ordinal(4) == "4th"
    assert _ordinal(11) == "11th"
    assert _ordinal(12) == "12th"
    assert _ordinal(13) == "13th"
    assert _ordinal(21) == "21st"
    assert _ordinal(81) == "81st"
    assert _ordinal(92) == "92nd"
    assert _ordinal(100) == "100th"
    assert _ordinal(111) == "111th"


@pytest.fixture(scope="module")
def one_hot_driven():
    rng = np.random.default_rng(7)
    n = 1200
    X = pd.DataFrame({col: rng.normal(size=n) for col in FEATURE_COLUMNS})
    X["Channel_Online"] = rng.uniform(-1, 1, size=n)
    X.loc[: n // 20, "Channel_Online"] = rng.uniform(3, 5, size=n // 20 + 1)
    y = (X["Channel_Online"] > 2).astype(int).to_numpy()
    result = train_surrogate(X, y, test_size=0.25, random_state=7)
    state = build_explainer_state(X)
    return ShapExplainer(result.model, FEATURE_COLUMNS, state), X, y


def test_one_hot_top_feature_renders_as_a_noun_phrase(one_hot_driven):
    explainer, X, y = one_hot_driven
    idx = int(np.argmax(y))
    result = explainer.explain(X.iloc[[idx]], is_anomaly=True)
    assert result["top_features"][0]["feature"] == "Channel_Online"
    text = result["plain_english"]
    assert "an online transaction" in text
    # The clause form ("the transaction was made online") does not compose
    # with the "Flagged primarily due to ..." slot; guard against regressing
    # to a clause fragment mid-sentence.
    assert "due to the transaction was" not in text
    assert "due to the customer is" not in text
