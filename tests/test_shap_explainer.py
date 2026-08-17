import numpy as np
import pandas as pd
import pytest

from ml.explain.shap_explainer import (
    FEATURE_PHRASES,
    HIGH_PERCENTILE,
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


def test_login_attempts_phrase_is_a_number_of_noun_phrase():
    """'an unusually high login attempts' is not English; needs 'number of'."""
    assert FEATURE_PHRASES["LoginAttempts"] == "number of login attempts"


def test_location_freq_phrase_is_a_noun_phrase_not_a_clause():
    """'the how common this location is' does not compose with the
    templates' 'the {...}' / 'due to {...}' slots -- a noun phrase does."""
    phrase = FEATURE_PHRASES["Location_Freq"]
    assert not phrase.startswith("how "), phrase
    assert "an unusually high " + phrase == "an unusually high location familiarity"


def test_every_feature_phrase_composes_into_both_templates():
    """Every FEATURE_PHRASES entry must read as a noun phrase across the
    high, low and mid percentile forms, in both sentence templates. A bare
    leading 'how' or an embedded copula ('is'/'was') is the signature of a
    clause smuggled in where a noun phrase is required -- the exact shape
    of both copy defects this guards against."""
    for column, noun in FEATURE_PHRASES.items():
        assert not noun.startswith("how "), f"{column}: {noun!r} reads as a clause"
        assert " is " not in noun and " was " not in noun, (
            f"{column}: {noun!r} reads as a clause"
        )
        for phrase in (
            f"an unusually high {noun} (95th percentile)",
            f"an unusually low {noun} (5th percentile)",
            f"the {noun} (50th percentile)",
        ):
            assert not phrase.startswith("an unusually high how"), phrase
            flagged = f"Flagged primarily due to {phrase}."
            clean = (
                "No strong anomaly indicators. The closest contributors were "
                f"{phrase}."
            )
            assert "due to an unusually high login attempts" not in flagged
            assert "were the how" not in clean


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
    """A model where Channel_Online genuinely drives the label.

    Built from real 0.0/1.0 one-hot draws -- the state the actual feature
    matrix can produce -- rather than the previous fixture's
    `rng.uniform(3, 5)`, a value the real matrix can never hold. That
    unreachable-value fixture is exactly why _phrase_for ignoring `value`
    for one-hot columns went undetected: every row it generated looked
    "set", so the unset branch was never exercised.
    """
    rng = np.random.default_rng(7)
    n = 1200
    X = pd.DataFrame({col: rng.normal(size=n) for col in FEATURE_COLUMNS})
    channel = rng.integers(0, 3, size=n)  # 0=ATM, 1=Branch, 2=Online
    X["Channel_ATM"] = (channel == 0).astype(float)
    X["Channel_Branch"] = (channel == 1).astype(float)
    X["Channel_Online"] = (channel == 2).astype(float)
    y = (X["Channel_Online"] == 1.0).astype(int).to_numpy()
    result = train_surrogate(X, y, test_size=0.25, random_state=7)
    state = build_explainer_state(X)
    return ShapExplainer(result.model, FEATURE_COLUMNS, state), X, y


def test_phrase_for_one_hot_matches_the_actual_value(fitted):
    """Direct, model-independent check: every one-hot phrase must follow the
    value passed in, not just the column name."""
    explainer, _, _ = fitted
    for column, (set_phrase, unset_phrase) in ONE_HOT_PHRASES.items():
        assert explainer._phrase_for(column, 1.0, percentile=50.0) == set_phrase
        assert explainer._phrase_for(column, 0.0, percentile=50.0) == unset_phrase


def test_one_hot_set_value_renders_as_a_noun_phrase(one_hot_driven):
    explainer, X, y = one_hot_driven
    idx = int(np.argmax(y))  # a row where Channel_Online is genuinely 1.0
    assert X.iloc[idx]["Channel_Online"] == 1.0
    result = explainer.explain(X.iloc[[idx]], is_anomaly=True)
    assert result["top_features"][0]["feature"] == "Channel_Online"
    text = result["plain_english"]
    assert "an online transaction" in text
    assert "not an online transaction" not in text
    # The clause form ("the transaction was made online") does not compose
    # with the "Flagged primarily due to ..." slot; guard against regressing
    # to a clause fragment mid-sentence.
    assert "due to the transaction was" not in text
    assert "due to the customer is" not in text


def test_one_hot_unset_value_renders_the_negated_phrase(one_hot_driven):
    """A row where Channel_Online is 0.0 must never be described as an
    online transaction -- that would state a false fact about the row.

    A negated one-hot no longer qualifies as a sentence headline (it is
    usually the most ordinary possible value -- see the qualifying-features
    tests below), so the sentence may cite a different qualifying feature
    instead. What must never happen, in top_features or the sentence, is
    the row being described as online when it was not.
    """
    explainer, X, y = one_hot_driven
    idx = int(np.argmin(y))  # a row where Channel_Online is genuinely 0.0
    assert X.iloc[idx]["Channel_Online"] == 0.0
    result = explainer.explain(X.iloc[[idx]], is_anomaly=False)
    top_features = {f["feature"]: f for f in result["top_features"]}
    assert "Channel_Online" in top_features
    assert top_features["Channel_Online"]["value"] == 0.0
    text = result["plain_english"]
    assert "an online transaction" not in text


# --- Sentence selection: qualifying (extreme) features, not raw SHAP rank ---
#
# explain() must build top_features from raw SHAP magnitude (the dashboard's
# bar chart needs true attribution) but build the *sentence* only from
# phrases that are actually remarkable: a one-hot fact, or a percentile that
# clears HIGH_PERCENTILE/LOW_PERCENTILE. These fixtures bypass __init__'s
# shap.TreeExplainer(model) construction and real training entirely, so the
# SHAP contributions and percentiles are exact and hand-picked rather than
# whatever a trained model happens to produce -- the selection logic is pure
# and does not need a real model to exercise it.


class _FakeTreeExplainer:
    """Stands in for shap.TreeExplainer: returns fixed contributions."""

    def __init__(self, contributions):
        self._contributions = np.asarray(contributions, dtype=float)

    def shap_values(self, frame):
        return self._contributions.reshape(1, -1)


class _FakeSurrogateModel:
    def predict_proba(self, frame):
        return np.array([[0.1, 0.9]])


def _make_rigged_explainer(feature_columns, contributions):
    """A ShapExplainer with hand-picked SHAP contributions, built without
    training or calling shap.TreeExplainer.

    Numeric columns get a 0..99 reference distribution (so percentile N
    corresponds to value N - 1). One-hot columns get a realistic 70/30
    zeros/ones reference instead -- a plain 0..99 array would searchsorted
    a *set* one-hot's value of 1.0 to 100th percentile and (worse) an
    *unset* one-hot's value of 0.0 to 1st percentile purely as an artifact
    of the array only having one element at each of 0.0 and 1.0, which
    would let the LOW_PERCENTILE numeric clause qualify an unset one-hot
    for reasons that have nothing to do with the one-hot-specific rule
    under test.
    """
    explainer = object.__new__(ShapExplainer)
    explainer.model = _FakeSurrogateModel()
    explainer.feature_columns = list(feature_columns)
    explainer._percentiles = {
        col: (
            np.array([0.0] * 70 + [1.0] * 30)
            if col in ONE_HOT_PHRASES
            else np.arange(100, dtype=float)
        )
        for col in feature_columns
    }
    explainer._explainer = _FakeTreeExplainer(contributions)
    return explainer


def test_sentence_cites_the_extreme_feature_not_the_mid_band_top_shap_feature():
    """Top-SHAP feature (TransactionDuration) sits at the 38th percentile --
    unremarkable. The #2-ranked feature (UtilizationRatio) sits at the 100th
    -- genuinely extreme. The sentence must headline the extreme one."""
    explainer = _make_rigged_explainer(
        ["TransactionDuration", "UtilizationRatio", "CustomerAge"],
        contributions=[0.5, 0.3, 0.05],
    )
    row = pd.DataFrame(
        {
            "TransactionDuration": [37.5],  # 38th percentile: mid-band
            "UtilizationRatio": [99.0],  # 100th percentile: extreme
            "CustomerAge": [37.5],  # 38th percentile: mid-band, rank 3
        }
    )
    text = explainer.explain(row, is_anomaly=True)["plain_english"]
    assert "share of the account balance drained (100th percentile)" in text
    assert "transaction duration" not in text
    assert text.startswith("Flagged primarily due to")
    assert text.endswith(".")


def test_sentence_falls_back_to_raw_shap_order_when_nothing_is_extreme():
    """No feature clears the high/low bands: the sentence must still say
    something, using the previous top-two-by-SHAP behaviour."""
    explainer = _make_rigged_explainer(
        ["TransactionDuration", "UtilizationRatio", "CustomerAge"],
        contributions=[0.5, 0.3, 0.05],
    )
    row = pd.DataFrame(
        {
            "TransactionDuration": [49.0],  # 50th percentile
            "UtilizationRatio": [49.0],  # 50th percentile
            "CustomerAge": [49.0],  # 50th percentile
        }
    )
    text = explainer.explain(row, is_anomaly=True)["plain_english"]
    assert "the transaction duration (50th percentile)" in text
    assert "the share of the account balance drained (50th percentile)" in text
    assert text.startswith("Flagged primarily due to")
    assert text.endswith(".")


def test_top_features_ranking_is_unaffected_by_sentence_selection():
    """top_features must stay fully SHAP-ranked regardless of which phrases
    the sentence picks -- the dashboard's bar chart needs true attribution."""
    explainer = _make_rigged_explainer(
        ["TransactionDuration", "UtilizationRatio", "CustomerAge"],
        contributions=[0.5, 0.3, 0.05],
    )
    row = pd.DataFrame(
        {
            "TransactionDuration": [37.5],
            "UtilizationRatio": [99.0],
            "CustomerAge": [37.5],
        }
    )
    result = explainer.explain(row, is_anomaly=True)
    names = [f["feature"] for f in result["top_features"]]
    assert names == ["TransactionDuration", "UtilizationRatio", "CustomerAge"]


# --- Sentence selection: a negated one-hot must never qualify ---
#
# The unconditional "one-hot phrases always qualify" clause let *negated*
# one-hots (e.g. "not a credit transaction" -- true for 77.4% of the
# training data, since TransactionType is binary) straight into the
# sentence. Every one-hot at value 1.0 happens to searchsorted to
# percentile 100.0 regardless of how common the level is, and at value 0.0
# to 100 * (1 - prevalence) -- so percentile alone can't gate this; the
# clause must check the feature's own value instead.


def test_sentence_skips_an_unset_one_hot_in_favour_of_an_extreme_feature():
    """Top-SHAP feature is an *unset* one-hot (a negated fact, not a
    magnitude claim of any kind); the #2-ranked feature is genuinely
    extreme. The sentence must cite the extreme feature, not the negation."""
    explainer = _make_rigged_explainer(
        ["Channel_Online", "UtilizationRatio", "CustomerAge"],
        contributions=[0.5, 0.3, 0.05],
    )
    row = pd.DataFrame(
        {
            "Channel_Online": [0.0],  # unset: "not an online transaction"
            "UtilizationRatio": [99.0],  # 100th percentile: extreme
            "CustomerAge": [37.5],  # mid-band, rank 3
        }
    )
    text = explainer.explain(row, is_anomaly=True)["plain_english"]
    assert "share of the account balance drained (100th percentile)" in text
    assert "online" not in text
    assert text.startswith("Flagged primarily due to")
    assert text.endswith(".")


def test_sentence_still_includes_a_set_one_hot():
    """Guard against over-correcting into excluding every categorical: a
    *set* one-hot (a true, positive fact about the row) must still
    qualify."""
    explainer = _make_rigged_explainer(
        ["Channel_Online", "CustomerAge", "TransactionDuration"],
        contributions=[0.5, 0.3, 0.05],
    )
    row = pd.DataFrame(
        {
            "Channel_Online": [1.0],  # set: "an online transaction"
            "CustomerAge": [49.0],  # mid-band
            "TransactionDuration": [49.0],  # mid-band
        }
    )
    text = explainer.explain(row, is_anomaly=True)["plain_english"]
    assert "an online transaction" in text
    assert "not an online transaction" not in text
    assert text.startswith("Flagged primarily due to an online transaction")


def test_top_features_still_reports_the_unset_one_hot_with_its_shap_value():
    """The filter excludes the negated one-hot from the sentence only --
    top_features (the dashboard's attribution list) must still carry it,
    with its real SHAP value, unchanged."""
    explainer = _make_rigged_explainer(
        ["Channel_Online", "UtilizationRatio", "CustomerAge"],
        contributions=[0.5, 0.3, 0.05],
    )
    row = pd.DataFrame(
        {
            "Channel_Online": [0.0],
            "UtilizationRatio": [99.0],
            "CustomerAge": [37.5],
        }
    )
    result = explainer.explain(row, is_anomaly=True)
    names = [f["feature"] for f in result["top_features"]]
    assert names == ["Channel_Online", "UtilizationRatio", "CustomerAge"]
    assert result["top_features"][0]["value"] == 0.0
    assert result["top_features"][0]["shap_value"] == pytest.approx(0.5)


# --- Percentile convention: mid-rank, not right-only searchsorted ---
#
# side="right" counts every value <= the target, so on a heavily tied
# column the minimum (which is also the mode, for e.g. LoginAttempts in the
# real training data: 2390/2512 rows equal 1) scores near the top just
# because every tied copy counts as "<=". Mid-rank (averaging the left and
# right insertion points) fixes this for tied values while leaving
# genuinely continuous features -- where a value has no exact tie in the
# reference distribution, so left and right insertion points already
# coincide -- completely unchanged.


def test_minimum_of_a_heavily_tied_column_reports_a_mid_range_percentile():
    """LoginAttempts=1 is both the minimum and the overwhelming majority
    value. The old right-only convention scored it near the top (95.1% on
    the real data) simply because every tied '1' counts as '<= 1'. Mid-rank
    must place it near the middle instead, well below HIGH_PERCENTILE."""
    explainer = _make_rigged_explainer(["LoginAttempts"], contributions=[1.0])
    explainer._percentiles["LoginAttempts"] = np.array([1.0] * 95 + [5.0] * 5)
    percentile = explainer._percentile_of("LoginAttempts", 1.0)
    assert percentile < HIGH_PERCENTILE
    assert 40.0 <= percentile <= 60.0


def test_a_genuinely_extreme_value_of_the_same_tied_column_still_reports_near_the_top():
    """LoginAttempts=5 is the rare, genuinely extreme value in the same
    distribution. Mid-rank must not flatten it toward the middle along
    with the tied minimum -- it should still read as clearly extreme."""
    explainer = _make_rigged_explainer(["LoginAttempts"], contributions=[1.0])
    explainer._percentiles["LoginAttempts"] = np.array([1.0] * 95 + [5.0] * 5)
    percentile = explainer._percentile_of("LoginAttempts", 5.0)
    assert percentile >= HIGH_PERCENTILE


def test_percentile_of_a_continuous_feature_is_unaffected_by_the_mid_rank_switch():
    """For a value with no exact tie in the reference distribution, the
    left and right insertion points already coincide, so mid-rank gives
    the identical result the old right-only convention gave -- only tied
    values are affected by the convention switch."""
    explainer = _make_rigged_explainer(["UtilizationRatio"], contributions=[1.0])
    reference = np.linspace(0.0, 2.0, 2000)  # unique values: no ties
    explainer._percentiles["UtilizationRatio"] = reference
    value = 0.9603  # deliberately not an exact reference element
    old_convention = 100.0 * np.searchsorted(reference, value, side="right") / len(reference)
    assert explainer._percentile_of("UtilizationRatio", value) == pytest.approx(old_convention)
