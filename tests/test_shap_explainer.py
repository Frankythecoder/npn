import numpy as np
import pandas as pd
import pytest

from ml.explain.shap_explainer import (
    FEATURE_PHRASES,
    HIGH_PERCENTILE,
    LOW_PERCENTILE,
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


# --- Sentence selection: the qualifying predicate as a rule, not three anecdotes ---
#
# The three defects fixed in the previous batch (an unremarkable feature
# headlining, a negated one-hot qualifying, a tied minimum reading as
# extreme) were each pinned by a test built around the specific row that
# exposed it. None of those tests exercises: two-or-more qualifying features
# (so `qualifying[:2]`'s ordering is untested), the LOW_PERCENTILE branch of
# the qualifying predicate (every prior "extreme feature" test used the high
# band), the >=/<= boundary at HIGH_PERCENTILE/LOW_PERCENTILE themselves, or
# _percentile_of's mid-rank formula away from the single 95/5 tie ratio
# already covered. The tests below cover the predicate as a rule so a fourth
# member of this family fails a test instead of requiring someone to notice
# a bad sentence.


def test_two_qualifying_features_both_appear_in_shap_order():
    """Rank 1 (UtilizationRatio) and rank 3 (CustomerAge) both qualify; rank 2
    (TransactionDuration) sits mid-band and does not. qualifying[:2] must
    keep both, in SHAP rank order, skipping over the non-qualifying rank 2 --
    behaviour no existing test exercises, since every prior case has zero or
    one qualifying feature."""
    explainer = _make_rigged_explainer(
        ["UtilizationRatio", "TransactionDuration", "CustomerAge"],
        contributions=[0.5, 0.3, 0.1],
    )
    row = pd.DataFrame(
        {
            "UtilizationRatio": [99.0],  # rank 1, ~99.5th percentile: extreme high
            "TransactionDuration": [49.0],  # rank 2, ~49.5th percentile: mid-band
            "CustomerAge": [0.5],  # rank 3, ~1st percentile: extreme low
        }
    )
    text = explainer.explain(row, is_anomaly=True)["plain_english"]
    assert "transaction duration" not in text
    utilization_pos = text.index("share of the account balance drained")
    age_pos = text.index("customer age")
    assert utilization_pos < age_pos, "qualifying features must stay in SHAP order"


def test_three_qualifying_features_still_headline_only_the_first_two():
    """All three top features qualify; qualifying[:2] must still truncate to
    two, exactly as it does for the raw-SHAP fallback path."""
    explainer = _make_rigged_explainer(
        ["UtilizationRatio", "CustomerAge", "TransactionDuration"],
        contributions=[0.5, 0.3, 0.1],
    )
    row = pd.DataFrame(
        {
            "UtilizationRatio": [99.0],  # rank 1: extreme high
            "CustomerAge": [0.5],  # rank 2: extreme low
            "TransactionDuration": [98.0],  # rank 3: extreme high too
        }
    )
    text = explainer.explain(row, is_anomaly=True)["plain_english"]
    assert "share of the account balance drained" in text
    assert "customer age" in text
    assert "transaction duration" not in text


def test_low_percentile_feature_qualifies_and_renders_with_low_phrasing():
    """Every prior 'extreme feature headlines' test used the high band. This
    pins the LOW_PERCENTILE half of the same qualifying predicate."""
    explainer = _make_rigged_explainer(
        ["CustomerAge", "TransactionDuration"],
        contributions=[0.5, 0.1],
    )
    row = pd.DataFrame(
        {
            "CustomerAge": [0.5],  # ~1st percentile: extreme low
            "TransactionDuration": [49.0],  # mid-band
        }
    )
    text = explainer.explain(row, is_anomaly=True)["plain_english"]
    assert "an unusually low customer age" in text
    assert "transaction duration" not in text


@pytest.mark.parametrize(
    "percentile, expect_high, expect_low",
    [
        (HIGH_PERCENTILE, True, False),  # exactly 90.0: inclusive high boundary
        (HIGH_PERCENTILE - 0.0001, False, False),  # just below: mid-band
        (HIGH_PERCENTILE + 0.0001, True, False),  # just above: still high
        (LOW_PERCENTILE, False, True),  # exactly 10.0: inclusive low boundary
        (LOW_PERCENTILE + 0.0001, False, False),  # just above: mid-band
        (LOW_PERCENTILE - 0.0001, False, True),  # just below: still low
    ],
)
def test_phrase_for_boundary_operators_are_inclusive_at_the_threshold(
    percentile, expect_high, expect_low
):
    """_phrase_for takes percentile as a direct argument, so the >=/<=
    comparisons against HIGH_PERCENTILE/LOW_PERCENTILE can be pinned exactly
    at, just inside, and just outside each threshold without going through
    percentile computation at all."""
    explainer = _make_rigged_explainer(["CustomerAge"], contributions=[1.0])
    phrase = explainer._phrase_for("CustomerAge", value=40.0, percentile=percentile)
    assert ("unusually high" in phrase) is expect_high
    assert ("unusually low" in phrase) is expect_low


@pytest.mark.parametrize(
    "value, expected_percentile, qualifies",
    [
        (899.5, 90.0, True),  # exactly HIGH_PERCENTILE: inclusive
        (898.5, 89.9, False),  # just below HIGH_PERCENTILE: excluded
        (900.5, 90.1, True),  # just above HIGH_PERCENTILE: included
        (99.5, 10.0, True),  # exactly LOW_PERCENTILE: inclusive
        (100.5, 10.1, False),  # just above LOW_PERCENTILE: excluded
        (98.5, 9.9, True),  # just below LOW_PERCENTILE: included
    ],
)
def test_qualifying_predicate_boundary_is_pinned_through_real_percentiles(
    value, expected_percentile, qualifies
):
    """The qualifying list comprehension in explain() re-applies the same
    >=HIGH_PERCENTILE / <=LOW_PERCENTILE comparisons to a percentile that
    _percentile_of actually computed, not one handed in directly as in the
    _phrase_for table above. A 1000-point reference gives 0.1-percentile
    resolution, so the boundary and its immediate neighbours land on exact
    values rather than being approximated."""
    explainer = _make_rigged_explainer(
        ["UtilizationRatio", "TransactionDuration"],
        contributions=[0.5, 0.1],
    )
    explainer._percentiles["UtilizationRatio"] = np.arange(1000, dtype=float)

    percentile = explainer._percentile_of("UtilizationRatio", value)
    assert percentile == pytest.approx(expected_percentile)

    row = pd.DataFrame({"UtilizationRatio": [value], "TransactionDuration": [49.0]})
    text = explainer.explain(row, is_anomaly=True)["plain_english"]
    headlines_as_extreme = "share of the account balance drained" in text and (
        "unusually high" in text or "unusually low" in text
    )
    assert headlines_as_extreme is qualifies
    if not qualifies:
        # TransactionDuration is mid-band too, so qualifying is empty and both
        # features fall back into the sentence via the raw-SHAP-order path.
        assert "the share of the account balance drained" in text


@pytest.mark.parametrize(
    "reference, value, expected_percentile",
    [
        # Below the entire reference range.
        ([10.0, 20.0, 30.0], 5.0, 0.0),
        # Above the entire reference range.
        ([10.0, 20.0, 30.0], 100.0, 100.0),
        # Tied at the very bottom: three copies of the minimum out of five
        # values. Mid-rank averages left=0 and right=3 -> 30.0, not 0.0.
        ([1.0, 1.0, 1.0, 5.0, 10.0], 1.0, 30.0),
        # Tied at the very top: three copies of the maximum out of five
        # values. Mid-rank averages left=2 and right=5 -> 70.0, not 100.0.
        ([1.0, 5.0, 10.0, 10.0, 10.0], 10.0, 70.0),
    ],
)
def test_percentile_of_mid_rank_across_tie_positions_and_out_of_range_values(
    reference, value, expected_percentile
):
    """The existing mid-rank tests use a single 95/5 tie ratio. This covers
    the other shapes _percentile_of must handle: ties anchored at the
    bottom of the reference, ties anchored at the top, and values that fall
    entirely outside the observed range in either direction."""
    explainer = _make_rigged_explainer(["UtilizationRatio"], contributions=[1.0])
    explainer._percentiles["UtilizationRatio"] = np.array(reference)
    percentile = explainer._percentile_of("UtilizationRatio", value)
    assert percentile == pytest.approx(expected_percentile)
