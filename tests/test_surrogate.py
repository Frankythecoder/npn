import numpy as np
import pandas as pd
import pytest

from ml.explain.surrogate import SurrogateResult, train_surrogate


@pytest.fixture(scope="module")
def separable():
    rng = np.random.default_rng(42)
    n = 2000
    X = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
        }
    )
    # A clean, learnable rule so fidelity should be high.
    y = ((X["a"] > 1.8) | (X["b"] > 2.0)).astype(int).to_numpy()
    return X, y


def test_returns_a_surrogate_result(separable):
    X, y = separable
    result = train_surrogate(X, y, test_size=0.25, random_state=42)
    assert isinstance(result, SurrogateResult)


def test_fidelity_is_high_on_a_learnable_label(separable):
    X, y = separable
    result = train_surrogate(X, y, test_size=0.25, random_state=42)
    # AUC is the fidelity measure that matters; scale_pos_weight deliberately
    # shifts the hard decision boundary, so agreement gets a looser floor.
    assert result.auc >= 0.95
    assert result.agreement >= 0.85


def test_scale_pos_weight_reflects_class_imbalance(separable):
    X, y = separable
    result = train_surrogate(X, y, test_size=0.25, random_state=42)
    expected = (len(y) - y.sum()) / y.sum()
    assert result.scale_pos_weight == pytest.approx(expected, rel=0.2)


def test_model_predicts_probabilities_for_a_single_row(separable):
    X, y = separable
    result = train_surrogate(X, y, test_size=0.25, random_state=42)
    proba = result.model.predict_proba(X.iloc[[0]])
    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0, 1] <= 1.0


def test_is_deterministic(separable):
    X, y = separable
    a = train_surrogate(X, y, test_size=0.25, random_state=42)
    b = train_surrogate(X, y, test_size=0.25, random_state=42)
    assert a.auc == pytest.approx(b.auc)


def test_rejects_a_single_class_label(separable):
    X, _ = separable
    with pytest.raises(ValueError, match="both classes"):
        train_surrogate(X, np.zeros(len(X), dtype=int), test_size=0.25, random_state=42)
