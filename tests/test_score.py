import pytest

from ml.config import Config
from ml.pipeline.score import Scorer
from ml.pipeline.train import run_training
from ml.storage.artifacts import load_bundle


@pytest.fixture(scope="module")
def artifact_dir(tmp_path_factory):
    dest = tmp_path_factory.mktemp("artifacts")
    run_training(Config.load(), dest=dest)
    return dest


@pytest.fixture
def scorer(artifact_dir):
    """Function-scoped on purpose.

    score_transaction() mutates the profile store, so a module-scoped scorer
    would let repeated calls drive DailyAccountVolume far above its training
    maximum of 2 and make the assertions order-dependent. Training runs once;
    each test gets a fresh profile store.
    """
    return Scorer(
        load_bundle(artifact_dir), threshold=Config.load().get("ensemble.threshold")
    )


NORMAL_TXN = {
    "TransactionID": "TX900001",
    "AccountID": "AC00128",
    "DeviceID": "D000380",
    "Location": "San Diego",
    "TransactionDate": "2023-12-01 10:15:00",
    "TransactionAmount": 120.00,
    "AccountBalance": 8000.00,
    "CustomerAge": 45,
    "TransactionDuration": 90,
    "LoginAttempts": 1,
    "TransactionType": "Debit",
    "Channel": "ATM",
    "CustomerOccupation": "Engineer",
}

WEIRD_TXN = {
    **NORMAL_TXN,
    "TransactionID": "TX900002",
    "TransactionAmount": 4800.00,
    "AccountBalance": 5000.00,
    "LoginAttempts": 5,
    "Channel": "Online",
}


def test_result_matches_the_documented_contract(scorer):
    result = scorer.score_transaction(NORMAL_TXN)
    assert set(result) == {
        "transaction_id",
        "scored_at",
        "ensemble",
        "detectors",
        "explanation",
        "features",
        "raw",
        "warnings",
    }


def test_ensemble_block_has_the_documented_keys(scorer):
    ensemble = scorer.score_transaction(NORMAL_TXN)["ensemble"]
    assert set(ensemble) == {
        "is_anomaly",
        "votes_for",
        "votes_total",
        "votes_required",
        "threshold",
    }
    assert ensemble["votes_total"] == 4
    assert ensemble["votes_required"] == 2


def test_only_live_detectors_appear(scorer):
    detectors = scorer.score_transaction(NORMAL_TXN)["detectors"]
    assert len(detectors) == 4
    assert [d["name"] for d in detectors] == [
        "isolation_forest",
        "lof",
        "one_class_svm",
        "dbscan",
    ]
    for entry in detectors:
        assert set(entry) == {
            "name",
            "flag",
            "score",
            "score_percentile",
            "live_scored",
        }
        assert entry["live_scored"] is True
        assert entry["flag"] in (0, 1)
        assert 0.0 <= entry["score_percentile"] <= 100.0


def test_train_only_detectors_are_absent(scorer):
    names = {d["name"] for d in scorer.score_transaction(NORMAL_TXN)["detectors"]}
    assert names.isdisjoint({"mcd", "gmm", "kmeans", "pca_reconstruction"})


def test_explanation_block_has_the_documented_keys(scorer):
    explanation = scorer.score_transaction(NORMAL_TXN)["explanation"]
    assert set(explanation) == {
        "top_features",
        "plain_english",
        "surrogate_probability",
    }
    assert explanation["plain_english"]


def test_votes_for_matches_the_detector_flags(scorer):
    result = scorer.score_transaction(WEIRD_TXN)
    assert result["ensemble"]["votes_for"] == sum(
        d["flag"] for d in result["detectors"]
    )


def test_an_obviously_weird_transaction_is_flagged(scorer):
    """A 96% account drain with five login attempts must be caught."""
    result = scorer.score_transaction(WEIRD_TXN)
    assert result["ensemble"]["is_anomaly"] is True
    assert result["ensemble"]["votes_for"] >= 2


def test_a_normal_transaction_is_not_flagged(scorer):
    """The clean result is what makes the flagged ones credible."""
    result = scorer.score_transaction(NORMAL_TXN)
    assert result["ensemble"]["is_anomaly"] is False


def test_features_block_carries_all_nineteen(scorer):
    features = scorer.score_transaction(NORMAL_TXN)["features"]
    assert len(features) == 19
    assert features["UtilizationRatio"] == pytest.approx(120.0 / 8000.0)


def test_unseen_account_is_scored_with_a_warning(scorer):
    txn = {**NORMAL_TXN, "AccountID": "AC99999"}
    result = scorer.score_transaction(txn)
    assert result["ensemble"]["votes_total"] == 4
    assert any("unseen account" in w for w in result["warnings"])


def test_transaction_predating_account_history_is_scored_with_a_warning(scorer):
    """A transaction dated before AC00128's last known training transaction
    (2023-11-13) must not silently produce a wildly out-of-distribution gap."""
    txn = {**NORMAL_TXN, "TransactionDate": "2023-06-15 14:30:00"}
    result = scorer.score_transaction(txn)
    assert result["ensemble"]["votes_total"] == 4
    assert any("predates" in w for w in result["warnings"])


def test_profile_store_updates_between_calls(scorer):
    txn = {
        **NORMAL_TXN,
        "AccountID": "AC77777",
        "DeviceID": "D77777",
        "TransactionDate": "2023-07-01 09:00:00",
    }
    first = scorer.score_transaction(txn)
    second = scorer.score_transaction(
        {**txn, "TransactionDate": "2023-07-01 09:05:00"}
    )
    assert first["features"]["DailyAccountVolume"] == 1
    assert second["features"]["DailyAccountVolume"] == 2
    assert second["features"]["TimeSinceLastTx_Hours"] == pytest.approx(5 / 60, rel=1e-6)


def test_scored_at_is_iso_8601(scorer):
    from datetime import datetime

    scored_at = scorer.score_transaction(NORMAL_TXN)["scored_at"]
    assert datetime.fromisoformat(scored_at)
