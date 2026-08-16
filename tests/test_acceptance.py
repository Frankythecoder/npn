"""Plan A acceptance: the spec 6 tables and a standalone score_transaction call."""
import pytest

from ml.config import Config
from ml.pipeline.score import Scorer
from ml.pipeline.train import format_report, run_training
from ml.storage.artifacts import load_bundle

LIVE = ["isolation_forest", "lof", "one_class_svm", "dbscan"]
TRAIN_ONLY = ["mcd", "gmm", "kmeans", "pca_reconstruction"]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    dest = tmp_path_factory.mktemp("acceptance")
    report = run_training(Config.load(), dest=dest)
    print("\n" + format_report(report))
    return report, dest


def test_spec_6_1_all_eight_detectors_flag_126_rows(trained):
    report, _ = trained
    assert sorted(report.rate_table) == sorted(LIVE + TRAIN_ONLY)
    for name in LIVE + TRAIN_ONLY:
        assert report.flag_counts[name] == 126, f"{name} flagged {report.flag_counts[name]}"


def test_spec_6_2_live_pairwise_agreement_is_moderate(trained):
    report, _ = trained
    for pair, jaccard in report.agreement.items():
        assert 0.10 <= jaccard <= 0.60, f"{pair} Jaccard {jaccard:.3f}"


def test_spec_6_3_vote_histogram_is_dominated_by_zero_votes(trained):
    report, _ = trained
    assert report.vote_histogram[0] / report.n_rows > 0.80
    assert report.vote_histogram[4] > 0, "some rows should be unanimous"


def test_spec_6_3_ensemble_rate_is_in_the_target_band(trained):
    report, _ = trained
    assert 100 <= report.ensemble_flagged <= 160
    assert 0.03 <= report.ensemble_rate <= 0.07


def test_spec_6_4_flagged_rows_have_higher_utilization(trained):
    """The ensemble must be finding drained accounts, not noise."""
    import pandas as pd

    from ml.data.loader import load_raw
    from ml.detectors.registry import live_detectors
    from ml.ensemble.voting import combine_matrix
    from ml.features.engineer import build_training_frame

    _, dest = trained
    bundle = load_bundle(dest)
    raw = load_raw(Config.load().get("data.csv_path"))
    X, _, _ = build_training_frame(raw)

    flags = {
        d.name: d.fit_flags_
        for d in live_detectors(list(bundle.detectors.values()))
    }
    _, labels = combine_matrix(flags, Config.load().get("ensemble.threshold"))

    flagged = X.loc[labels == 1, "UtilizationRatio"].mean()
    normal = X.loc[labels == 0, "UtilizationRatio"].mean()
    assert flagged > normal * 3, f"flagged {flagged:.3f} vs normal {normal:.3f}"


def test_surrogate_meets_the_configured_fidelity_floor(trained):
    report, _ = trained
    assert report.surrogate_auc >= Config.load().get("surrogate.min_auc")


def test_score_transaction_works_standalone(trained):
    """Plan A's terminating condition: no API, just the function."""
    _, dest = trained
    scorer = Scorer(load_bundle(dest), Config.load().get("ensemble.threshold"))
    result = scorer.score_transaction(
        {
            "TransactionID": "TX999999",
            "AccountID": "AC00128",
            "DeviceID": "D000380",
            "Location": "San Diego",
            "TransactionDate": "2023-08-01 03:14:00",
            "TransactionAmount": 4800.00,
            "AccountBalance": 5000.00,
            "CustomerAge": 24,
            "TransactionDuration": 12,
            "LoginAttempts": 5,
            "TransactionType": "Debit",
            "Channel": "Online",
            "CustomerOccupation": "Student",
        }
    )
    assert result["ensemble"]["is_anomaly"] is True
    assert result["ensemble"]["votes_total"] == 4
    assert len(result["detectors"]) == 4
    assert result["explanation"]["plain_english"].startswith("Flagged primarily due to")
    print("\nplain english:", result["explanation"]["plain_english"])
