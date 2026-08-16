import pytest

from ml.config import Config
from ml.pipeline.train import TrainingReport, format_report, run_training
from ml.storage.artifacts import load_bundle


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    dest = tmp_path_factory.mktemp("artifacts")
    report = run_training(Config.load(), dest=dest)
    return report, dest


def test_every_detector_flags_the_contamination_rate(trained):
    report, _ = trained
    assert len(report.rate_table) == 8
    for name, rate in report.rate_table.items():
        assert 0.045 <= rate <= 0.055, f"{name} flagged {rate:.4f}"
        assert report.flag_counts[name] == 126, name


def test_no_detector_falls_outside_the_sane_band(trained):
    report, _ = trained
    assert report.warnings == [], report.warnings


def test_ensemble_rate_is_in_the_target_band(trained):
    report, _ = trained
    assert 0.03 <= report.ensemble_rate <= 0.07
    assert 100 <= report.ensemble_flagged <= 160


def test_vote_histogram_covers_zero_through_four(trained):
    report, _ = trained
    assert set(report.vote_histogram) == {0, 1, 2, 3, 4}
    assert sum(report.vote_histogram.values()) == 2512


def test_threshold_sweep_is_monotonically_decreasing(trained):
    report, _ = trained
    counts = [report.threshold_sweep[k] for k in sorted(report.threshold_sweep)]
    assert counts == sorted(counts, reverse=True)


def test_pairwise_agreement_covers_every_live_pair(trained):
    report, _ = trained
    assert len(report.agreement) == 6
    for pair, jaccard in report.agreement.items():
        assert 0.0 <= jaccard <= 1.0, pair


def test_dbscan_native_noise_rate_is_reported(trained):
    report, _ = trained
    assert 0.0 < report.dbscan_native_noise_rate < 0.20


def test_surrogate_fidelity_meets_the_configured_floor(trained):
    report, _ = trained
    assert report.surrogate_auc >= Config.load().get("surrogate.min_auc")


def test_bundle_is_written_and_loadable(trained):
    _, dest = trained
    bundle = load_bundle(dest)
    assert len(bundle.detectors) == 8
    assert set(bundle.scalers) == {"standard", "robust", "continuous"}
    assert bundle.surrogate is not None
    assert bundle.manifest["rate_table"]
    assert bundle.feature_artifacts.feature_columns


def test_report_formats_without_error(trained):
    report, _ = trained
    text = format_report(report)
    assert "PER-DETECTOR ANOMALY RATES" in text
    assert "isolation_forest" in text
    assert "VOTE HISTOGRAM" in text


def test_training_is_deterministic(tmp_path):
    a = run_training(Config.load(), dest=tmp_path / "a")
    b = run_training(Config.load(), dest=tmp_path / "b")
    assert a.rate_table == b.rate_table
    assert a.vote_histogram == b.vote_histogram
    assert a.ensemble_flagged == b.ensemble_flagged
