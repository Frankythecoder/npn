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
    assert len(report.rate_table) == 7
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
    assert len(bundle.detectors) == 7
    assert set(bundle.scalers) == {"standard", "robust", "continuous"}
    assert bundle.surrogate is not None
    assert bundle.manifest["rate_table"]
    assert bundle.feature_artifacts.feature_columns


def test_manifest_records_library_versions_and_a_config_hash(trained):
    """Spec 9: pinning exists specifically to stop a scikit-learn
    minor-version drift from breaking unpickling -- without recording the
    versions that produced the bundle, nothing catches that drift."""
    _, dest = trained
    bundle = load_bundle(dest)
    versions = bundle.manifest["versions"]
    for lib in ("sklearn", "catboost", "numpy", "pandas", "shap"):
        assert lib in versions
        assert isinstance(versions[lib], str) and versions[lib]

    assert isinstance(bundle.manifest["config_hash"], str)
    assert len(bundle.manifest["config_hash"]) == 64  # sha256 hex digest

    # Deterministic: hashing the same config twice must produce the same hash.
    from ml.pipeline.train import _config_hash

    assert bundle.manifest["config_hash"] == _config_hash(Config.load())


def test_manifest_records_agreement_and_threshold_sweep(trained):
    """Both are computed and printed by format_report but must also be
    persisted, not just displayed."""
    report, dest = trained
    bundle = load_bundle(dest)
    assert bundle.manifest["agreement"] == report.agreement
    assert set(bundle.manifest["threshold_sweep"]) == {"1", "2", "3", "4"}
    assert bundle.manifest["threshold_sweep"]["1"] == report.threshold_sweep[1]


def test_manifest_vote_histogram_keys_survive_a_reload(trained):
    """manifest['vote_histogram'] uses int keys in-process; JSON turns them
    into strings on disk. Both sides must use string keys so
    bundle.manifest['vote_histogram']['0'] does not KeyError after a
    reload while report.vote_histogram[0] still works in-process."""
    report, dest = trained
    bundle = load_bundle(dest)
    assert set(bundle.manifest["vote_histogram"]) == {"0", "1", "2", "3", "4"}
    for v in range(5):
        assert bundle.manifest["vote_histogram"][str(v)] == report.vote_histogram[v]


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


# ---------- the held-out split ----------


def test_the_split_partitions_every_row_exactly_once(trained):
    report, _ = trained
    sizes = report.split_sizes
    assert set(sizes) == {"train", "validation", "test"}
    assert sum(sizes.values()) == report.n_rows


def test_the_split_honours_the_configured_70_20_10(trained):
    report, _ = trained
    n = report.n_rows
    sizes = report.split_sizes
    assert sizes["train"] / n == pytest.approx(0.70, abs=0.01)
    assert sizes["validation"] / n == pytest.approx(0.20, abs=0.01)
    assert sizes["test"] / n == pytest.approx(0.10, abs=0.01)


def test_every_detector_is_scored_on_every_split(trained):
    report, _ = trained
    assert set(report.split_rates) == set(report.rate_table)
    for name, rates in report.split_rates.items():
        assert set(rates) == {"train", "validation", "test"}, name


def test_the_frozen_threshold_still_cuts_roughly_the_rate_on_held_out_rows(trained):
    """The point of the split: a threshold fitted on train must generalise.

    Wide bounds deliberately -- this catches a threshold that collapsed or ran
    away on unseen rows, not ordinary sampling noise on a 251-row test split.
    """
    report, _ = trained
    for name, rates in report.split_rates.items():
        for part in ("validation", "test"):
            assert 0.01 <= rates[part] <= 0.15, f"{name} {part} {rates[part]:.2%}"


def test_the_shipped_detectors_are_still_fitted_on_every_row(trained):
    """Holding rows back must not cost the bundle anything.

    Spec 6.1 fixes each detector at 126 flagged rows, which is the
    contamination rate over all 2,512 -- not over the 70% used to evaluate.
    """
    report, _ = trained
    for name, count in report.flag_counts.items():
        assert count == 126, f"{name} flagged {count}"


def test_the_split_is_recorded_in_the_manifest(trained):
    _, dest = trained
    manifest = load_bundle(dest).manifest
    assert sum(manifest["split_sizes"].values()) == manifest["n_rows"]
    assert set(manifest["split_rates"]) == set(manifest["rate_table"])
