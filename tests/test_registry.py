from ml.config import Config
from ml.detectors.registry import DETECTOR_ORDER, build_detectors, live_detectors


def test_registry_builds_seven_detectors():
    detectors = build_detectors(Config.load())
    assert len(detectors) == 7
    assert [d.name for d in detectors] == DETECTOR_ORDER


def test_exactly_four_are_live_scorable():
    live = live_detectors(build_detectors(Config.load()))
    assert [d.name for d in live] == [
        "isolation_forest",
        "lof",
        "one_class_svm",
        "dbscan",
    ]


def test_detectors_receive_config_hyperparameters():
    by_name = {d.name: d for d in build_detectors(Config.load())}
    assert by_name["dbscan"].params["eps"] == 3.0
    assert by_name["dbscan"].params["min_samples"] == 5
    assert by_name["lof"].params["n_neighbors"] == 20
    assert by_name["kmeans"].params["n_clusters"] == 8
    assert all(d.contamination == 0.05 for d in by_name.values())


def test_every_detector_declares_a_known_scaler():
    for det in build_detectors(Config.load()):
        assert det.scaler in {"standard", "robust", "continuous"}, det.name
