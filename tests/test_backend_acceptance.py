"""Plan B acceptance: the API surface from spec section 10, end to end."""
import pytest
from fastapi.testclient import TestClient

from backend import deps
from backend.main import create_app
from backend.storage import InMemoryTransactionLog
from ml.config import Config
from ml.pipeline.score import Scorer
from ml.pipeline.train import run_training
from ml.storage.artifacts import load_bundle

EXPECTED_ROUTES = {
    ("POST", "/score"),
    ("POST", "/batch-score"),
    ("GET", "/transactions/recent"),
    ("POST", "/demo/inject"),
    ("GET", "/demo/presets"),
    ("GET", "/health"),
}


@pytest.fixture(scope="module")
def artifact_dir(tmp_path_factory):
    dest = tmp_path_factory.mktemp("artifacts")
    run_training(Config.load(), dest=dest)
    return dest


@pytest.fixture
def client(artifact_dir):
    app = create_app(load_artifacts=False)
    deps.override(
        Scorer(load_bundle(artifact_dir), Config.load().get("ensemble.threshold")),
        InMemoryTransactionLog(),
    )
    with TestClient(app) as test_client:
        yield test_client
    deps.shutdown()


def test_every_spec_route_is_present(client):
    app = client.app
    actual = {
        (method, route.path)
        for route in app.routes
        for method in getattr(route, "methods", set())
        if method in {"GET", "POST"}
    }
    assert EXPECTED_ROUTES <= actual


def test_the_pitch_flow_works_end_to_end(client):
    """Clean preset reads clean, drain preset flags, both land in the feed."""
    clean = client.post("/demo/inject", json={"preset": "normal"}).json()
    drain = client.post("/demo/inject", json={"preset": "account_drain"}).json()

    assert clean["ensemble"]["is_anomaly"] is False
    assert drain["ensemble"]["is_anomaly"] is True
    assert drain["ensemble"]["votes_for"] >= drain["ensemble"]["votes_required"]
    assert len(drain["detectors"]) == 4
    assert drain["explanation"]["top_features"]
    print("\nflagged:", drain["explanation"]["plain_english"])

    feed = client.get("/transactions/recent").json()
    assert [r["transaction_id"] for r in feed] == [
        drain["transaction_id"],
        clean["transaction_id"],
    ]


def test_no_endpoint_reimplements_scoring(client, artifact_dir):
    """/score and /demo/inject must agree given identical input.

    Both calls use the exact same account_drain preset fields -- the only
    override is TransactionID, which is not a feature and cannot affect the
    engineered frame. The scorer mutates its ProfileStore on every call, so
    without intervention the second (injected) call would see the first
    (direct) call's transaction as history -- a different
    TimeSinceLastTx_Hours, DailyAccountVolume, etc. -- and any votes_for
    difference would be a state-mutation artifact, not evidence that the two
    endpoints diverge. Rebuilding the scorer from the same artifact bundle
    before the second call gives it the same pristine profile state the
    first call saw, so the comparison is actually between two code paths
    given identical input, not between two different accounts.
    """
    from backend.presets import PRESETS

    fields = {**PRESETS["account_drain"]["fields"], "TransactionID": "CMP-1"}
    direct = client.post("/score", json=fields).json()

    # Reset to a freshly-loaded scorer so the injected call starts from the
    # same clean profile store the direct call did, rather than inheriting
    # the direct call's side effect.
    deps.override(
        Scorer(load_bundle(artifact_dir), Config.load().get("ensemble.threshold")),
        deps.get_log(),
    )
    injected = client.post(
        "/demo/inject",
        json={"preset": "account_drain", "overrides": {"TransactionID": "CMP-2"}},
    ).json()

    assert direct["features"] == injected["features"]
    assert direct["ensemble"]["votes_for"] == injected["ensemble"]["votes_for"]
    assert [d["name"] for d in direct["detectors"]] == [
        d["name"] for d in injected["detectors"]
    ]


def test_health_is_available_without_a_model():
    app = create_app(load_artifacts=False)
    with TestClient(app) as bare:
        assert bare.get("/health").json()["status"] == "ok"


def test_startup_wires_the_real_source_and_log(monkeypatch, artifact_dir):
    """Exercise deps.startup() — the code Cloud Run actually runs at cold start.

    Every other test in the suite passes load_artifacts=False and injects
    through deps.override, so the real wiring (artifact source -> load_bundle
    -> Scorer -> transaction log) was proven only by deploying. This is also
    the positive control for create_app's flag: elsewhere we assert that
    load_artifacts=False skips loading, and nothing asserted that True loads.
    """
    monkeypatch.setenv("ANOMALY_ARTIFACT_DIR", str(artifact_dir))
    monkeypatch.setenv("ANOMALY_ARTIFACT_SOURCE", "local")
    monkeypatch.setenv("ANOMALY_TRANSACTION_LOG", "memory")
    deps.shutdown()

    app = create_app(load_artifacts=True)
    with TestClient(app) as client:
        assert client.get("/health").json()["model_loaded"] is True
        scored = client.post(
            "/demo/inject", json={"preset": "account_drain"}
        )
        assert scored.status_code == 200
        assert scored.json()["ensemble"]["votes_total"] == 4
        assert len(client.get("/transactions/recent").json()) == 1

    assert deps.is_loaded() is False


def test_startup_fails_loudly_on_a_missing_artifact_directory(monkeypatch, tmp_path):
    """A misconfigured bundle path must surface, not yield a half-built app."""
    monkeypatch.setenv("ANOMALY_ARTIFACT_DIR", str(tmp_path / "absent"))
    monkeypatch.setenv("ANOMALY_ARTIFACT_SOURCE", "local")
    deps.shutdown()

    with pytest.raises(FileNotFoundError, match="artifact directory"):
        with TestClient(create_app(load_artifacts=True)):
            pass
    deps.shutdown()
