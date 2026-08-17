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


def test_no_endpoint_reimplements_scoring(client):
    """/score and /demo/inject must agree given identical input.

    The two calls use distinct AccountID/DeviceID pairs, both otherwise unseen
    by this test's scorer: the scorer mutates its ProfileStore on every call
    (recording each transaction as history for the next), so scoring the same
    account twice in a row -- once direct, once injected -- would make the
    second call see elevated DailyAccountVolume and a near-zero
    TimeSinceLastTx_Hours purely from the first call's side effect, not from
    any divergence between the two endpoints. Distinct, equally-fresh
    identities isolate the comparison to what this test is actually about:
    both routes computing identical votes from identical preset fields.
    """
    from backend.presets import PRESETS

    fields = {**PRESETS["account_drain"]["fields"], "TransactionID": "CMP-1"}
    direct = client.post("/score", json=fields).json()

    deps.get_log()  # same process, same scorer
    injected = client.post(
        "/demo/inject",
        json={
            "preset": "account_drain",
            "overrides": {
                "TransactionID": "CMP-2",
                "AccountID": "AC-CMP-2",
                "DeviceID": "D-CMP-2",
            },
        },
    ).json()

    assert direct["ensemble"]["votes_for"] == injected["ensemble"]["votes_for"]
    assert [d["name"] for d in direct["detectors"]] == [
        d["name"] for d in injected["detectors"]
    ]


def test_health_is_available_without_a_model(artifact_dir):
    app = create_app(load_artifacts=False)
    with TestClient(app) as bare:
        assert bare.get("/health").json()["status"] == "ok"
