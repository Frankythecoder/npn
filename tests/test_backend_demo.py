import pytest
from fastapi.testclient import TestClient

from backend import deps
from backend.main import create_app
from backend.presets import PRESETS
from backend.storage import InMemoryTransactionLog
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
def client(artifact_dir):
    app = create_app(load_artifacts=False)
    threshold = Config.load().get("ensemble.threshold")
    deps.override(Scorer(load_bundle(artifact_dir), threshold), InMemoryTransactionLog())
    with TestClient(app) as test_client:
        yield test_client
    deps.shutdown()


def test_presets_endpoint_lists_every_scenario(client):
    body = client.get("/demo/presets").json()
    assert {p["name"] for p in body["presets"]} == set(PRESETS)
    for preset in body["presets"]:
        assert preset["label"]
        assert preset["description"]
        assert isinstance(preset["fields"], dict)


def test_normal_preset_returns_clean(client):
    """A demo where every input is flagged proves nothing."""
    body = client.post("/demo/inject", json={"preset": "normal"}).json()
    assert body["ensemble"]["is_anomaly"] is False


@pytest.mark.parametrize("name", ["account_drain", "credential_stuffing"])
def test_anomalous_presets_are_flagged(client, name):
    body = client.post("/demo/inject", json={"preset": name}).json()
    assert body["ensemble"]["is_anomaly"] is True
    assert body["explanation"]["plain_english"].startswith("Flagged primarily due to")


def test_overrides_take_precedence_over_the_preset(client):
    body = client.post(
        "/demo/inject",
        json={"preset": "normal", "overrides": {"LoginAttempts": 5, "TransactionAmount": 4900.0}},
    ).json()
    assert body["raw"]["LoginAttempts"] == 5
    assert body["raw"]["TransactionAmount"] == 4900.0


def test_inject_persists_to_the_log(client):
    client.post("/demo/inject", json={"preset": "account_drain"})
    assert len(client.get("/transactions/recent").json()) == 1


def test_unknown_preset_is_a_422(client):
    response = client.post("/demo/inject", json={"preset": "nope"})
    assert response.status_code == 422
    assert "nope" in response.json()["detail"]


def test_inject_generates_a_transaction_id_when_absent(client):
    body = client.post("/demo/inject", json={"preset": "normal"}).json()
    assert body["transaction_id"]


def test_rapid_fire_preset_increments_the_daily_count(client):
    first = client.post("/demo/inject", json={"preset": "rapid_fire"}).json()
    second = client.post("/demo/inject", json={"preset": "rapid_fire"}).json()
    assert second["features"]["DailyAccountVolume"] > first["features"]["DailyAccountVolume"]
