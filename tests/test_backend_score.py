import pytest
from fastapi.testclient import TestClient

from backend import deps
from backend.main import create_app
from backend.storage import InMemoryTransactionLog
from ml.config import Config
from ml.pipeline.score import Scorer
from ml.pipeline.train import run_training
from ml.storage.artifacts import load_bundle

NORMAL = {
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
WEIRD = {
    **NORMAL,
    "TransactionID": "TX900002",
    "TransactionAmount": 4800.00,
    "AccountBalance": 5000.00,
    "LoginAttempts": 5,
    "Channel": "Online",
}


@pytest.fixture(scope="module")
def artifact_dir(tmp_path_factory):
    dest = tmp_path_factory.mktemp("artifacts")
    run_training(Config.load(), dest=dest)
    return dest


@pytest.fixture
def client(artifact_dir):
    """Fresh scorer and log per test — the scorer mutates its profile store."""
    app = create_app(load_artifacts=False)
    threshold = Config.load().get("ensemble.threshold")
    deps.override(Scorer(load_bundle(artifact_dir), threshold), InMemoryTransactionLog())
    with TestClient(app) as test_client:
        yield test_client
    deps.shutdown()


def test_score_returns_the_result_contract(client):
    response = client.post("/score", json=NORMAL)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "transaction_id",
        "scored_at",
        "ensemble",
        "detectors",
        "explanation",
        "features",
        "raw",
        "warnings",
    }
    assert body["ensemble"]["votes_total"] == 4
    assert len(body["detectors"]) == 4


def test_score_flags_a_weird_transaction_and_clears_a_normal_one(client):
    assert client.post("/score", json=NORMAL).json()["ensemble"]["is_anomaly"] is False
    assert client.post("/score", json=WEIRD).json()["ensemble"]["is_anomaly"] is True


def test_score_persists_to_the_log(client):
    client.post("/score", json=NORMAL)
    recent = client.get("/transactions/recent").json()
    assert len(recent) == 1
    assert recent[0]["transaction_id"] == "TX900001"


def test_recent_is_newest_first_and_respects_limit(client):
    for i in range(3):
        client.post("/score", json={**NORMAL, "TransactionID": f"TX{i}"})
    recent = client.get("/transactions/recent?limit=2").json()
    assert [r["transaction_id"] for r in recent] == ["TX2", "TX1"]


def test_recent_rejects_a_limit_above_the_maximum(client):
    assert client.get("/transactions/recent?limit=99999").status_code == 422


def test_batch_score_returns_one_result_per_input(client):
    response = client.post("/batch-score", json={"transactions": [NORMAL, WEIRD]})
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    assert results[0]["ensemble"]["is_anomaly"] is False
    assert results[1]["ensemble"]["is_anomaly"] is True


def test_batch_score_rejects_an_empty_list(client):
    assert client.post("/batch-score", json={"transactions": []}).status_code == 422


def test_missing_required_field_is_a_422(client):
    payload = {k: v for k, v in NORMAL.items() if k != "AccountID"}
    assert client.post("/score", json=payload).status_code == 422


def test_zero_balance_is_a_422_not_a_500(client):
    """The ml layer raises ValueError; the API must translate it, not leak a 500."""
    response = client.post("/score", json={**NORMAL, "AccountBalance": 0})
    assert response.status_code == 422
    assert "AccountBalance" in response.json()["detail"]


def test_warnings_are_surfaced_in_the_response(client):
    response = client.post("/score", json={**NORMAL, "AccountID": "AC99999"})
    assert any("unseen account" in w for w in response.json()["warnings"])
