import pytest
from fastapi.testclient import TestClient

from backend import deps
from backend.main import create_app
from backend.storage import InMemoryTransactionLog
from ml.config import Config
from ml.pipeline.score import Scorer
from ml.pipeline.train import run_training
from ml.storage.artifacts import load_bundle

# The verbatim header of original.csv, including the three columns the scorer has
# never read -- uploading the training file unchanged is the case this endpoint
# exists to serve.
FULL_COLUMNS = [
    "TransactionID",
    "AccountID",
    "TransactionAmount",
    "TransactionDate",
    "TransactionType",
    "Location",
    "DeviceID",
    "IP Address",
    "MerchantID",
    "Channel",
    "CustomerAge",
    "CustomerOccupation",
    "TransactionDuration",
    "LoginAttempts",
    "AccountBalance",
    "PreviousTransactionDate",
]
FULL_ROW = [
    "TX000001",
    "AC00128",
    "14.09",
    "2023-04-11 16:29:14",
    "Debit",
    "San Diego",
    "D000380",
    "162.198.218.92",
    "M015",
    "ATM",
    "70",
    "Doctor",
    "81",
    "1",
    "5112.21",
    "2024-11-04 08:08:08",
]


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


def post(client, columns, rows, **extra):
    return client.post("/score-csv", json={"columns": columns, "rows": rows, **extra})


def test_a_verbatim_original_csv_row_scores(client):
    response = post(client, FULL_COLUMNS, [FULL_ROW])
    assert response.status_code == 200
    body = response.json()
    assert body["rejected"] == []
    assert len(body["results"]) == 1
    assert body["results"][0]["transaction_id"] == "TX000001"


def test_the_result_keeps_the_score_contract(client):
    result = post(client, FULL_COLUMNS, [FULL_ROW]).json()["results"][0]
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
    assert len(result["detectors"]) == 4


def test_ignored_columns_do_not_reach_the_scorer(client):
    """extra='forbid' on TransactionIn is the guarantee this endpoint must not break."""
    raw = post(client, FULL_COLUMNS, [FULL_ROW]).json()["results"][0]["raw"]
    for column in ("IP Address", "MerchantID", "PreviousTransactionDate"):
        assert column not in raw


def test_a_file_with_no_crucial_column_is_a_422(client):
    response = post(client, ["AccountID", "DeviceID", "Location"], [["A", "D", "Houston"]])
    assert response.status_code == 422
    assert "TransactionAmount" in response.json()["detail"]


def test_one_crucial_column_is_enough_and_the_rest_are_filled(client):
    response = post(client, ["TransactionAmount"], [["4800.00"]])
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 1
    joined = " ".join(body["results"][0]["warnings"])
    assert "AccountBalance missing from the file" in joined


def test_fill_warnings_join_the_engineered_warnings(client):
    """A synthetic account must still produce transform_one's unseen-account note."""
    warnings = post(client, ["TransactionAmount"], [["4800.00"]]).json()["results"][0][
        "warnings"
    ]
    assert any("unseen account" in w for w in warnings)
    assert any("missing from the file" in w for w in warnings)


def test_a_null_cell_rejects_only_that_row(client):
    rows = [FULL_ROW, [*FULL_ROW[:2], "", *FULL_ROW[3:]]]
    body = post(client, FULL_COLUMNS, rows).json()
    assert len(body["results"]) == 1
    assert len(body["rejected"]) == 1
    assert body["rejected"][0]["row"] == 3
    assert "TransactionAmount" in body["rejected"][0]["reason"]


def test_a_rejected_row_never_reaches_the_log(client):
    rows = [FULL_ROW, [*FULL_ROW[:2], "", *FULL_ROW[3:]]]
    post(client, FULL_COLUMNS, rows)
    assert len(client.get("/transactions/recent").json()) == 1


def test_a_row_the_ml_layer_rejects_becomes_a_rejected_row_not_a_500(client):
    """A zero balance raises ValueError in transform_one; the batch must survive it."""
    bad = [*FULL_ROW]
    bad[FULL_COLUMNS.index("AccountBalance")] = "0"
    body = post(client, FULL_COLUMNS, [FULL_ROW, bad]).json()
    assert len(body["results"]) == 1
    assert len(body["rejected"]) == 1
    assert "AccountBalance" in body["rejected"][0]["reason"]


def test_results_land_in_the_shared_feed(client):
    post(client, FULL_COLUMNS, [FULL_ROW])
    recent = client.get("/transactions/recent").json()
    assert [r["transaction_id"] for r in recent] == ["TX000001"]


def test_start_row_is_echoed_in_rejection_line_numbers(client):
    blank = [*FULL_ROW[:2], "", *FULL_ROW[3:]]
    body = post(client, FULL_COLUMNS, [blank], start_row=502).json()
    assert body["rejected"][0]["row"] == 502


def test_an_empty_row_list_is_a_422(client):
    assert post(client, FULL_COLUMNS, []).status_code == 422


def test_a_chunk_above_the_cap_is_a_422(client):
    assert post(client, FULL_COLUMNS, [FULL_ROW] * 501).status_code == 422


# ---------- profile-store isolation ----------
#
# The bundle ships the profile store training left behind, which has already
# observed every row of original.csv. Scoring those rows against it makes the
# self-inclusive daily counters read 2 instead of 1, which is far outside the
# training distribution, so the whole file flags. seed.py hits the same wall and
# solves it the same way: give the replay its own store.


def volume(body, index=0):
    return body["results"][index]["features"]["DailyAccountVolume"]


def test_an_upload_does_not_see_the_training_stores_history(client):
    body = post(client, FULL_COLUMNS, [FULL_ROW]).json()
    # 1, not 2: the row is the first the batch store has seen for this account.
    assert volume(body) == 1.0


def test_an_upload_leaves_the_live_scorers_store_untouched(client):
    before = deps.get_scorer().profiles
    post(client, FULL_COLUMNS, [FULL_ROW] * 3, upload_id="u1")
    assert deps.get_scorer().profiles is before
    # A preset fired after an upload must behave exactly as it did before one.
    assert client.post("/demo/inject", json={"preset": "normal"}).status_code == 200


def test_rows_within_one_chunk_accumulate_history(client):
    body = post(client, FULL_COLUMNS, [FULL_ROW, FULL_ROW]).json()
    assert [volume(body, 0), volume(body, 1)] == [1.0, 2.0]


def test_chunks_of_one_upload_share_history(client):
    first = post(client, FULL_COLUMNS, [FULL_ROW], upload_id="u1").json()
    second = post(
        client, FULL_COLUMNS, [FULL_ROW], upload_id="u1", start_row=502
    ).json()
    assert volume(first) == 1.0
    assert volume(second) == 2.0


def test_a_new_upload_starts_from_a_clean_store(client):
    post(client, FULL_COLUMNS, [FULL_ROW], upload_id="u1")
    body = post(client, FULL_COLUMNS, [FULL_ROW], upload_id="u2").json()
    assert volume(body) == 1.0


def test_the_store_is_restored_even_when_a_row_raises(client):
    before = deps.get_scorer().profiles
    bad = [*FULL_ROW]
    bad[FULL_COLUMNS.index("AccountBalance")] = "0"
    post(client, FULL_COLUMNS, [bad])
    assert deps.get_scorer().profiles is before


def test_the_existing_score_endpoint_still_forbids_unknown_fields(client):
    """Guards the reason /score-csv is a separate endpoint rather than a loosened one."""
    payload = {
        "TransactionID": "TX1",
        "AccountID": "AC00128",
        "DeviceID": "D000380",
        "Location": "San Diego",
        "TransactionDate": "2023-12-01 10:15:00",
        "TransactionAmount": 120.0,
        "AccountBalance": 8000.0,
        "CustomerAge": 45,
        "TransactionDuration": 90,
        "LoginAttempts": 1,
        "TransactionType": "Debit",
        "Channel": "ATM",
        "CustomerOccupation": "Engineer",
        "MerchantID": "M015",
    }
    assert client.post("/score", json=payload).status_code == 422
