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


def test_an_incomplete_file_is_a_422_naming_what_is_missing(client):
    """Nothing is substituted, so a file with a gap cannot be scored at all."""
    response = post(client, ["TransactionAmount"], [["4800.00"]])
    assert response.status_code == 422
    detail = response.json()["detail"]
    for column in ("AccountBalance", "Channel", "CustomerOccupation", "Location"):
        assert column in detail, column
    assert "nothing is substituted" in detail.lower()


# Every scoring column, with the identity three deliberately left out.
REQUIRED_ONLY = [
    "TransactionAmount",
    "AccountBalance",
    "CustomerAge",
    "TransactionDuration",
    "LoginAttempts",
    "TransactionType",
    "Channel",
    "CustomerOccupation",
    "Location",
]
REQUIRED_ONLY_ROW = ["4800.00", "5000", "40", "90", "1", "Debit", "ATM", "Student", "Houston"]


def test_a_file_without_identity_columns_still_scores(client):
    """Identity is synthesised rather than filled, so its absence is survivable
    where a missing scoring column is not."""
    response = post(client, REQUIRED_ONLY, [REQUIRED_ONLY_ROW])
    assert response.status_code == 200
    assert len(response.json()["results"]) == 1


def test_a_synthesised_account_still_produces_the_unseen_account_note(client):
    """transform_one's own warning must survive; it is how the operator learns
    the row's history features had nothing to work from."""
    warnings = post(client, REQUIRED_ONLY, [REQUIRED_ONLY_ROW]).json()["results"][0][
        "warnings"
    ]
    assert any("unseen account" in w for w in warnings)
    assert not any("filled with the training default" in w for w in warnings)


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
    post(
        client,
        FULL_COLUMNS,
        [_row(TransactionID=f"TX00000{i}") for i in (1, 2, 3)],
        upload_id="u1",
    )
    assert deps.get_scorer().profiles is before
    # A preset fired after an upload must behave exactly as it did before one.
    assert client.post("/demo/inject", json={"preset": "normal"}).status_code == 200


# Distinct transaction ids throughout: these are two transactions on ONE
# account, which is what accumulating history means. Repeating a single id
# would be a duplicate, and is now dropped as one.


def test_rows_within_one_chunk_accumulate_history(client):
    body = post(
        client, FULL_COLUMNS, [FULL_ROW, _row(TransactionID="TX000002")]
    ).json()
    assert [volume(body, 0), volume(body, 1)] == [1.0, 2.0]


def test_chunks_of_one_upload_share_history(client):
    first = post(client, FULL_COLUMNS, [FULL_ROW], upload_id="u1").json()
    second = post(
        client,
        FULL_COLUMNS,
        [_row(TransactionID="TX000002")],
        upload_id="u1",
        start_row=502,
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


# ---------- the opt-in fill path ----------


def test_fill_missing_accepts_the_same_file_the_default_refuses(client):
    """The opt-in path, end to end."""
    refused = post(client, ["TransactionAmount"], [["4800.00"]])
    assert refused.status_code == 422

    filled = post(client, ["TransactionAmount"], [["4800.00"]], fill_missing=True)
    assert filled.status_code == 200
    body = filled.json()
    assert len(body["results"]) == 1
    joined = " ".join(body["results"][0]["warnings"])
    assert "AccountBalance missing from the file" in joined
    assert "filled with the training default" in joined


def test_fill_missing_is_off_unless_asked_for(client):
    """A complete file must not acquire fill warnings just because the flag
    exists, and the default must stay strict."""
    body = post(client, FULL_COLUMNS, [FULL_ROW]).json()
    assert not any(
        "filled with the training default" in w
        for w in body["results"][0]["warnings"]
    )


def test_fill_missing_cannot_rescue_a_non_transaction_file(client):
    response = post(
        client, ["IP Address", "MerchantID"], [["1.2.3.4", "M01"]], fill_missing=True
    )
    assert response.status_code == 422


# ---------- validation and duplicate removal, end to end ----------


def _row(**overrides):
    values = dict(zip(FULL_COLUMNS, FULL_ROW))
    values.update(overrides)
    return [values[c] for c in FULL_COLUMNS]


def test_an_invalid_row_is_dropped_and_the_valid_one_still_scores(client):
    body = post(client, FULL_COLUMNS, [_row(), _row(CustomerAge="900")]).json()
    assert len(body["results"]) == 1
    assert len(body["rejected"]) == 1
    assert "CustomerAge" in body["rejected"][0]["reason"]


def test_an_unknown_channel_never_reaches_the_model(client):
    body = post(client, FULL_COLUMNS, [_row(Channel="Pigeon")]).json()
    assert body["results"] == []
    assert "Channel" in body["rejected"][0]["reason"]


def test_an_anomalous_but_valid_row_is_still_scored(client):
    """The validator must not delete the account-drain case."""
    body = post(
        client, FULL_COLUMNS, [_row(TransactionAmount="9000", AccountBalance="100")]
    ).json()
    assert len(body["results"]) == 1
    assert body["rejected"] == []


def test_duplicates_are_dropped_across_chunks_of_one_upload(client):
    """Chunks share an upload_id, so the second copy is caught even though it
    arrives in a separate request."""
    first = post(
        client, FULL_COLUMNS, [_row()], upload_id="u1", start_row=2
    ).json()
    second = post(
        client, FULL_COLUMNS, [_row()], upload_id="u1", start_row=502
    ).json()
    assert len(first["results"]) == 1
    assert second["results"] == []
    assert "duplicate" in second["rejected"][0]["reason"]


def test_a_new_upload_starts_from_a_clean_id_set(client):
    post(client, FULL_COLUMNS, [_row()], upload_id="u1")
    body = post(client, FULL_COLUMNS, [_row()], upload_id="u2").json()
    assert len(body["results"]) == 1, "a new upload inherited the previous one's ids"
