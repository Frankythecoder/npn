import pytest

from backend.seed import seed_log
from backend.storage import InMemoryTransactionLog
from ml.config import Config
from ml.data.loader import load_raw
from ml.pipeline.score import Scorer
from ml.pipeline.train import run_training
from ml.storage.artifacts import load_bundle


@pytest.fixture(scope="module")
def scorer(tmp_path_factory):
    dest = tmp_path_factory.mktemp("artifacts")
    run_training(Config.load(), dest=dest)
    return Scorer(load_bundle(dest), Config.load().get("ensemble.threshold"))


def test_seed_writes_the_requested_number_of_rows(scorer):
    log = InMemoryTransactionLog()
    raw = load_raw(Config.load().get("data.csv_path"))
    written = seed_log(scorer, log, raw, limit=25)
    assert written == 25
    assert log.count() == 25


def test_seeded_rows_carry_the_result_contract(scorer):
    log = InMemoryTransactionLog()
    raw = load_raw(Config.load().get("data.csv_path"))
    seed_log(scorer, log, raw, limit=10)
    row = log.recent(limit=1)[0]
    assert set(row) >= {"transaction_id", "scored_at", "ensemble", "explanation"}
    assert row["ensemble"]["votes_total"] == 4


def test_seed_produces_a_mix_of_verdicts(scorer):
    """A feed where nothing is flagged makes a poor first impression."""
    log = InMemoryTransactionLog()
    raw = load_raw(Config.load().get("data.csv_path"))
    seed_log(scorer, log, raw, limit=200)
    verdicts = {r["ensemble"]["is_anomaly"] for r in log.recent(limit=200)}
    assert verdicts == {True, False}
