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


def test_no_seeded_sentence_negates_a_one_hot(scorer):
    """The check that would have caught every explainer defect so far.

    All four defects in the explanation layer were found by reading rendered
    sentences, never by a failing test: unremarkable features headlining,
    negated one-hots qualifying, tied minimums reading as extreme, and the
    fallback re-admitting what the filter excluded. Each passed every test at
    the time. This asserts a property over real output instead of a rigged
    fixture -- it would have failed on all four.
    """
    import re

    log = InMemoryTransactionLog(maxlen=500)
    raw = load_raw(Config.load().get("data.csv_path"))
    seed_log(scorer, log, raw, limit=300)

    offenders = [
        row["explanation"]["plain_english"]
        for row in log.recent(limit=300)
        if re.search(r"\bnot an? ", row["explanation"]["plain_english"])
    ]
    assert offenders == [], f"{len(offenders)} sentences negate a one-hot, e.g. {offenders[:1]}"


def test_every_seeded_sentence_is_well_formed(scorer):
    """No empty or truncated copy reaches the feed the dashboard opens on."""
    log = InMemoryTransactionLog(maxlen=500)
    raw = load_raw(Config.load().get("data.csv_path"))
    seed_log(scorer, log, raw, limit=300)

    for row in log.recent(limit=300):
        sentence = row["explanation"]["plain_english"]
        assert sentence.strip()
        assert sentence.endswith(".")
        assert "  " not in sentence


def test_seed_restores_the_callers_profile_store(scorer):
    """Ruling B9 swaps in a fresh store; the restore must actually happen."""
    original = scorer.profiles
    raw = load_raw(Config.load().get("data.csv_path"))
    seed_log(scorer, InMemoryTransactionLog(), raw, limit=10)
    assert scorer.profiles is original
