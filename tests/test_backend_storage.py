from pathlib import Path

import pytest

from backend.config import Settings
from backend.storage import (
    InMemoryTransactionLog,
    LocalArtifactSource,
    build_artifact_source,
    build_transaction_log,
)


def test_local_artifact_source_returns_its_directory(tmp_path):
    source = LocalArtifactSource(tmp_path)
    assert source.ensure_local() == Path(tmp_path)


def test_local_artifact_source_rejects_a_missing_directory(tmp_path):
    source = LocalArtifactSource(tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match="artifact directory"):
        source.ensure_local()


def test_in_memory_log_returns_newest_first():
    log = InMemoryTransactionLog()
    for i in range(3):
        log.append({"transaction_id": f"T{i}", "scored_at": f"2026-01-0{i + 1}T00:00:00"})
    recent = log.recent(limit=2)
    assert [r["transaction_id"] for r in recent] == ["T2", "T1"]
    assert log.count() == 3


def test_in_memory_log_is_bounded():
    log = InMemoryTransactionLog(maxlen=2)
    for i in range(5):
        log.append({"transaction_id": f"T{i}", "scored_at": "2026-01-01T00:00:00"})
    assert log.count() == 2
    assert [r["transaction_id"] for r in log.recent(limit=10)] == ["T4", "T3"]


def test_in_memory_log_recent_rejects_a_non_positive_limit():
    """list(self._items)[-limit:] with limit=0 slices as [0:], returning the
    entire history instead of nothing -- negative zero is zero in Python."""
    log = InMemoryTransactionLog()
    for i in range(3):
        log.append({"transaction_id": f"T{i}", "scored_at": "2026-01-01T00:00:00"})
    assert log.recent(limit=0) == []


def test_append_does_not_alias_the_caller_dict():
    log = InMemoryTransactionLog()
    payload = {"transaction_id": "T1", "scored_at": "2026-01-01T00:00:00"}
    log.append(payload)
    payload["transaction_id"] = "MUTATED"
    assert log.recent(limit=1)[0]["transaction_id"] == "T1"


def test_factories_select_local_implementations_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("ANOMALY_ARTIFACT_DIR", str(tmp_path))
    settings = Settings.from_env()
    assert isinstance(build_artifact_source(settings), LocalArtifactSource)
    assert isinstance(build_transaction_log(settings), InMemoryTransactionLog)


def test_factories_reject_an_unknown_backend(monkeypatch):
    monkeypatch.setenv("ANOMALY_TRANSACTION_LOG", "carrier-pigeon")
    settings = Settings.from_env()
    with pytest.raises(ValueError, match="unknown transaction log"):
        build_transaction_log(settings)
