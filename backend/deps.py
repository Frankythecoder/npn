"""Process-wide wiring, resolved once at startup.

The scorer holds the loaded artifact bundle and a mutable profile store, so it is
created once per process rather than per request.
"""
from __future__ import annotations

from typing import Any

from backend.config import Settings
from backend.storage import TransactionLog, build_artifact_source, build_transaction_log
from ml.config import Config
from ml.pipeline.score import Scorer
from ml.storage.artifacts import load_bundle

_scorer: Scorer | None = None
_log: TransactionLog | None = None


def startup(settings: Settings) -> None:
    global _scorer, _log
    directory = build_artifact_source(settings).ensure_local()
    threshold = Config.load().get("ensemble.threshold")
    _scorer = Scorer(load_bundle(directory), threshold)
    _log = build_transaction_log(settings)


def shutdown() -> None:
    global _scorer, _log
    _scorer = None
    _log = None


def is_loaded() -> bool:
    return _scorer is not None


def get_scorer() -> Scorer:
    if _scorer is None:
        raise RuntimeError("scorer not loaded; startup() has not run")
    return _scorer


def get_log() -> TransactionLog:
    if _log is None:
        raise RuntimeError("transaction log not initialised; startup() has not run")
    return _log


def override(scorer: Any, log: Any) -> None:
    """Inject test doubles. Used by the test suite only."""
    global _scorer, _log
    _scorer = scorer
    _log = log
