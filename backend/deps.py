"""Process-wide wiring, resolved once at startup.

The scorer holds the loaded artifact bundle and a mutable profile store, so it is
created once per process rather than per request.
"""
from __future__ import annotations

from typing import Any

from backend.config import Settings
from backend.csvingest import defaults_from_bundle
from backend.storage import TransactionLog, build_artifact_source, build_transaction_log
from ml.config import Config
from ml.features.engineer import ProfileStore
from ml.pipeline.score import Scorer
from ml.storage.artifacts import load_bundle

_scorer: Scorer | None = None
_log: TransactionLog | None = None
_csv_defaults: dict | None = None
# One slot, not a cache: only the upload currently being received needs a store,
# and a new upload replaces the previous one rather than accumulating.
_batch_profiles: tuple[str, ProfileStore] | None = None


def startup(settings: Settings) -> None:
    global _scorer, _log
    directory = build_artifact_source(settings).ensure_local()
    threshold = Config.load().get("ensemble.threshold")
    _scorer = Scorer(load_bundle(directory), threshold)
    _log = build_transaction_log(settings)


def shutdown() -> None:
    global _scorer, _log, _csv_defaults, _batch_profiles
    _scorer = None
    _log = None
    _csv_defaults = None
    _batch_profiles = None


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


def get_csv_defaults() -> dict:
    """Fill values for CSV columns an upload did not supply.

    Derived from the loaded bundle on first use rather than at startup, so a
    deployment that never receives an upload never pays for it.
    """
    global _csv_defaults
    if _csv_defaults is None:
        _csv_defaults = defaults_from_bundle(get_scorer().bundle)
    return _csv_defaults


def get_batch_profiles(upload_id: str) -> ProfileStore:
    """A profile store scoped to one uploaded file.

    The store the bundle ships with has already observed every training row, so
    scoring a file drawn from that data against it makes the self-inclusive daily
    counters read one higher than they did in training -- a level the detectors
    almost never saw, which flags the whole file. seed.py documents the same trap
    for its replay. An upload therefore gets its own store: history within the
    file counts, history the service happens to remember does not.

    Chunks of one upload share a store so an account spanning a chunk boundary
    still accumulates. An empty id gets an unshared store rather than colliding
    with whatever ran last.
    """
    global _batch_profiles
    if not upload_id:
        return ProfileStore()
    if _batch_profiles is None or _batch_profiles[0] != upload_id:
        _batch_profiles = (upload_id, ProfileStore())
    return _batch_profiles[1]


def override(scorer: Any, log: Any) -> None:
    """Inject test doubles. Used by the test suite only."""
    global _scorer, _log, _csv_defaults, _batch_profiles
    _scorer = scorer
    _log = log
    # Defaults are derived from the scorer's bundle, so a new scorer must not
    # inherit the previous one's cached values.
    _csv_defaults = None
    _batch_profiles = None
