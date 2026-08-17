"""Storage seams between the API and its environment.

Two narrow interfaces keep the cloud out of the service's core. Each has a local
implementation used by every test, and a GCP implementation in backend/gcp.py used
only in deployment. The GCP module is imported lazily by the factories below, so
neither the test suite nor local development needs the google-cloud libraries.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Protocol

from backend.config import Settings

DEFAULT_LOG_CAPACITY = 1000


class ArtifactSource(Protocol):
    def ensure_local(self) -> Path:
        """Return a local directory containing the artifact bundle."""


class TransactionLog(Protocol):
    def append(self, result: dict) -> None: ...
    def recent(self, limit: int) -> list[dict]: ...
    def count(self) -> int: ...


class LocalArtifactSource:
    """Reads the bundle straight from a directory on disk."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def ensure_local(self) -> Path:
        if not self.directory.exists():
            raise FileNotFoundError(
                f"artifact directory not found: {self.directory}. "
                "Run `python -m ml.pipeline.train` first."
            )
        return self.directory


class InMemoryTransactionLog:
    """Bounded, newest-first store. Process-local and lost on restart."""

    def __init__(self, maxlen: int = DEFAULT_LOG_CAPACITY) -> None:
        self._items: deque[dict] = deque(maxlen=maxlen)

    def append(self, result: dict) -> None:
        # Copy so a later mutation by the caller cannot rewrite history.
        self._items.append(dict(result))

    def recent(self, limit: int) -> list[dict]:
        items = list(self._items)[-limit:]
        return [dict(item) for item in reversed(items)]

    def count(self) -> int:
        return len(self._items)


def build_artifact_source(settings: Settings) -> ArtifactSource:
    if settings.artifact_source == "local":
        return LocalArtifactSource(settings.artifact_dir)
    if settings.artifact_source == "gcs":
        from backend.gcp import GCSArtifactSource

        return GCSArtifactSource(settings.gcs_bucket, settings.gcs_prefix)
    raise ValueError(f"unknown artifact source: {settings.artifact_source!r}")


def build_transaction_log(settings: Settings) -> TransactionLog:
    if settings.transaction_log == "memory":
        return InMemoryTransactionLog()
    if settings.transaction_log == "firestore":
        from backend.gcp import FirestoreTransactionLog

        return FirestoreTransactionLog(settings.firestore_collection)
    raise ValueError(f"unknown transaction log: {settings.transaction_log!r}")
