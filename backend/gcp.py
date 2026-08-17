"""Cloud Storage and Firestore implementations of the storage seams.

The google-cloud libraries are imported inside __init__ rather than at module
scope, so importing this module costs nothing and the test suite never needs them
installed. Both classes accept an injected client so their logic is testable
against fakes without credentials or network access.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any


class GCSArtifactSource:
    """Downloads the artifact bundle from Cloud Storage to a local cache once."""

    def __init__(
        self,
        bucket: str,
        prefix: str,
        cache_dir: str | Path | None = None,
        client: Any = None,
    ) -> None:
        if client is None:
            from google.cloud import storage  # imported lazily; deployment only

            client = storage.Client()
        self._client = client
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "anomaly-artifacts"
        self._ready = False

    def ensure_local(self) -> Path:
        if self._ready:
            return self.cache_dir

        bucket = self._client.bucket(self.bucket)
        blobs = list(bucket.list_blobs(prefix=self.prefix + "/"))
        if not blobs:
            raise FileNotFoundError(
                f"no artifacts found at gs://{self.bucket}/{self.prefix}. "
                "Upload a trained bundle before starting the service."
            )

        for blob in blobs:
            relative = blob.name[len(self.prefix):].lstrip("/")
            if not relative:
                continue
            destination = self.cache_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(destination))

        self._ready = True
        return self.cache_dir


class FirestoreTransactionLog:
    """One document per scored transaction, ordered by scored_at descending.

    Ordering on a single field is auto-indexed, so no composite index is needed.
    """

    def __init__(self, collection: str, client: Any = None) -> None:
        if client is None:
            from google.cloud import firestore  # imported lazily; deployment only

            client = firestore.Client()
        self._client = client
        self.collection = collection

    def append(self, result: dict) -> None:
        self._client.collection(self.collection).add(dict(result))

    def recent(self, limit: int) -> list[dict]:
        query = self._client.collection(self.collection).order_by(
            "scored_at", direction="DESCENDING"
        )
        return [doc.to_dict() for doc in query.limit(limit).stream()]

    def count(self) -> int:
        return sum(1 for _ in self._client.collection(self.collection).stream())
