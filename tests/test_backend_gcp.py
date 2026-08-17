import pytest

from backend.gcp import FirestoreTransactionLog, GCSArtifactSource


class FakeBlob:
    def __init__(self, name, payload=b"x"):
        self.name = name
        self._payload = payload
        self.downloaded_to = None
        self.download_calls = 0

    def download_to_filename(self, path):
        self.download_calls += 1
        self.downloaded_to = path
        with open(path, "wb") as fh:
            fh.write(self._payload)


class FakeBucket:
    def __init__(self, blobs):
        self._blobs = blobs
        self.list_blobs_calls = 0

    def list_blobs(self, prefix=None):
        self.list_blobs_calls += 1
        return [b for b in self._blobs if b.name.startswith(prefix or "")]


class FakeGCSClient:
    def __init__(self, bucket):
        self._bucket = bucket

    def bucket(self, name):
        return self._bucket


class FakeDocument:
    def __init__(self, data):
        self._data = data

    def to_dict(self):
        return dict(self._data)


class FakeQuery:
    def __init__(self, docs):
        self._docs = docs

    def order_by(self, field, direction=None):
        self._docs = sorted(self._docs, key=lambda d: d._data.get(field, ""), reverse=True)
        return self

    def limit(self, n):
        self._docs = self._docs[:n]
        return self

    def stream(self):
        return iter(self._docs)


class FakeCollection:
    def __init__(self):
        self.docs = []

    def add(self, data):
        self.docs.append(FakeDocument(data))

    def order_by(self, field, direction=None):
        return FakeQuery(list(self.docs)).order_by(field, direction)

    def stream(self):
        return iter(self.docs)


class FakeFirestoreClient:
    def __init__(self):
        self._collections = {}

    def collection(self, name):
        return self._collections.setdefault(name, FakeCollection())


def test_gcs_source_downloads_every_blob_under_the_prefix(tmp_path):
    blobs = [
        FakeBlob("artifacts/latest/manifest.json"),
        FakeBlob("artifacts/latest/detectors/lof.pkl"),
        FakeBlob("other/ignored.pkl"),
    ]
    source = GCSArtifactSource(
        "bucket", "artifacts/latest", cache_dir=tmp_path, client=FakeGCSClient(FakeBucket(blobs))
    )
    local = source.ensure_local()
    assert (local / "manifest.json").exists()
    assert (local / "detectors" / "lof.pkl").exists()
    assert not (local / "ignored.pkl").exists()


def test_gcs_source_is_idempotent(tmp_path):
    """A second ensure_local() must skip re-listing and re-downloading, not
    merely return the same path -- cache_dir is a fixed instance attribute
    returned either way, so asserting equal return values alone would pass
    even without the `_ready` guard."""
    blob = FakeBlob("artifacts/latest/manifest.json")
    bucket = FakeBucket([blob])
    source = GCSArtifactSource(
        "bucket", "artifacts/latest", cache_dir=tmp_path, client=FakeGCSClient(bucket)
    )

    first = source.ensure_local()
    assert bucket.list_blobs_calls == 1
    assert blob.download_calls == 1

    second = source.ensure_local()
    assert second == first
    assert bucket.list_blobs_calls == 1
    assert blob.download_calls == 1


def test_gcs_source_rejects_an_empty_prefix(tmp_path):
    source = GCSArtifactSource(
        "bucket", "artifacts/latest", cache_dir=tmp_path, client=FakeGCSClient(FakeBucket([]))
    )
    with pytest.raises(FileNotFoundError, match="no artifacts found"):
        source.ensure_local()


def test_firestore_log_round_trips_newest_first():
    log = FirestoreTransactionLog("scored", client=FakeFirestoreClient())
    for i in range(3):
        log.append({"transaction_id": f"T{i}", "scored_at": f"2026-01-0{i + 1}T00:00:00"})
    assert [r["transaction_id"] for r in log.recent(limit=2)] == ["T2", "T1"]
    assert log.count() == 3


def test_firestore_append_does_not_alias_the_caller_dict():
    log = FirestoreTransactionLog("scored", client=FakeFirestoreClient())
    payload = {"transaction_id": "T1", "scored_at": "2026-01-01T00:00:00"}
    log.append(payload)
    payload["transaction_id"] = "MUTATED"
    assert log.recent(limit=1)[0]["transaction_id"] == "T1"
