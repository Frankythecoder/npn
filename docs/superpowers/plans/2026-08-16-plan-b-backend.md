# Plan B — Backend and Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap `score_transaction()` in a FastAPI service, back it with a pluggable artifact source and transaction log, and make it deployable to Cloud Run with a README a reader can follow from an empty GCP project.

**Architecture:** The API owns no scoring logic — every endpoint delegates to Plan A's `score_transaction()`. Two small interfaces isolate the cloud: an `ArtifactSource` that yields a local directory of model artifacts, and a `TransactionLog` that stores and retrieves scored results. Each has a local implementation used by every test and a GCP implementation used only in deployment, behind guarded imports so the test suite never needs cloud credentials or libraries.

**Tech Stack:** FastAPI 0.115.9, uvicorn 0.40.0, pydantic 2.12.5, httpx 0.28.1 (TestClient). Deployment-only: google-cloud-storage, google-cloud-firestore (not installed locally by design).

**Spec:** `docs/superpowers/specs/2026-08-16-anomaly-detection-design.md` — §10 Backend, §11 Deployment.

**Branch:** `plan-a-ml-core` (continues Plan A's branch; no separate branch).

## Global Constraints

- **`final.ipynb` and `original.csv` are read-only.** Never modified, never re-executed, never re-added to git.
- **Plan A's `/ml` package is closed for feature work.** Import from it; do not modify it. The one permitted exception is Task 7's config additions. If a task seems to need an `/ml` change, stop and report BLOCKED.
- **All 221 existing tests must keep passing.** Run the full suite before every commit.
- **No test may require GCP credentials, network access, or the `google-cloud-*` libraries.** Those libraries are deliberately absent locally; every GCP import must be lazy and guarded.
- **Scope is `/backend` and `/tests` only** (plus the config keys in Task 7). No React, no dashboard — that is Plan C.
- Import only what you use.
- No "Phase 2", roadmap, rules-engine or case-management references in any code, comment, docstring, README or output string.

## Two Design Decisions Fixed Here

**1. Seeding moves out of `train.py` into `backend/seed.py`.** Spec §10 says *"`train.py` seeds approximately 200 scored historical transactions"*. Implementing that literally would make Plan A's training pipeline import Firestore, which would break its 221 tests on any machine without GCP libraries and couple offline training to a cloud service it has no other need for. A standalone `backend/seed.py` reads the artifact bundle, scores historical rows through the same `score_transaction()` path, and writes them to whichever `TransactionLog` is configured. Same outcome, no coupling. **Spec §10 should be amended to match.**

**2. `/score` and `/demo/inject` share one code path, and neither owns scoring.** Both build a full transaction dict and hand it to `score_transaction()`. `/demo/inject` differs only in filling absent fields from a named preset first. No endpoint may compute a feature, a vote, or an explanation itself.

## File Structure

| File | Responsibility |
|---|---|
| `backend/__init__.py` | Package marker |
| `backend/config.py` | Backend settings from env with documented defaults |
| `backend/storage.py` | `ArtifactSource` + `TransactionLog` protocols; `LocalArtifactSource`, `InMemoryTransactionLog` |
| `backend/gcp.py` | `GCSArtifactSource`, `FirestoreTransactionLog` — guarded imports, deployment only |
| `backend/presets.py` | The demo scenarios, as data |
| `backend/schemas.py` | Pydantic request/response models |
| `backend/deps.py` | Process-wide scorer and log wiring, resolved once at startup |
| `backend/routers/score.py` | `POST /score`, `POST /batch-score`, `GET /transactions/recent` |
| `backend/routers/demo.py` | `POST /demo/inject`, `GET /demo/presets` |
| `backend/main.py` | App factory, `lifespan`, CORS, `GET /health` |
| `backend/seed.py` | Populates the transaction log from historical rows |
| `backend/requirements.txt` | Runtime pins, including the GCP libraries |
| `backend/Dockerfile` | Container image |
| `backend/cloudbuild.yaml` | Build and deploy pipeline |
| `README.md` (root) | gcloud sequence from an empty project, plus cost estimate |

---

### Task 1: Backend scaffolding, config, health, and the app factory

**Files:**
- Create: `backend/__init__.py`, `backend/config.py`, `backend/main.py`, `backend/requirements.txt`
- Test: `tests/test_backend_health.py`

**Interfaces:**
- Produces: `backend.config.Settings` (frozen dataclass) with `artifact_dir: str`, `artifact_source: str` (`"local"`/`"gcs"`), `transaction_log: str` (`"memory"`/`"firestore"`), `gcs_bucket: str`, `gcs_prefix: str`, `firestore_collection: str`, `cors_origins: list[str]`, `recent_limit_default: int`, `recent_limit_max: int`; classmethod `Settings.from_env()`
- Produces: `backend.main.create_app() -> FastAPI`, and module-level `app`

- [ ] **Step 1: Write the failing test**

`tests/test_backend_health.py`:

```python
import os

from fastapi.testclient import TestClient

from backend.config import Settings
from backend.main import create_app


def test_settings_have_documented_defaults(monkeypatch):
    for key in list(os.environ):
        if key.startswith("ANOMALY_"):
            monkeypatch.delenv(key, raising=False)
    settings = Settings.from_env()
    assert settings.artifact_source == "local"
    assert settings.transaction_log == "memory"
    assert settings.artifact_dir == "artifacts"
    assert settings.recent_limit_default == 50
    assert settings.recent_limit_max == 500


def test_settings_read_the_environment(monkeypatch):
    monkeypatch.setenv("ANOMALY_ARTIFACT_SOURCE", "gcs")
    monkeypatch.setenv("ANOMALY_GCS_BUCKET", "some-bucket")
    monkeypatch.setenv("ANOMALY_CORS_ORIGINS", "http://a.test,http://b.test")
    settings = Settings.from_env()
    assert settings.artifact_source == "gcs"
    assert settings.gcs_bucket == "some-bucket"
    assert settings.cors_origins == ["http://a.test", "http://b.test"]


def test_health_reports_ok_without_loading_artifacts():
    """Cloud Run probes /health; it must answer even if the model is unavailable."""
    app = create_app(load_artifacts=False)
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False


def test_openapi_schema_builds():
    app = create_app(load_artifacts=False)
    with TestClient(app) as client:
        assert client.get("/openapi.json").status_code == 200
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `python -m pytest tests/test_backend_health.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.config'`

- [ ] **Step 3: Write `backend/config.py`**

```python
"""Backend settings, read from the environment with local-friendly defaults.

Defaults select the local artifact directory and the in-memory transaction log,
so the service and its tests run with no cloud dependencies. Deployment overrides
them with environment variables set on the Cloud Run service.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

PREFIX = "ANOMALY_"


def _env(name: str, default: str) -> str:
    return os.environ.get(PREFIX + name, default)


@dataclass(frozen=True)
class Settings:
    artifact_source: str
    artifact_dir: str
    gcs_bucket: str
    gcs_prefix: str
    transaction_log: str
    firestore_collection: str
    cors_origins: list[str]
    recent_limit_default: int
    recent_limit_max: int

    @classmethod
    def from_env(cls) -> "Settings":
        origins = _env("CORS_ORIGINS", "*")
        return cls(
            artifact_source=_env("ARTIFACT_SOURCE", "local"),
            artifact_dir=_env("ARTIFACT_DIR", "artifacts"),
            gcs_bucket=_env("GCS_BUCKET", ""),
            gcs_prefix=_env("GCS_PREFIX", "artifacts/latest"),
            transaction_log=_env("TRANSACTION_LOG", "memory"),
            firestore_collection=_env("FIRESTORE_COLLECTION", "scored_transactions"),
            cors_origins=[o.strip() for o in origins.split(",") if o.strip()],
            recent_limit_default=int(_env("RECENT_LIMIT_DEFAULT", "50")),
            recent_limit_max=int(_env("RECENT_LIMIT_MAX", "500")),
        )
```

- [ ] **Step 4: Write `backend/main.py`**

```python
"""FastAPI application for the anomaly scoring service.

The app owns no scoring logic. Every endpoint delegates to the ml package's
score_transaction(), so training and serving share one code path.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend import deps
from backend.config import Settings
from backend.routers import demo, score


def create_app(load_artifacts: bool = True) -> FastAPI:
    """Build the application.

    `load_artifacts=False` skips model loading, which lets the health probe and
    schema generation be tested without a trained bundle on disk.
    """
    settings = Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Artifacts load once here, never per request: unpickling the detectors
        # and initialising the SHAP explainer is the expensive part of a cold start.
        if load_artifacts:
            deps.startup(settings)
        yield
        deps.shutdown()

    app = FastAPI(
        title="Transaction Anomaly Detection",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_loaded": deps.is_loaded()}

    app.include_router(score.router)
    app.include_router(demo.router)
    return app


app = create_app()
```

- [ ] **Step 5: Write `backend/requirements.txt`**

Pins must match Plan A's exactly where they overlap — the detectors are joblib pickles and a scikit-learn drift breaks unpickling.

```
pandas==2.2.3
numpy==2.0.2
scikit-learn==1.6.0
xgboost==3.2.0
shap==0.51.0
joblib==1.4.2
PyYAML==6.0.2
fastapi==0.115.9
uvicorn[standard]==0.40.0
pydantic==2.12.5
google-cloud-storage==2.19.0
google-cloud-firestore==2.20.0
```

- [ ] **Step 6: Create `backend/__init__.py`** (empty) and the placeholder modules `backend/deps.py` and `backend/routers/__init__.py` so the imports in `main.py` resolve. Tasks 2-4 fill them; for now `deps.py` needs only `startup`, `shutdown`, `is_loaded` returning `False`, and each router needs an empty `APIRouter`.

- [ ] **Step 7: Run the test and the full suite**

Run: `python -m pytest tests/test_backend_health.py -v` then `python -m pytest -q`
Expected: 4 new pass; 221 existing still pass.

- [ ] **Step 8: Commit**

```bash
git add backend tests/test_backend_health.py
git commit -m "feat: scaffold FastAPI app with settings and health probe"
```

---

### Task 2: Storage interfaces and their local implementations

**Files:**
- Create: `backend/storage.py`
- Modify: `backend/deps.py`
- Test: `tests/test_backend_storage.py`

**Interfaces:**
- Produces: `ArtifactSource` protocol with `ensure_local() -> Path`; `TransactionLog` protocol with `append(result: dict) -> None`, `recent(limit: int) -> list[dict]`, `count() -> int`
- Produces: `LocalArtifactSource(directory)`, `InMemoryTransactionLog(maxlen)`
- Produces: `build_artifact_source(settings)`, `build_transaction_log(settings)` factories
- Produces in `deps.py`: `startup(settings)`, `shutdown()`, `is_loaded() -> bool`, `get_scorer() -> Scorer`, `get_log() -> TransactionLog`

- [ ] **Step 1: Write the failing test**

`tests/test_backend_storage.py`:

```python
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
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `python -m pytest tests/test_backend_storage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.storage'`

- [ ] **Step 3: Write `backend/storage.py`**

```python
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
```

- [ ] **Step 4: Write `backend/deps.py`**

```python
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
```

- [ ] **Step 5: Run the tests and the full suite**

Run: `python -m pytest tests/test_backend_storage.py -v` then `python -m pytest -q`
Expected: 7 new pass; everything else still passes.

- [ ] **Step 6: Commit**

```bash
git add backend/storage.py backend/deps.py tests/test_backend_storage.py
git commit -m "feat: add artifact source and transaction log seams"
```

---

### Task 3: Scoring endpoints

**Files:**
- Create: `backend/schemas.py`, `backend/routers/score.py`
- Test: `tests/test_backend_score.py`

**Interfaces:**
- Produces: `TransactionIn` (pydantic model of the 13 `RAW_INPUT_FIELDS` plus optional `TransactionID`), `BatchIn`
- Produces: `POST /score`, `POST /batch-score`, `GET /transactions/recent`

- [ ] **Step 1: Write the failing test**

`tests/test_backend_score.py`:

```python
import pytest
from fastapi.testclient import TestClient

from backend import deps
from backend.config import Settings
from backend.main import create_app
from backend.storage import InMemoryTransactionLog
from ml.config import Config
from ml.pipeline.score import Scorer
from ml.pipeline.train import run_training
from ml.storage.artifacts import load_bundle

NORMAL = {
    "TransactionID": "TX900001",
    "AccountID": "AC00128",
    "DeviceID": "D000380",
    "Location": "San Diego",
    "TransactionDate": "2023-12-01 10:15:00",
    "TransactionAmount": 120.00,
    "AccountBalance": 8000.00,
    "CustomerAge": 45,
    "TransactionDuration": 90,
    "LoginAttempts": 1,
    "TransactionType": "Debit",
    "Channel": "ATM",
    "CustomerOccupation": "Engineer",
}
WEIRD = {
    **NORMAL,
    "TransactionID": "TX900002",
    "TransactionAmount": 4800.00,
    "AccountBalance": 5000.00,
    "LoginAttempts": 5,
    "Channel": "Online",
}


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


def test_score_returns_the_result_contract(client):
    response = client.post("/score", json=NORMAL)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "transaction_id",
        "scored_at",
        "ensemble",
        "detectors",
        "explanation",
        "features",
        "raw",
        "warnings",
    }
    assert body["ensemble"]["votes_total"] == 4
    assert len(body["detectors"]) == 4


def test_score_flags_a_weird_transaction_and_clears_a_normal_one(client):
    assert client.post("/score", json=NORMAL).json()["ensemble"]["is_anomaly"] is False
    assert client.post("/score", json=WEIRD).json()["ensemble"]["is_anomaly"] is True


def test_score_persists_to_the_log(client):
    client.post("/score", json=NORMAL)
    recent = client.get("/transactions/recent").json()
    assert len(recent) == 1
    assert recent[0]["transaction_id"] == "TX900001"


def test_recent_is_newest_first_and_respects_limit(client):
    for i in range(3):
        client.post("/score", json={**NORMAL, "TransactionID": f"TX{i}"})
    recent = client.get("/transactions/recent?limit=2").json()
    assert [r["transaction_id"] for r in recent] == ["TX2", "TX1"]


def test_recent_rejects_a_limit_above_the_maximum(client):
    assert client.get("/transactions/recent?limit=99999").status_code == 422


def test_batch_score_returns_one_result_per_input(client):
    response = client.post("/batch-score", json={"transactions": [NORMAL, WEIRD]})
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    assert results[0]["ensemble"]["is_anomaly"] is False
    assert results[1]["ensemble"]["is_anomaly"] is True


def test_batch_score_rejects_an_empty_list(client):
    assert client.post("/batch-score", json={"transactions": []}).status_code == 422


def test_missing_required_field_is_a_422(client):
    payload = {k: v for k, v in NORMAL.items() if k != "AccountID"}
    assert client.post("/score", json=payload).status_code == 422


def test_zero_balance_is_a_422_not_a_500(client):
    """The ml layer raises ValueError; the API must translate it, not leak a 500."""
    response = client.post("/score", json={**NORMAL, "AccountBalance": 0})
    assert response.status_code == 422
    assert "AccountBalance" in response.json()["detail"]


def test_warnings_are_surfaced_in_the_response(client):
    response = client.post("/score", json={**NORMAL, "AccountID": "AC99999"})
    assert any("unseen account" in w for w in response.json()["warnings"])
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `python -m pytest tests/test_backend_score.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.schemas'`

- [ ] **Step 3: Write `backend/schemas.py`**

```python
"""Request models.

Field names mirror the raw CSV columns exactly, because ml.features.engineer's
transform_one() reads them by those names. Renaming here would mean translating
in two places.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TransactionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    TransactionID: str = Field(default="")
    AccountID: str
    DeviceID: str
    Location: str
    TransactionDate: str
    TransactionAmount: float
    AccountBalance: float
    CustomerAge: int
    TransactionDuration: int
    LoginAttempts: int
    TransactionType: str
    Channel: str
    CustomerOccupation: str


class BatchIn(BaseModel):
    transactions: list[TransactionIn] = Field(min_length=1, max_length=500)


class BatchOut(BaseModel):
    results: list[dict]
```

- [ ] **Step 4: Write `backend/routers/score.py`**

```python
"""Scoring endpoints.

None of these compute anything. Each builds a plain dict and hands it to the ml
package's score_transaction(), so the API cannot drift from the training path.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend import deps
from backend.config import Settings
from backend.schemas import BatchIn, BatchOut, TransactionIn

router = APIRouter(tags=["scoring"])


def _score_one(payload: TransactionIn) -> dict:
    try:
        result = deps.get_scorer().score_transaction(payload.model_dump())
    except ValueError as exc:
        # The ml layer rejects impossible inputs (a zero balance, a missing field).
        # That is a client error, not a server fault.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    deps.get_log().append(result)
    return result


@router.post("/score")
def score(payload: TransactionIn) -> dict:
    return _score_one(payload)


@router.post("/batch-score", response_model=BatchOut)
def batch_score(payload: BatchIn) -> BatchOut:
    return BatchOut(results=[_score_one(item) for item in payload.transactions])


@router.get("/transactions/recent")
def recent(limit: int | None = Query(default=None, ge=1)) -> list[dict]:
    settings = Settings.from_env()
    effective = limit if limit is not None else settings.recent_limit_default
    if effective > settings.recent_limit_max:
        raise HTTPException(
            status_code=422,
            detail=f"limit exceeds maximum of {settings.recent_limit_max}",
        )
    return deps.get_log().recent(effective)
```

- [ ] **Step 5: Run the tests and the full suite**

Run: `python -m pytest tests/test_backend_score.py -v` then `python -m pytest -q`
Expected: 10 new pass; everything else still passes.

If `test_score_flags_a_weird_transaction_and_clears_a_normal_one` fails, do not weaken it — that pairing is the demo's whole credibility. Check the fixture is function-scoped so profile mutation is not leaking between tests.

- [ ] **Step 6: Commit**

```bash
git add backend/schemas.py backend/routers/score.py tests/test_backend_score.py
git commit -m "feat: add scoring and recent-transaction endpoints"
```

---

### Task 4: Demo endpoints and presets

**Files:**
- Create: `backend/presets.py`, `backend/routers/demo.py`
- Test: `tests/test_backend_demo.py`

**Interfaces:**
- Produces: `PRESETS: dict[str, dict]` with keys `normal`, `account_drain`, `credential_stuffing`, `rapid_fire`; `preset_names() -> list[str]`
- Produces: `POST /demo/inject`, `GET /demo/presets`

- [ ] **Step 1: Write the failing test**

`tests/test_backend_demo.py`:

```python
import pytest
from fastapi.testclient import TestClient

from backend import deps
from backend.main import create_app
from backend.presets import PRESETS
from backend.storage import InMemoryTransactionLog
from ml.config import Config
from ml.pipeline.score import Scorer
from ml.pipeline.train import run_training
from ml.storage.artifacts import load_bundle


@pytest.fixture(scope="module")
def artifact_dir(tmp_path_factory):
    dest = tmp_path_factory.mktemp("artifacts")
    run_training(Config.load(), dest=dest)
    return dest


@pytest.fixture
def client(artifact_dir):
    app = create_app(load_artifacts=False)
    threshold = Config.load().get("ensemble.threshold")
    deps.override(Scorer(load_bundle(artifact_dir), threshold), InMemoryTransactionLog())
    with TestClient(app) as test_client:
        yield test_client
    deps.shutdown()


def test_presets_endpoint_lists_every_scenario(client):
    body = client.get("/demo/presets").json()
    assert {p["name"] for p in body["presets"]} == set(PRESETS)
    for preset in body["presets"]:
        assert preset["label"]
        assert preset["description"]
        assert isinstance(preset["fields"], dict)


def test_normal_preset_returns_clean(client):
    """A demo where every input is flagged proves nothing."""
    body = client.post("/demo/inject", json={"preset": "normal"}).json()
    assert body["ensemble"]["is_anomaly"] is False


@pytest.mark.parametrize("name", ["account_drain", "credential_stuffing"])
def test_anomalous_presets_are_flagged(client, name):
    body = client.post("/demo/inject", json={"preset": name}).json()
    assert body["ensemble"]["is_anomaly"] is True
    assert body["explanation"]["plain_english"].startswith("Flagged primarily due to")


def test_overrides_take_precedence_over_the_preset(client):
    body = client.post(
        "/demo/inject",
        json={"preset": "normal", "overrides": {"LoginAttempts": 5, "TransactionAmount": 4900.0}},
    ).json()
    assert body["raw"]["LoginAttempts"] == 5
    assert body["raw"]["TransactionAmount"] == 4900.0


def test_inject_persists_to_the_log(client):
    client.post("/demo/inject", json={"preset": "account_drain"})
    assert len(client.get("/transactions/recent").json()) == 1


def test_unknown_preset_is_a_422(client):
    response = client.post("/demo/inject", json={"preset": "nope"})
    assert response.status_code == 422
    assert "nope" in response.json()["detail"]


def test_inject_generates_a_transaction_id_when_absent(client):
    body = client.post("/demo/inject", json={"preset": "normal"}).json()
    assert body["transaction_id"]


def test_rapid_fire_preset_increments_the_daily_count(client):
    first = client.post("/demo/inject", json={"preset": "rapid_fire"}).json()
    second = client.post("/demo/inject", json={"preset": "rapid_fire"}).json()
    assert second["features"]["DailyAccountVolume"] > first["features"]["DailyAccountVolume"]
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `python -m pytest tests/test_backend_demo.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.presets'`

- [ ] **Step 3: Write `backend/presets.py`**

Every preset must use a `TransactionDate` **after** the training era ends (the data runs to 2024-01-01) and an account present in training, so the gap feature is meaningful rather than clamped. `_recent_timestamp()` returns a fixed in-era date rather than `now()` so preset behaviour is reproducible.

```python
"""Demo scenarios, served from the backend so they can be tuned without a rebuild.

The 'normal' preset is not optional. A demonstration where every input is flagged
proves nothing; the clean result is what makes the flagged ones credible.
"""
from __future__ import annotations

BASE = {
    "AccountID": "AC00128",
    "DeviceID": "D000380",
    "Location": "San Diego",
    "TransactionDate": "2023-12-01 10:15:00",
    "TransactionAmount": 120.00,
    "AccountBalance": 8000.00,
    "CustomerAge": 45,
    "TransactionDuration": 90,
    "LoginAttempts": 1,
    "TransactionType": "Debit",
    "Channel": "ATM",
    "CustomerOccupation": "Engineer",
}

PRESETS: dict[str, dict] = {
    "normal": {
        "label": "Routine transaction",
        "description": "An ordinary debit well within the account's normal behaviour.",
        "fields": dict(BASE),
    },
    "account_drain": {
        "label": "Account drain",
        "description": "96% of the available balance moved in a single transaction.",
        "fields": {**BASE, "TransactionAmount": 4800.00, "AccountBalance": 5000.00},
    },
    "credential_stuffing": {
        "label": "Credential stuffing",
        "description": "Five login attempts before an online transfer.",
        "fields": {**BASE, "LoginAttempts": 5, "Channel": "Online", "TransactionAmount": 900.00},
    },
    "rapid_fire": {
        "label": "Rapid-fire activity",
        "description": "Repeated transactions on one account the same day. Inject twice to see the daily count climb.",
        "fields": {**BASE, "AccountID": "AC00455", "TransactionAmount": 300.00},
    },
}


def preset_names() -> list[str]:
    return list(PRESETS)
```

- [ ] **Step 4: Write `backend/routers/demo.py`**

```python
"""Demo endpoints.

/demo/inject is a convenience wrapper, not a second scoring path: it fills absent
fields from a named preset and then delegates to exactly the same code as /score.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from backend import deps
from backend.presets import PRESETS
from backend.schemas import TransactionIn

router = APIRouter(prefix="/demo", tags=["demo"])


class InjectIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preset: str = "normal"
    overrides: dict = Field(default_factory=dict)


@router.get("/presets")
def presets() -> dict:
    return {
        "presets": [
            {
                "name": name,
                "label": preset["label"],
                "description": preset["description"],
                "fields": preset["fields"],
            }
            for name, preset in PRESETS.items()
        ]
    }


@router.post("/inject")
def inject(payload: InjectIn) -> dict:
    preset = PRESETS.get(payload.preset)
    if preset is None:
        raise HTTPException(
            status_code=422,
            detail=f"unknown preset {payload.preset!r}; choose one of {list(PRESETS)}",
        )

    fields = {**preset["fields"], **payload.overrides}
    fields.setdefault("TransactionID", f"DEMO-{uuid.uuid4().hex[:8].upper()}")

    try:
        transaction = TransactionIn(**fields)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        result = deps.get_scorer().score_transaction(transaction.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    deps.get_log().append(result)
    return result
```

- [ ] **Step 5: Run the tests and the full suite**

Run: `python -m pytest tests/test_backend_demo.py -v` then `python -m pytest -q`
Expected: 9 new pass (one is parametrised ×2); everything else still passes.

The preset assertions are behavioural, not cosmetic. If `normal` flags or an anomalous preset comes back clean, that is a real finding — report it rather than retuning the preset to make the test green.

- [ ] **Step 6: Commit**

```bash
git add backend/presets.py backend/routers/demo.py tests/test_backend_demo.py
git commit -m "feat: add demo preset and injection endpoints"
```

---

### Task 5: GCP implementations

**Files:**
- Create: `backend/gcp.py`
- Test: `tests/test_backend_gcp.py`

**Interfaces:**
- Produces: `GCSArtifactSource(bucket, prefix, cache_dir=None)` implementing `ensure_local()`; `FirestoreTransactionLog(collection, client=None)` implementing `append`/`recent`/`count`

The `google-cloud-*` libraries are **not installed locally and must not become a test dependency.** Both classes import them inside `__init__`, and both accept an injected client so the tests exercise the real logic against fakes.

- [ ] **Step 1: Write the failing test**

`tests/test_backend_gcp.py`:

```python
import pytest

from backend.gcp import FirestoreTransactionLog, GCSArtifactSource


class FakeBlob:
    def __init__(self, name, payload=b"x"):
        self.name = name
        self._payload = payload
        self.downloaded_to = None

    def download_to_filename(self, path):
        self.downloaded_to = path
        with open(path, "wb") as fh:
            fh.write(self._payload)


class FakeBucket:
    def __init__(self, blobs):
        self._blobs = blobs

    def list_blobs(self, prefix=None):
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
    blobs = [FakeBlob("artifacts/latest/manifest.json")]
    source = GCSArtifactSource(
        "bucket", "artifacts/latest", cache_dir=tmp_path, client=FakeGCSClient(FakeBucket(blobs))
    )
    assert source.ensure_local() == source.ensure_local()


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
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `python -m pytest tests/test_backend_gcp.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.gcp'`

- [ ] **Step 3: Write `backend/gcp.py`**

```python
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
        blobs = list(bucket.list_blobs(prefix=self.prefix))
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
```

- [ ] **Step 4: Run the tests and the full suite**

Run: `python -m pytest tests/test_backend_gcp.py -v` then `python -m pytest -q`
Expected: 6 new pass; everything else still passes.

- [ ] **Step 5: Verify the libraries really are not needed**

Run: `python -c "import backend.gcp; print('imported without google-cloud installed')"`
Expected: prints the message. If it raises `ModuleNotFoundError`, an import escaped to module scope.

- [ ] **Step 6: Commit**

```bash
git add backend/gcp.py tests/test_backend_gcp.py
git commit -m "feat: add Cloud Storage and Firestore storage implementations"
```

---

### Task 6: Seed script and container build

**Files:**
- Create: `backend/seed.py`, `backend/Dockerfile`, `backend/.dockerignore`, `backend/cloudbuild.yaml`
- Test: `tests/test_backend_seed.py`

**Interfaces:**
- Produces: `seed_log(scorer, log, raw_df, limit) -> int`, and a `python -m backend.seed` CLI

- [ ] **Step 1: Write the failing test**

`tests/test_backend_seed.py`:

```python
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
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `python -m pytest tests/test_backend_seed.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.seed'`

- [ ] **Step 3: Write `backend/seed.py`**

Seeding lives here rather than in `ml/pipeline/train.py` so that offline training never imports Firestore — see Design Decision 1.

```python
"""Populates the transaction log so the dashboard feed is not empty on first load.

Deliberately separate from ml/pipeline/train.py: putting this in the training
pipeline would make offline training depend on Firestore, and would break the ml
package's tests on any machine without the google-cloud libraries.
"""
from __future__ import annotations

import argparse

import pandas as pd

from backend.config import Settings
from backend.storage import TransactionLog, build_artifact_source, build_transaction_log
from ml.config import Config
from ml.data.loader import load_raw
from ml.features.engineer import RAW_INPUT_FIELDS
from ml.pipeline.score import Scorer
from ml.storage.artifacts import load_bundle

DEFAULT_LIMIT = 200


def seed_log(
    scorer: Scorer,
    log: TransactionLog,
    raw_df: pd.DataFrame,
    limit: int = DEFAULT_LIMIT,
) -> int:
    """Score the most recent `limit` historical rows into `log`. Returns the count."""
    ordered = raw_df.sort_values("TransactionDate").tail(limit)
    written = 0
    for row in ordered.to_dict(orient="records"):
        payload = {field: row[field] for field in RAW_INPUT_FIELDS}
        payload["TransactionID"] = str(row["TransactionID"])
        payload["TransactionDate"] = str(row["TransactionDate"])
        log.append(scorer.score_transaction(payload))
        written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the scored-transaction log.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    args = parser.parse_args()

    settings = Settings.from_env()
    directory = build_artifact_source(settings).ensure_local()
    scorer = Scorer(load_bundle(directory), Config.load().get("ensemble.threshold"))
    log = build_transaction_log(settings)
    raw = load_raw(Config.load().get("data.csv_path"))

    written = seed_log(scorer, log, raw, limit=args.limit)
    print(f"seeded {written} transactions into {settings.transaction_log}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write `backend/Dockerfile`**

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Requirements first so the dependency layer caches across code changes.
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY ml ./ml
COPY backend ./backend
COPY original.csv ./original.csv

# Cloud Run injects PORT; uvicorn must bind it rather than a fixed port.
CMD exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT} --workers 1
```

- [ ] **Step 5: Write `backend/.dockerignore`**

```
__pycache__/
*.py[cod]
.pytest_cache/
tests/
docs/
artifacts/
final.ipynb
.git/
.superpowers/
```

- [ ] **Step 6: Write `backend/cloudbuild.yaml`**

```yaml
substitutions:
  _REGION: us-central1
  _REPO: anomaly
  _SERVICE: anomaly-api
  _BUCKET: ""

steps:
  - name: gcr.io/cloud-builders/docker
    args:
      - build
      - -f
      - backend/Dockerfile
      - -t
      - ${_REGION}-docker.pkg.dev/${PROJECT_ID}/${_REPO}/${_SERVICE}:${SHORT_SHA}
      - .

  - name: gcr.io/cloud-builders/docker
    args:
      - push
      - ${_REGION}-docker.pkg.dev/${PROJECT_ID}/${_REPO}/${_SERVICE}:${SHORT_SHA}

  - name: gcr.io/google.com/cloudsdktool/cloud-sdk
    entrypoint: gcloud
    args:
      - run
      - deploy
      - ${_SERVICE}
      - --image=${_REGION}-docker.pkg.dev/${PROJECT_ID}/${_REPO}/${_SERVICE}:${SHORT_SHA}
      - --region=${_REGION}
      - --platform=managed
      - --allow-unauthenticated
      - --memory=2Gi
      - --cpu=1
      - --min-instances=1
      - --max-instances=1
      - --set-env-vars=ANOMALY_ARTIFACT_SOURCE=gcs,ANOMALY_GCS_BUCKET=${_BUCKET},ANOMALY_TRANSACTION_LOG=firestore

options:
  logging: CLOUD_LOGGING_ONLY
```

- [ ] **Step 7: Run the tests and the full suite**

Run: `python -m pytest tests/test_backend_seed.py -v` then `python -m pytest -q`
Expected: 3 new pass; everything else still passes.

- [ ] **Step 8: Commit**

```bash
git add backend/seed.py backend/Dockerfile backend/.dockerignore backend/cloudbuild.yaml tests/test_backend_seed.py
git commit -m "feat: add log seeding and container build configuration"
```

---

### Task 7: README, config keys, and acceptance

**Files:**
- Create: `README.md` (repo root)
- Modify: `ml/config.yaml` (add a `gcp:` block)
- Test: `tests/test_backend_acceptance.py`

**Interfaces:**
- Consumes: everything
- Produces: nothing new

- [ ] **Step 1: Add the `gcp` block to `ml/config.yaml`**

Append (do not disturb existing keys):

```yaml
gcp:
  project_id: ""
  region: us-central1
  bucket: ""
  artifact_prefix: artifacts/latest
  firestore_collection: scored_transactions
  artifact_repo: anomaly
  service_name: anomaly-api
```

Leave `project_id` and `bucket` empty — README step 1 sets them, and no code reads them at import time.

- [ ] **Step 2: Write the acceptance test**

`tests/test_backend_acceptance.py`:

```python
"""Plan B acceptance: the API surface from spec section 10, end to end."""
import pytest
from fastapi.testclient import TestClient

from backend import deps
from backend.main import create_app
from backend.storage import InMemoryTransactionLog
from ml.config import Config
from ml.pipeline.score import Scorer
from ml.pipeline.train import run_training
from ml.storage.artifacts import load_bundle

EXPECTED_ROUTES = {
    ("POST", "/score"),
    ("POST", "/batch-score"),
    ("GET", "/transactions/recent"),
    ("POST", "/demo/inject"),
    ("GET", "/demo/presets"),
    ("GET", "/health"),
}


@pytest.fixture(scope="module")
def artifact_dir(tmp_path_factory):
    dest = tmp_path_factory.mktemp("artifacts")
    run_training(Config.load(), dest=dest)
    return dest


@pytest.fixture
def client(artifact_dir):
    app = create_app(load_artifacts=False)
    deps.override(
        Scorer(load_bundle(artifact_dir), Config.load().get("ensemble.threshold")),
        InMemoryTransactionLog(),
    )
    with TestClient(app) as test_client:
        yield test_client
    deps.shutdown()


def test_every_spec_route_is_present(client):
    app = client.app
    actual = {
        (method, route.path)
        for route in app.routes
        for method in getattr(route, "methods", set())
        if method in {"GET", "POST"}
    }
    assert EXPECTED_ROUTES <= actual


def test_the_pitch_flow_works_end_to_end(client):
    """Clean preset reads clean, drain preset flags, both land in the feed."""
    clean = client.post("/demo/inject", json={"preset": "normal"}).json()
    drain = client.post("/demo/inject", json={"preset": "account_drain"}).json()

    assert clean["ensemble"]["is_anomaly"] is False
    assert drain["ensemble"]["is_anomaly"] is True
    assert drain["ensemble"]["votes_for"] >= drain["ensemble"]["votes_required"]
    assert len(drain["detectors"]) == 4
    assert drain["explanation"]["top_features"]
    print("\nflagged:", drain["explanation"]["plain_english"])

    feed = client.get("/transactions/recent").json()
    assert [r["transaction_id"] for r in feed] == [
        drain["transaction_id"],
        clean["transaction_id"],
    ]


def test_no_endpoint_reimplements_scoring(client):
    """/score and /demo/inject must agree given identical input."""
    from backend.presets import PRESETS

    fields = {**PRESETS["account_drain"]["fields"], "TransactionID": "CMP-1"}
    direct = client.post("/score", json=fields).json()

    deps.get_log()  # same process, same scorer
    injected = client.post(
        "/demo/inject",
        json={"preset": "account_drain", "overrides": {"TransactionID": "CMP-2"}},
    ).json()

    assert direct["ensemble"]["votes_for"] == injected["ensemble"]["votes_for"]
    assert [d["name"] for d in direct["detectors"]] == [
        d["name"] for d in injected["detectors"]
    ]


def test_health_is_available_without_a_model(artifact_dir):
    app = create_app(load_artifacts=False)
    with TestClient(app) as bare:
        assert bare.get("/health").json()["status"] == "ok"
```

- [ ] **Step 3: Run it and confirm it fails**

Run: `python -m pytest tests/test_backend_acceptance.py -v`
Expected: FAIL until the `gcp` block and all prior tasks are in place.

- [ ] **Step 4: Write `README.md` at the repo root**

It must contain, in order: what the project is; local quickstart (train, run, score); the complete gcloud sequence from an empty project; the min-instances toggle; and the cost estimate. The gcloud sequence:

```bash
# 1. Set these once
export PROJECT_ID="your-project-id"
export REGION="us-central1"
export BUCKET="gs://${PROJECT_ID}-anomaly-artifacts"
export REPO="anomaly"
export SERVICE="anomaly-api"

gcloud config set project "${PROJECT_ID}"

# 2. Enable the APIs
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com \
  firestore.googleapis.com

# 3. Create the artifact bucket
gcloud storage buckets create "${BUCKET}" --location="${REGION}"

# 4. Create the container repository
gcloud artifacts repositories create "${REPO}" \
  --repository-format=docker --location="${REGION}"

# 5. Provision Firestore in Native mode
gcloud firestore databases create --location="${REGION}"

# 6. Train locally and upload the bundle
python -m ml.pipeline.train
gcloud storage cp -r artifacts/* "${BUCKET}/artifacts/latest/"

# 7. Build and deploy
gcloud builds submit --config backend/cloudbuild.yaml \
  --substitutions=_REGION="${REGION}",_REPO="${REPO}",_SERVICE="${SERVICE}",_BUCKET="${BUCKET#gs://}"

# 8. Seed the feed so the dashboard is not empty
SERVICE_URL=$(gcloud run services describe "${SERVICE}" --region "${REGION}" --format='value(status.url)')
curl -s -X POST "${SERVICE_URL}/demo/inject" -H 'Content-Type: application/json' -d '{"preset":"normal"}'
```

Include the min-instances toggle:

```bash
# Before a demonstration — removes the 8-20s cold start
gcloud run services update "${SERVICE}" --region "${REGION}" --min-instances=1
# Afterwards — back to scale-to-zero
gcloud run services update "${SERVICE}" --region "${REGION}" --min-instances=0
```

And the cost table, stated as approximate with a pointer to current pricing:

| Item | Weekly, warm | Weekly, scale-to-zero |
|---|---|---|
| Cloud Run idle instance (1 vCPU, 2 GiB) | ~$1.50–2.00 | $0 |
| Cloud Run requests (demo volume) | <$0.05 | <$0.05 |
| Firestore reads/writes (dashboard polling) | <$0.10 | <$0.10 |
| Cloud Storage (bundle ~7MB) | <$0.01 | <$0.01 |
| Artifact Registry (image ~1.5GB) | ~$0.04 | ~$0.04 |
| **Total** | **~$2–3** | **<$0.25** |

Document `--max-instances=1` and why: the profile store is mutated in process, so a second instance would develop a divergent view of recent activity.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q`
Expected: everything passes — 221 from Plan A plus roughly 43 new.

- [ ] **Step 6: Verify the notebook is untouched**

Run: `git status --porcelain final.ipynb`
Expected: empty.

- [ ] **Step 7: Commit**

```bash
git add README.md ml/config.yaml tests/test_backend_acceptance.py
git commit -m "docs: add deployment README and Plan B acceptance suite"
```

---

## Acceptance Criteria

1. `python -m pytest` passes — Plan A's 221 plus the new backend tests.
2. No test imports `google.cloud`, and `python -c "import backend.gcp"` succeeds with those libraries absent.
3. All six spec §10 routes exist and respond.
4. The `normal` preset returns clean; `account_drain` and `credential_stuffing` are flagged.
5. `/score` and `/demo/inject` produce identical verdicts for identical input.
6. `README.md` contains the full gcloud sequence, the min-instances toggle and the cost estimate.
7. `git status --porcelain final.ipynb` is empty.
8. `/ml` is unchanged except for the `gcp:` block in `config.yaml`.
