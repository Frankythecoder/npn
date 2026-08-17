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
