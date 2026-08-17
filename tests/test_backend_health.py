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
