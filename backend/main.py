"""FastAPI application for the anomaly scoring service.

The app owns no scoring logic. Every endpoint delegates to the ml package's
score_transaction(), so training and serving share one code path.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend import deps
from backend.config import Settings
from backend.routers import demo, score

# The built dashboard. Resolved from this file rather than the working
# directory, so it holds wherever the process is started from.
DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard" / "dist"


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

    # Mounted after the routers, and only when the bundle has been built.
    # Starlette matches routes in registration order, so every API route and
    # /docs registered above wins and only unmatched paths reach the dashboard
    # -- mounting first would shadow the entire API. The is_dir() guard keeps
    # local runs and the test suite working against a tree with no dist/.
    # Serving the dashboard from the same origin as the API is why it needs
    # neither a base URL nor CORS; see dashboard/src/api.js.
    if DASHBOARD_DIR.is_dir():
        app.mount("/", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")

    return app


app = create_app()
