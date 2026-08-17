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
