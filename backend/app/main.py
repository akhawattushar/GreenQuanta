"""Application factory and router wiring."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api import deps
from app.api.routes import admin, auth, health, optimization, prediction, report, scenario, voyage
from app.core.config import get_settings
from app.db.database import Database, DatabaseError
from app.services.model_registry import ModelUnavailableError, get_bundle

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("quantafleet")

DESCRIPTION = """
Backend for **GreenQuanta / QuantaFleet** — AI-assisted fuel prediction and fleet optimisation.

Fuel predictions come from the trained artifacts shipped in `artifacts/`. If those
artifacts cannot be loaded the prediction, optimisation and scenario endpoints
return **503** with the reason — they never substitute a placeholder number.

Cost and greenhouse-gas figures are derived from the model's fuel output using
configured conversion factors. Those factors are assumptions, not learned values,
and every response carries them in an `assumptions` block.

Voyage monitoring uses voyages you record yourself: no AIS or telemetry feed is
connected, so progress and fuel burn are computed from the trained model and the
voyage's departure time rather than a live position feed.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    try:
        database = Database()
    except DatabaseError as exc:
        logger.error("MongoDB unavailable at startup: %s", exc)
        raise
    deps.init_database(database)

    try:
        bundle = get_bundle()
        logger.info(
            "Model ready: %s + %s, %d transformed features.",
            bundle.model_class,
            bundle.preprocessor_class,
            bundle.n_transformed_features,
        )
        for warning in bundle.warnings:
            logger.warning("Artifact warning: %s", warning)
    except ModelUnavailableError as exc:
        # Start anyway so /health can report the problem; ML endpoints will 503.
        logger.error("Model unavailable at startup: %s", exc)

    if settings.secret_key == "dev-insecure-change-me" and settings.app_env != "development":
        logger.error("SECRET_KEY is still the development default outside development.")

    yield
    database.close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(ModelUnavailableError)
    async def _model_unavailable(_: Request, exc: ModelUnavailableError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": f"The trained model is unavailable. {exc}"},
        )

    @app.exception_handler(ValueError)
    async def _value_error(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"detail": str(exc)}
        )

    prefix = settings.api_prefix
    app.include_router(health.router, prefix=prefix)
    app.include_router(auth.router, prefix=prefix)
    app.include_router(prediction.router, prefix=prefix)
    app.include_router(optimization.router, prefix=prefix)
    app.include_router(scenario.router, prefix=prefix)
    app.include_router(voyage.router, prefix=prefix)
    app.include_router(report.router, prefix=prefix)
    app.include_router(admin.router, prefix=prefix)

    @app.get("/", tags=["health"], summary="Service banner")
    def root() -> dict:
        return {
            "service": settings.app_name,
            "version": __version__,
            "docs": "/docs",
            "api_prefix": prefix,
        }

    return app


app = create_app()
