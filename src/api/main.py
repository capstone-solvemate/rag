from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src.api.routes.chat import router as chat_router
from src.api.routes.detection import router as detection_router
from src.api.routes.image_chat import router as image_chat_router
from src.api.routes.knowledge_base import router as kb_router
from src.api.routes.health import health_check
from src.api.schemas.common import ErrorResponse
from src.api.schemas.health import HealthResponse
from src.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — startup and shutdown events
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup validation before the app begins accepting requests.

    config.validate() raises immediately if OPENAI_API_KEY is missing,
    preventing the service from starting in a broken state.
    """
    logger.info(f"Starting enterprise-rag-chatbot | env={config.APP_ENV}")
    config.validate()
    logger.info("Config validated. Application ready.")
    yield
    logger.info("Application shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Enterprise RAG Chatbot",
    description="Retrieval-augmented generation API for enterprise documents.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description=(
        "Returns service status, Chroma document count, "
        "and OpenAI reachability. Status is 'ok' only when "
        "all dependencies are healthy and documents are indexed."
    ),
)
async def get_health() -> HealthResponse:
    return await health_check()

app.include_router(chat_router)
app.include_router(detection_router)
app.include_router(image_chat_router)
app.include_router(kb_router)


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Normalize HTTPException responses to ErrorResponse shape.

    Routes raise HTTPException with detail as either:
    - str  → wrapped as { detail: str, error_code: null }
    - dict → must contain 'detail' and 'error_code' keys

    This keeps all error responses consistent for the Express consumer.
    """
    if isinstance(exc.detail, dict):
        content = ErrorResponse(
            detail=exc.detail.get("detail", str(exc.detail)),
            error_code=exc.detail.get("error_code"),
        ).model_dump()
    else:
        content = ErrorResponse(
            detail=str(exc.detail),
            error_code=None,
        ).model_dump()

    return JSONResponse(status_code=exc.status_code, content=content)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {type(exc).__name__}: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            detail="An unexpected internal error occurred.",
            error_code="INTERNAL_ERROR",
        ).model_dump(),
    )