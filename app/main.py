"""FastAPI application entrypoint with 7-layer zero-leak exception and route guards."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.database import init_db
from app.routers import admin, proxy
from app.security import redact_secrets
from app.upstream import build_openai_error, close_shared_client


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize SQLite WAL schema on startup and close shared httpx client on shutdown."""
    init_db()
    try:
        yield
    finally:
        await close_shared_client()


app = FastAPI(
    title="Model-Gate Zero-Leak Gemini Proxy",
    description="OpenAI-compatible proxy that isolates the upstream Gemini API key and enforces per-consumer virtual key policies.",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Api-Key", "X-Admin-Key"],
)


@app.exception_handler(HTTPException)
async def sanitized_http_exception_handler(
    _request: Request, exc: HTTPException
) -> JSONResponse:
    """Ensure HTTPExceptions never leak secrets and return OpenAI-compatible JSON."""
    settings = get_settings()
    detail_str = redact_secrets(str(exc.detail), settings)
    return JSONResponse(
        status_code=exc.status_code,
        content=build_openai_error(
            message=detail_str,
            error_type="http_error",
            code=f"http_{exc.status_code}",
            settings=settings,
        ),
        headers={"Cache-Control": "no-store"},
    )


@app.exception_handler(RequestValidationError)
async def sanitized_validation_exception_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Scrub request validation errors and format as OpenAI 400 invalid_request_error."""
    settings = get_settings()
    clean_msg = redact_secrets(str(exc.errors()), settings)
    return JSONResponse(
        status_code=400,
        content=build_openai_error(
            message=f"Invalid request payload: {clean_msg}",
            error_type="invalid_request_error",
            code="validation_error",
            settings=settings,
        ),
        headers={"Cache-Control": "no-store"},
    )


@app.exception_handler(Exception)
async def global_zero_leak_exception_handler(
    _request: Request, exc: Exception
) -> JSONResponse:
    """L7 Global Exception Handler: scrub any exception and never expose stack traces."""
    settings = get_settings()
    _ = redact_secrets(str(exc), settings)
    return JSONResponse(
        status_code=500,
        content=build_openai_error(
            message="An internal proxy error occurred.",
            error_type="server_error",
            code="internal_proxy_error",
            settings=settings,
        ),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Public health check endpoint (never exposes secrets)."""
    settings = get_settings()
    return {
        "status": "ok",
        "service": "model-gate",
        "version": "0.1.0",
        "upstream_configured": bool(settings.gemini_api_key),
    }


# L5 Explicit Route Surface Only: /v1/models, /v1/chat/completions, /admin/*
app.include_router(proxy.router)
app.include_router(admin.router)
