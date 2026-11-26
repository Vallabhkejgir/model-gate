"""Protected /admin/api/* control plane endpoints and /admin web dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from app.config import get_settings
from app.database import (
    create_virtual_key_record,
    get_usage_summary,
    get_virtual_key_by_id,
    list_virtual_keys,
    revoke_virtual_key,
    update_virtual_key_policy,
)
from app.models import CreateVirtualKeyRequest, UpdateVirtualKeyRequest
from app.rate_limit import rate_limiter
from app.security import (
    create_admin_session_token,
    verify_admin_key,
    verify_admin_session_token,
)

router = APIRouter(prefix="/admin", tags=["Admin Control Plane"])

_STATIC_ADMIN_HTML = Path(__file__).resolve().parent.parent / "static" / "admin.html"
ADMIN_SESSION_COOKIE = "mg_admin_session"


def require_admin_auth(request: Request) -> None:
    """Authenticate operator requests using X-Admin-Key, Authorization: Bearer, or mg_admin_session cookie."""
    header_key = request.headers.get("X-Admin-Key", "").strip()
    if header_key:
        if verify_admin_key(header_key):
            return
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing operator admin key.",
        )

    auth_header = request.headers.get("Authorization", "").strip()
    if auth_header.lower().startswith("bearer "):
        bearer_key = auth_header[7:].strip()
        if bearer_key:
            if verify_admin_key(bearer_key):
                return
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing operator admin key.",
            )

    cookie_token = request.cookies.get(ADMIN_SESSION_COOKIE, "").strip()
    if cookie_token and verify_admin_session_token(cookie_token):
        return

    raise HTTPException(
        status_code=401,
        detail="Invalid or missing operator admin key.",
    )


@router.post("/api/session")
async def create_admin_session(request: Request) -> JSONResponse:
    """Validate operator admin key and set an HttpOnly signed session cookie."""
    candidate = request.headers.get("X-Admin-Key", "").strip()
    if not candidate:
        auth_header = request.headers.get("Authorization", "").strip()
        if auth_header.lower().startswith("bearer "):
            candidate = auth_header[7:].strip()

    if not candidate:
        try:
            body = await request.json()
            if isinstance(body, dict) and isinstance(body.get("admin_key"), str):
                candidate = body["admin_key"].strip()
        except Exception:
            pass

    if not candidate or not verify_admin_key(candidate):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing operator admin key.",
        )

    token = create_admin_session_token()
    resp = JSONResponse(status_code=200, content={"authenticated": True})
    resp.set_cookie(
        key=ADMIN_SESSION_COOKIE,
        value=token,
        max_age=86400 * 7,
        httponly=True,
        samesite="lax",
        path="/admin",
    )
    return resp


@router.post("/api/logout")
async def logout_admin_session() -> JSONResponse:
    """Clear the operator admin session cookie."""
    resp = JSONResponse(status_code=200, content={"authenticated": False})
    resp.delete_cookie(key=ADMIN_SESSION_COOKIE, path="/admin")
    return resp


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def serve_admin_dashboard() -> HTMLResponse:
    """Serve the built-in operator web dashboard."""
    if not _STATIC_ADMIN_HTML.exists():
        return HTMLResponse(
            content="<h1>Admin UI file missing</h1>", status_code=404
        )
    return HTMLResponse(
        content=_STATIC_ADMIN_HTML.read_text(encoding="utf-8"),
        status_code=200,
    )


@router.post("/api/keys", dependencies=[Depends(require_admin_auth)])
async def create_consumer_key(payload: CreateVirtualKeyRequest) -> JSONResponse:
    """Create a new virtual consumer API key (returns plaintext key once)."""
    settings = get_settings()
    allowed = payload.allowed_models or [settings.default_limited_model]
    record = create_virtual_key_record(
        name=payload.name.strip(),
        allowed_models=allowed,
        forced_system_prompt=payload.forced_system_prompt,
        max_tokens_per_request=payload.max_tokens_per_request,
        max_input_chars=payload.max_input_chars,
        rpm_limit=payload.rpm_limit,
        token_budget=payload.token_budget,
        expires_at=payload.expires_at,
    )
    return JSONResponse(status_code=201, content=record)


@router.get("/api/keys", dependencies=[Depends(require_admin_auth)])
async def list_consumer_keys() -> Dict[str, Any]:
    """List all consumer virtual keys and their usage counters (never exposes hashes/secrets)."""
    keys = list_virtual_keys()
    return {"object": "list", "data": keys}


@router.get("/api/keys/{key_id}", dependencies=[Depends(require_admin_auth)])
async def get_consumer_key(key_id: str) -> Dict[str, Any]:
    """Inspect a single virtual consumer key by key_id."""
    record = get_virtual_key_by_id(key_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Virtual key '{key_id}' not found.")
    return record


@router.patch("/api/keys/{key_id}", dependencies=[Depends(require_admin_auth)])
async def update_consumer_key(
    key_id: str, payload: UpdateVirtualKeyRequest
) -> Dict[str, Any]:
    """Update policy limits, allowed models, active status, or reset usage on a virtual key."""
    updates = payload.model_dump(exclude_unset=True)
    updated = update_virtual_key_policy(key_id=key_id, updates=updates)
    if not updated:
        raise HTTPException(status_code=404, detail=f"Virtual key '{key_id}' not found.")
    if updates.get("reset_usage"):
        rate_limiter.reset(key_id)
    return updated


@router.delete("/api/keys/{key_id}", dependencies=[Depends(require_admin_auth)])
async def delete_or_revoke_consumer_key(
    key_id: str,
    hard: bool = Query(default=False, description="Permanently delete instead of revoking"),
) -> Dict[str, Any]:
    """Revoke (deactivate) or permanently delete a consumer virtual key."""
    ok = revoke_virtual_key(key_id=key_id, hard_delete=hard)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Virtual key '{key_id}' not found.")
    rate_limiter.reset(key_id)
    return {
        "key_id": key_id,
        "revoked": True,
        "deleted": bool(hard),
    }


@router.get("/api/usage", dependencies=[Depends(require_admin_auth)])
async def get_proxy_usage(
    limit: int = Query(default=100, ge=1, le=1000),
    key_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """Return aggregated token metrics and recent request audit logs."""
    return get_usage_summary(limit=limit, key_id=key_id)


@router.get("/api/config", dependencies=[Depends(require_admin_auth)])
async def get_proxy_config() -> Dict[str, Any]:
    """Return non-sensitive proxy configuration (never exposes GEMINI_API_KEY)."""
    settings = get_settings()
    return {
        "gemini_key_configured": bool(settings.gemini_api_key),
        "upstream_base_url": settings.upstream_base_url,
        "default_limited_model": settings.default_limited_model,
        "model_map": settings.model_map,
        "available_virtual_models": list(settings.model_map.keys()),
        "upstream_timeout_seconds": settings.upstream_timeout_seconds,
    }
