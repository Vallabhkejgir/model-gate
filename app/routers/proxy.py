"""OpenAI-compatible /v1/models and /v1/chat/completions proxy endpoints."""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import get_settings
from app.database import authenticate_virtual_key, record_request_usage
from app.models import ChatCompletionRequest
from app.rate_limit import check_token_budget, rate_limiter
from app.upstream import (
    build_openai_error,
    forward_chat_completion,
    prepare_stream_chat_completion,
)

router = APIRouter(prefix="/v1", tags=["OpenAI Compatible Proxy"])

# Safe OpenAI fields forwarded to the upstream provider
_ALLOWED_UPSTREAM_FIELDS = {
    "model",
    "messages",
    "stream",
    "max_tokens",
    "temperature",
    "top_p",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "n",
    "response_format",
    "tools",
    "tool_choice",
    "seed",
}


def _extract_consumer_key(request: Request) -> Optional[str]:
    """Extract the virtual consumer key from Authorization Bearer or x-api-key header."""
    auth_header = request.headers.get("Authorization", "").strip()
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    x_api_key = request.headers.get("x-api-key", "").strip()
    if x_api_key:
        return x_api_key
    return None


def _build_proxy_headers(
    request_id: str,
    key_record: Optional[Dict[str, Any]] = None,
    remaining_rpm: Optional[int] = None,
    remaining_tokens: Optional[int] = None,
    is_stream: bool = False,
) -> Dict[str, str]:
    """L4 Strict Response Header Allowlist: emit only proxy-controlled headers."""
    headers: Dict[str, str] = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "X-Proxy-Request-Id": request_id,
    }
    if key_record:
        headers["X-Consumer-Key-Prefix"] = str(key_record.get("key_prefix", ""))
        headers["X-RateLimit-Limit-RPM"] = str(key_record.get("rpm_limit", 0))
        if remaining_rpm is not None:
            headers["X-RateLimit-Remaining-RPM"] = str(max(0, remaining_rpm))
        if remaining_tokens is not None:
            headers["X-Quota-Remaining-Tokens"] = str(max(0, remaining_tokens))
        else:
            headers["X-Quota-Remaining-Tokens"] = "unlimited"
    if is_stream:
        headers["Content-Type"] = "text/event-stream; charset=utf-8"
        headers["X-Accel-Buffering"] = "no"
        headers["Connection"] = "keep-alive"
    return headers


def _calculate_input_chars(messages: List[Dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    total += len(item["text"])
    return total


def _authenticate_request(
    request: Request, request_id: str
) -> Tuple[Optional[Dict[str, Any]], Optional[JSONResponse]]:
    token = _extract_consumer_key(request)
    if not token:
        err = build_openai_error(
            message="Missing virtual API key. Provide 'Authorization: Bearer vk_live_...'.",
            error_type="authentication_error",
            code="missing_api_key",
        )
        return None, JSONResponse(
            status_code=401,
            content=err,
            headers=_build_proxy_headers(request_id),
        )

    key_record, err_reason = authenticate_virtual_key(token)
    if not key_record:
        err = build_openai_error(
            message=err_reason or "Invalid virtual API key.",
            error_type="authentication_error",
            code="invalid_api_key",
        )
        return None, JSONResponse(
            status_code=401,
            content=err,
            headers=_build_proxy_headers(request_id),
        )

    return key_record, None


@router.get("/models")
async def list_allowed_models(request: Request) -> JSONResponse:
    """Return an OpenAI-formatted model list containing ONLY the key's allowed_models."""
    request_id = f"req_{uuid.uuid4().hex[:16]}"
    key_record, error_resp = _authenticate_request(request, request_id)
    if error_resp is not None:
        return error_resp

    assert key_record is not None
    rpm_remaining = rate_limiter.get_remaining(
        key_record["key_id"], int(key_record["rpm_limit"])
    )
    _, rem_tokens = check_token_budget(key_record)

    created_ts = int(time.time())
    model_cards = [
        {
            "id": model_id,
            "object": "model",
            "created": created_ts,
            "owned_by": "model-gate",
        }
        for model_id in key_record.get("allowed_models", [])
    ]

    return JSONResponse(
        status_code=200,
        content={"object": "list", "data": model_cards},
        headers=_build_proxy_headers(
            request_id=request_id,
            key_record=key_record,
            remaining_rpm=rpm_remaining,
            remaining_tokens=rem_tokens,
        ),
    )


@router.post("/chat/completions", response_model=None)
async def create_chat_completion(
    payload: ChatCompletionRequest, request: Request
) -> JSONResponse | StreamingResponse:
    """OpenAI-compatible chat completions endpoint with model policy and zero-leak forwarding."""
    request_id = f"req_{uuid.uuid4().hex[:16]}"
    settings = get_settings()

    # 1. Authenticate consumer virtual key
    key_record, error_resp = _authenticate_request(request, request_id)
    if error_resp is not None:
        return error_resp
    assert key_record is not None

    key_id = key_record["key_id"]
    rpm_limit = int(key_record["rpm_limit"])

    # 2. Enforce sliding-window RPM rate limit
    rpm_ok, rpm_remaining = rate_limiter.check_and_record(key_id, rpm_limit)
    budget_ok, rem_tokens = check_token_budget(key_record)

    if not rpm_ok:
        err = build_openai_error(
            message=f"Rate limit exceeded for this virtual key ({rpm_limit} RPM). Please slow down.",
            error_type="rate_limit_error",
            code="rate_limit_exceeded",
        )
        return JSONResponse(
            status_code=429,
            content=err,
            headers=_build_proxy_headers(
                request_id, key_record, remaining_rpm=0, remaining_tokens=rem_tokens
            ),
        )

    # 3. Enforce token budget
    if not budget_ok:
        err = build_openai_error(
            message=(
                f"Token budget exhausted for this virtual key "
                f"({key_record['tokens_used']}/{key_record['token_budget']} tokens used)."
            ),
            error_type="rate_limit_error",
            code="token_budget_exhausted",
        )
        return JSONResponse(
            status_code=429,
            content=err,
            headers=_build_proxy_headers(
                request_id,
                key_record,
                remaining_rpm=rpm_remaining,
                remaining_tokens=0,
            ),
        )

    # 4. Enforce model allowlist
    requested_model = payload.model.strip()
    allowed_models: List[str] = key_record.get("allowed_models", [])
    if requested_model not in allowed_models:
        err = build_openai_error(
            message=(
                f"Model '{requested_model}' is not permitted for this API key. "
                f"Allowed models: {allowed_models}"
            ),
            error_type="permission_error",
            code="model_not_allowed",
        )
        return JSONResponse(
            status_code=403,
            content=err,
            headers=_build_proxy_headers(
                request_id,
                key_record,
                remaining_rpm=rpm_remaining,
                remaining_tokens=rem_tokens,
            ),
        )

    upstream_model = settings.resolve_upstream_model(requested_model)

    # 5. Convert and validate messages + enforce max_input_chars
    raw_dict = payload.model_dump(exclude_none=True)
    messages: List[Dict[str, Any]] = [
        m.model_dump(exclude_none=True) for m in payload.messages
    ]
    input_chars = _calculate_input_chars(messages)
    max_input_chars = int(key_record.get("max_input_chars") or 16000)

    if input_chars > max_input_chars:
        err = build_openai_error(
            message=(
                f"Input length ({input_chars} chars) exceeds maximum allowed for this key "
                f"({max_input_chars} chars)."
            ),
            error_type="invalid_request_error",
            code="max_input_chars_exceeded",
        )
        return JSONResponse(
            status_code=400,
            content=err,
            headers=_build_proxy_headers(
                request_id,
                key_record,
                remaining_rpm=rpm_remaining,
                remaining_tokens=rem_tokens,
            ),
        )

    # 6. Inject forced_system_prompt if configured on the key
    forced_prompt = key_record.get("forced_system_prompt")
    if forced_prompt and str(forced_prompt).strip():
        messages = [
            {"role": "system", "content": str(forced_prompt).strip()},
            *messages,
        ]

    # 7. Smart max_tokens clamping to key.max_tokens_per_request
    cap = int(key_record.get("max_tokens_per_request") or 1024)
    req_max = payload.max_tokens or payload.max_completion_tokens
    if req_max and req_max > 0:
        clamped_max_tokens: Optional[int] = min(req_max, cap)
    elif cap <= 2048:
        clamped_max_tokens = cap
    else:
        clamped_max_tokens = None

    # 8. Sanitize outbound payload
    upstream_payload: Dict[str, Any] = {}
    for k, v in raw_dict.items():
        if k in _ALLOWED_UPSTREAM_FIELDS:
            upstream_payload[k] = v

    upstream_payload["model"] = upstream_model
    upstream_payload["messages"] = messages
    if clamped_max_tokens is not None:
        upstream_payload["max_tokens"] = clamped_max_tokens
    else:
        upstream_payload.pop("max_tokens", None)
    upstream_payload["stream"] = bool(payload.stream)

    # 9. Dispatch Streaming vs Non-Streaming
    if payload.stream:

        def _on_stream_done(
            p_tok: int, c_tok: int, t_tok: int, status_code: int, latency_ms: float
        ) -> None:
            record_request_usage(
                request_id=request_id,
                key_id=key_id,
                requested_model=requested_model,
                upstream_model=upstream_model,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                total_tokens=t_tok,
                status_code=status_code,
                latency_ms=latency_ms,
            )

        pre_status, err_payload, sse_generator = await prepare_stream_chat_completion(
            upstream_payload=upstream_payload,
            virtual_model_name=requested_model,
            upstream_model_name=upstream_model,
            on_complete=_on_stream_done,
            settings=settings,
        )
        if err_payload is not None or sse_generator is None:
            return JSONResponse(
                status_code=pre_status,
                content=err_payload,
                headers=_build_proxy_headers(
                    request_id=request_id,
                    key_record=key_record,
                    remaining_rpm=rpm_remaining,
                    remaining_tokens=rem_tokens,
                ),
            )
        return StreamingResponse(
            sse_generator,
            status_code=200,
            media_type="text/event-stream",
            headers=_build_proxy_headers(
                request_id=request_id,
                key_record=key_record,
                remaining_rpm=rpm_remaining,
                remaining_tokens=rem_tokens,
                is_stream=True,
            ),
        )

    status_code, resp_body, p_tok, c_tok, t_tok, latency_ms = (
        await forward_chat_completion(
            upstream_payload=upstream_payload,
            virtual_model_name=requested_model,
            upstream_model_name=upstream_model,
            settings=settings,
        )
    )

    record_request_usage(
        request_id=request_id,
        key_id=key_id,
        requested_model=requested_model,
        upstream_model=upstream_model,
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        total_tokens=t_tok,
        status_code=status_code,
        latency_ms=latency_ms,
    )

    updated_rem_tokens = (
        max(0, rem_tokens - t_tok) if rem_tokens is not None else None
    )

    return JSONResponse(
        status_code=status_code,
        content=resp_body,
        headers=_build_proxy_headers(
            request_id=request_id,
            key_record=key_record,
            remaining_rpm=rpm_remaining,
            remaining_tokens=updated_rem_tokens,
        ),
    )
