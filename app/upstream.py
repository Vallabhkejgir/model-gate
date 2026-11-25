"""Zero-leak async httpx forwarder to the upstream OpenAI-compatible Gemini endpoint."""

from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator, Callable, Dict, Optional, Tuple

import httpx

from app.config import Settings, get_settings
from app.security import redact_secrets

# Custom transport hook used by tests to intercept outbound httpx requests in-memory
_custom_transport: Optional[httpx.AsyncBaseTransport] = None


def set_custom_transport(transport: Optional[httpx.AsyncBaseTransport]) -> None:
    """Configure an optional custom httpx transport (used in test suites)."""
    global _custom_transport
    _custom_transport = transport


def get_custom_transport() -> Optional[httpx.AsyncBaseTransport]:
    """Return the active custom transport if set."""
    return _custom_transport


def build_openai_error(
    message: str,
    error_type: str = "api_error",
    code: Optional[str] = None,
    param: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """Construct an OpenAI-formatted error payload with guaranteed secret scrubbing."""
    clean_message = redact_secrets(message, settings)
    return {
        "error": {
            "message": clean_message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }


def estimate_tokens_from_text(text: str) -> int:
    """Fallback token estimator (~4 chars per token, minimum 1 when non-empty)."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def estimate_prompt_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate prompt token count from a list of OpenAI chat messages."""
    total_chars = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    total_chars += len(part["text"])
    return max(1, (total_chars + 3) // 4)


def map_upstream_error(
    status_code: int,
    raw_body: str,
    settings: Optional[Settings] = None,
) -> Tuple[int, Dict[str, Any]]:
    """L3 Upstream Error Reflection Blocker.

    Never forward raw upstream 401/403/429/5xx payloads containing upstream key hints,
    Google project numbers, or quota account identifiers.
    """
    cfg = settings or get_settings()

    if status_code in (401, 403):
        return (
            502,
            build_openai_error(
                message="Upstream model provider authentication or configuration error.",
                error_type="upstream_error",
                code="upstream_auth_error",
                settings=cfg,
            ),
        )

    if status_code == 429:
        return (
            429,
            build_openai_error(
                message="Upstream capacity temporarily limited, please retry shortly.",
                error_type="rate_limit_error",
                code="upstream_rate_limit",
                settings=cfg,
            ),
        )

    if status_code == 400:
        safe_detail = "Invalid request parameters rejected by upstream model provider."
        try:
            parsed = json.loads(raw_body)
            if isinstance(parsed, dict):
                err_obj = parsed.get("error")
                if isinstance(err_obj, dict) and isinstance(err_obj.get("message"), str):
                    candidate = redact_secrets(err_obj["message"], cfg)
                    # Strip any leaked project/account numbers if present
                    if candidate and len(candidate) <= 400:
                        safe_detail = candidate
        except Exception:
            pass
        return (
            400,
            build_openai_error(
                message=safe_detail,
                error_type="invalid_request_error",
                code="upstream_bad_request",
                settings=cfg,
            ),
        )

    return (
        502,
        build_openai_error(
            message="Upstream model provider returned an unexpected error.",
            error_type="upstream_error",
            code="upstream_service_error",
            settings=cfg,
        ),
    )


def _build_outbound_headers(settings: Settings) -> Dict[str, str]:
    """L1 Credential Stripping & In-Memory Swap: build clean outbound headers."""
    return {
        "Authorization": f"Bearer {settings.gemini_api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "User-Agent": "model-gate/0.1.0",
    }


def _create_async_client(settings: Settings) -> httpx.AsyncClient:
    """L6 SSRF & Redirect Immunity: follow_redirects=False pinned to upstream."""
    kwargs: Dict[str, Any] = {
        "follow_redirects": False,
        "timeout": httpx.Timeout(settings.upstream_timeout_seconds),
    }
    if _custom_transport is not None:
        kwargs["transport"] = _custom_transport
    return httpx.AsyncClient(**kwargs)


async def forward_chat_completion(
    upstream_payload: Dict[str, Any],
    virtual_model_name: str,
    upstream_model_name: str,
    settings: Optional[Settings] = None,
) -> Tuple[int, Dict[str, Any], int, int, int, float]:
    """Forward a non-streaming chat completion request to the upstream Gemini endpoint.

    Returns:
        (http_status, response_dict, prompt_tokens, completion_tokens, total_tokens, latency_ms)
    """
    cfg = settings or get_settings()
    start_time = time.perf_counter()

    if not cfg.gemini_api_key:
        latency = (time.perf_counter() - start_time) * 1000.0
        return (
            502,
            build_openai_error(
                message="Upstream model provider authentication or configuration error.",
                error_type="upstream_error",
                code="upstream_not_configured",
                settings=cfg,
            ),
            0,
            0,
            0,
            latency,
        )

    target_url = f"{cfg.upstream_base_url}/chat/completions"
    headers = _build_outbound_headers(cfg)

    try:
        async with _create_async_client(cfg) as client:
            resp = await client.post(
                target_url,
                headers=headers,
                json=upstream_payload,
            )
    except Exception as exc:
        latency = (time.perf_counter() - start_time) * 1000.0
        # L2 & L7: Never expose exception details or upstream key in error responses
        _ = redact_secrets(str(exc), cfg)
        return (
            502,
            build_openai_error(
                message="Upstream model provider connection failed.",
                error_type="upstream_error",
                code="upstream_network_error",
                settings=cfg,
            ),
            0,
            0,
            0,
            latency,
        )

    latency = (time.perf_counter() - start_time) * 1000.0

    # Block redirects (L6)
    if 300 <= resp.status_code < 400:
        return (
            502,
            build_openai_error(
                message="Upstream redirect blocked by security policy.",
                error_type="upstream_error",
                code="upstream_redirect_blocked",
                settings=cfg,
            ),
            0,
            0,
            0,
            latency,
        )

    raw_text = resp.text
    if resp.status_code != 200:
        mapped_status, err_payload = map_upstream_error(resp.status_code, raw_text, cfg)
        return mapped_status, err_payload, 0, 0, 0, latency

    # Scrub raw response body before parsing
    scrubbed_text = redact_secrets(raw_text, cfg)
    try:
        body = json.loads(scrubbed_text)
        if not isinstance(body, dict):
            raise ValueError("Upstream JSON root is not an object")
    except Exception:
        return (
            502,
            build_openai_error(
                message="Upstream model provider returned malformed JSON.",
                error_type="upstream_error",
                code="upstream_malformed_response",
                settings=cfg,
            ),
            0,
            0,
            0,
            latency,
        )

    # Rewrite model back to the consumer's virtual model alias
    body["model"] = virtual_model_name

    # Extract or estimate token usage
    usage = body.get("usage")
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    if isinstance(usage, dict):
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or 0)

    if prompt_tokens <= 0:
        prompt_tokens = estimate_prompt_tokens(upstream_payload.get("messages", []))

    if completion_tokens <= 0:
        collected_out = []
        for choice in body.get("choices", []):
            if isinstance(choice, dict):
                msg = choice.get("message") or {}
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    collected_out.append(msg["content"])
        completion_tokens = estimate_tokens_from_text("".join(collected_out))

    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens

    body["usage"] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

    return 200, body, prompt_tokens, completion_tokens, total_tokens, latency


async def stream_chat_completion(
    upstream_payload: Dict[str, Any],
    virtual_model_name: str,
    upstream_model_name: str,
    on_complete: Callable[[int, int, int, int, float], None],
    settings: Optional[Settings] = None,
) -> AsyncGenerator[str, None]:
    """Stream SSE chunks from upstream while scrubbing secrets and rewriting model aliases."""
    cfg = settings or get_settings()
    start_time = time.perf_counter()
    prompt_tokens = estimate_prompt_tokens(upstream_payload.get("messages", []))
    completion_tokens = 0
    total_tokens = 0
    accumulated_text: list[str] = []
    status_recorded = 200

    if not cfg.gemini_api_key:
        status_recorded = 502
        err = build_openai_error(
            message="Upstream model provider authentication or configuration error.",
            error_type="upstream_error",
            code="upstream_not_configured",
            settings=cfg,
        )
        yield f"data: {json.dumps(err)}\n\n"
        yield "data: [DONE]\n\n"
        latency = (time.perf_counter() - start_time) * 1000.0
        on_complete(0, 0, 0, status_recorded, latency)
        return

    target_url = f"{cfg.upstream_base_url}/chat/completions"
    headers = _build_outbound_headers(cfg)

    try:
        async with _create_async_client(cfg) as client:
            async with client.stream(
                "POST",
                target_url,
                headers=headers,
                json=upstream_payload,
            ) as resp:
                if 300 <= resp.status_code < 400:
                    status_recorded = 502
                    err = build_openai_error(
                        message="Upstream redirect blocked by security policy.",
                        error_type="upstream_error",
                        code="upstream_redirect_blocked",
                        settings=cfg,
                    )
                    yield f"data: {json.dumps(err)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                if resp.status_code != 200:
                    raw_err_bytes = await resp.aread()
                    raw_err_text = raw_err_bytes.decode("utf-8", errors="replace")
                    mapped_status, err_payload = map_upstream_error(
                        resp.status_code, raw_err_text, cfg
                    )
                    status_recorded = mapped_status
                    yield f"data: {json.dumps(err_payload)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                saw_done = False
                async for raw_line in resp.aiter_lines():
                    if raw_line is None:
                        continue
                    line = redact_secrets(raw_line.strip(), cfg)
                    if not line:
                        continue

                    if not line.startswith("data:"):
                        # Scrub and ignore non-data comments or headers
                        continue

                    data_str = line[len("data:") :].strip()
                    if data_str == "[DONE]":
                        saw_done = True
                        yield "data: [DONE]\n\n"
                        break

                    try:
                        chunk_obj = json.loads(data_str)
                    except Exception:
                        # Never emit raw unparsed upstream frames
                        continue

                    if not isinstance(chunk_obj, dict):
                        continue

                    # Intercept upstream inline error frame inside SSE stream
                    if "error" in chunk_obj:
                        status_recorded = 502
                        safe_err = build_openai_error(
                            message="Upstream stream interrupted by provider error.",
                            error_type="upstream_error",
                            code="upstream_stream_error",
                            settings=cfg,
                        )
                        yield f"data: {json.dumps(safe_err)}\n\n"
                        yield "data: [DONE]\n\n"
                        saw_done = True
                        break

                    # Rewrite model field to virtual alias
                    chunk_obj["model"] = virtual_model_name

                    # Track streamed content and usage if provided
                    for choice in chunk_obj.get("choices", []):
                        if isinstance(choice, dict):
                            delta = choice.get("delta") or {}
                            if isinstance(delta, dict) and isinstance(
                                delta.get("content"), str
                            ):
                                content_piece = redact_secrets(delta["content"], cfg)
                                delta["content"] = content_piece
                                accumulated_text.append(content_piece)

                    usage_obj = chunk_obj.get("usage")
                    if isinstance(usage_obj, dict):
                        if usage_obj.get("prompt_tokens"):
                            prompt_tokens = int(usage_obj["prompt_tokens"])
                        if usage_obj.get("completion_tokens"):
                            completion_tokens = int(usage_obj["completion_tokens"])
                        if usage_obj.get("total_tokens"):
                            total_tokens = int(usage_obj["total_tokens"])

                    safe_serialized = redact_secrets(json.dumps(chunk_obj), cfg)
                    yield f"data: {safe_serialized}\n\n"

                if not saw_done:
                    yield "data: [DONE]\n\n"

    except Exception as exc:
        _ = redact_secrets(str(exc), cfg)
        status_recorded = 502
        err = build_openai_error(
            message="Upstream model provider stream connection failed.",
            error_type="upstream_error",
            code="upstream_network_error",
            settings=cfg,
        )
        yield f"data: {json.dumps(err)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        if completion_tokens <= 0 and accumulated_text:
            completion_tokens = estimate_tokens_from_text("".join(accumulated_text))
        if total_tokens <= 0:
            if status_recorded == 200:
                total_tokens = prompt_tokens + completion_tokens
            else:
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
        latency = (time.perf_counter() - start_time) * 1000.0
        on_complete(
            prompt_tokens,
            completion_tokens,
            total_tokens,
            status_recorded,
            latency,
        )
