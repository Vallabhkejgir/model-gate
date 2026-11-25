"""Zero-leak security test suite proving GEMINI_API_KEY never leaks under any condition."""

from __future__ import annotations

import base64
import json
import urllib.parse

import httpx
from fastapi.testclient import TestClient

from app.security import redact_secrets
from tests.conftest import DUMMY_ADMIN_KEY, DUMMY_GEMINI_KEY, DUMMY_PEPPER


def _assert_no_secret_in_response(resp, consumer_key_str: str | None = None) -> None:
    """Assert that neither body nor headers contain any raw or encoded upstream secrets."""
    raw_body = resp.text
    headers_dump = json.dumps(dict(resp.headers))

    forbidden_values = [
        DUMMY_GEMINI_KEY,
        DUMMY_ADMIN_KEY,
        DUMMY_PEPPER,
        urllib.parse.quote(DUMMY_GEMINI_KEY, safe=""),
        base64.b64encode(DUMMY_GEMINI_KEY.encode("utf-8")).decode("ascii"),
    ]
    for secret_val in forbidden_values:
        assert secret_val not in raw_body, f"Leaked secret {secret_val} in response body!"
        assert (
            secret_val not in headers_dump
        ), f"Leaked secret {secret_val} in response headers!"

    # L4: Ensure upstream Google / internal headers are never forwarded
    lower_headers = {k.lower() for k in resp.headers.keys()}
    for banned in ("x-goog-api-key", "x-goog-request-params", "alt-svc", "openai-organization"):
        assert banned not in lower_headers


def test_redact_secrets_scrubs_raw_encoded_and_patterns():
    """L2 unit test: verify direct, URL-encoded, Base64, and regex pattern scrubbing."""
    b64_key = base64.b64encode(DUMMY_GEMINI_KEY.encode()).decode()
    sample = (
        f"Error with key={DUMMY_GEMINI_KEY} and b64={b64_key} "
        f"and master={DUMMY_ADMIN_KEY} and Bearer sk-proj1234567890abcdefghij"
    )
    cleaned = redact_secrets(sample)
    assert DUMMY_GEMINI_KEY not in cleaned
    assert b64_key not in cleaned
    assert DUMMY_ADMIN_KEY not in cleaned
    assert "sk-proj1234567890abcdefghij" not in cleaned
    assert "[REDACTED_SECRET]" in cleaned


def test_no_leak_on_normal_json_and_response_header_stripping(
    client: TestClient, consumer_key, mock_upstream
):
    """L1, L2, L4: Consumer token is stripped, GEMINI_API_KEY injected only to upstream, and upstream headers stripped."""
    captured_auth = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_auth["header"] = request.headers.get("Authorization")
        return httpx.Response(
            200,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": DUMMY_GEMINI_KEY,
                "x-goog-project": "987654321",
                "alt-svc": 'h3=":443"',
            },
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "gemini-2.5-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"Hello! Echo check: {DUMMY_GEMINI_KEY}",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 6,
                    "total_tokens": 14,
                },
            },
        )

    mock_upstream(handler)

    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {consumer_key['api_key']}"},
        json={
            "model": "limited-default",
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    assert resp.status_code == 200
    assert captured_auth["header"] == f"Bearer {DUMMY_GEMINI_KEY}"
    _assert_no_secret_in_response(resp)
    data = resp.json()
    assert data["model"] == "limited-default"
    assert "[REDACTED_SECRET]" in data["choices"][0]["message"]["content"]


def test_no_leak_on_upstream_401_echoing_gemini_key(
    client: TestClient, consumer_key, mock_upstream
):
    """L3: When upstream returns 401 echoing the Gemini key, proxy maps to 502 and never leaks the key."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={
                "error": {
                    "code": 401,
                    "message": f"API key not valid. Received key: {DUMMY_GEMINI_KEY} for project 123456789",
                    "status": "UNAUTHENTICATED",
                }
            },
        )

    mock_upstream(handler)

    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {consumer_key['api_key']}"},
        json={
            "model": "limited-default",
            "messages": [{"role": "user", "content": "Test 401"}],
        },
    )
    assert resp.status_code == 502
    _assert_no_secret_in_response(resp)
    assert "123456789" not in resp.text
    assert resp.json()["error"]["code"] == "upstream_auth_error"


def test_no_leak_on_upstream_429_and_500_containing_key(
    client: TestClient, consumer_key, mock_upstream
):
    """L3: Upstream 429 and 500 responses echoing GEMINI_API_KEY never reach the consumer."""

    def handler_429(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={
                "error": {
                    "message": f"Quota exceeded for key {DUMMY_GEMINI_KEY} on billing account 998877"
                }
            },
        )

    mock_upstream(handler_429)
    resp_429 = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {consumer_key['api_key']}"},
        json={
            "model": "limited-default",
            "messages": [{"role": "user", "content": "Test 429"}],
        },
    )
    assert resp_429.status_code == 429
    _assert_no_secret_in_response(resp_429)
    assert "998877" not in resp_429.text

    def handler_500(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            text=f"Internal upstream crash for Authorization: Bearer {DUMMY_GEMINI_KEY}",
        )

    mock_upstream(handler_500)
    resp_500 = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {consumer_key['api_key']}"},
        json={
            "model": "limited-default",
            "messages": [{"role": "user", "content": "Test 500"}],
        },
    )
    assert resp_500.status_code == 502
    _assert_no_secret_in_response(resp_500)


def test_no_leak_on_httpx_network_exception_containing_key(
    client: TestClient, consumer_key, mock_upstream
):
    """L2 & L7: Network exception inside httpx whose message includes GEMINI_API_KEY is scrubbed."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout(
            f"Timeout connecting to upstream with header Authorization: Bearer {DUMMY_GEMINI_KEY}",
            request=request,
        )

    mock_upstream(handler)
    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {consumer_key['api_key']}"},
        json={
            "model": "limited-default",
            "messages": [{"role": "user", "content": "Test timeout"}],
        },
    )
    assert resp.status_code == 502
    _assert_no_secret_in_response(resp)
    assert resp.json()["error"]["code"] == "upstream_network_error"


def test_no_leak_on_streaming_sse_chunks_and_errors(
    client: TestClient, consumer_key, mock_upstream
):
    """L2 & L3: Streaming SSE chunks that contain GEMINI_API_KEY are scrubbed in real time."""
    sse_payload = (
        f'data: {{"id":"1","object":"chat.completion.chunk","model":"gemini-2.5-flash",'
        f'"choices":[{{"index":0,"delta":{{"content":"Secret is {DUMMY_GEMINI_KEY}"}}}}]}}\n\n'
        f"data: [DONE]\n\n"
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "Content-Type": "text/event-stream",
                "x-goog-api-key": DUMMY_GEMINI_KEY,
            },
            content=sse_payload.encode("utf-8"),
        )

    mock_upstream(handler)

    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {consumer_key['api_key']}"},
        json={
            "model": "limited-default",
            "stream": True,
            "messages": [{"role": "user", "content": "Stream test"}],
        },
    )
    assert resp.status_code == 200
    _assert_no_secret_in_response(resp)
    assert "[REDACTED_SECRET]" in resp.text
    assert '"model": "limited-default"' in resp.text


def test_no_leak_on_admin_config_and_v1_models_and_wildcard_routes(
    client: TestClient, consumer_key
):
    """L5: /admin/api/config, /v1/models, /health, and arbitrary routes never expose GEMINI_API_KEY."""
    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    _assert_no_secret_in_response(resp_health)
    assert resp_health.json()["upstream_configured"] is True

    resp_models = client.get(
        "/v1/models",
        headers={"Authorization": f"Bearer {consumer_key['api_key']}"},
    )
    assert resp_models.status_code == 200
    _assert_no_secret_in_response(resp_models)

    resp_cfg = client.get(
        "/admin/api/config",
        headers={"X-Admin-Key": DUMMY_ADMIN_KEY},
    )
    assert resp_cfg.status_code == 200
    _assert_no_secret_in_response(resp_cfg)
    cfg_json = resp_cfg.json()
    assert cfg_json["gemini_key_configured"] is True
    assert "gemini_api_key" not in cfg_json

    # L5: Arbitrary unexposed route returns 404 without forwarding to upstream
    resp_unexposed = client.get(
        "/v1/files",
        headers={"Authorization": f"Bearer {consumer_key['api_key']}"},
    )
    assert resp_unexposed.status_code == 404
    _assert_no_secret_in_response(resp_unexposed)


def test_ssrf_redirect_blocked(client: TestClient, consumer_key, mock_upstream):
    """L6: Upstream 302 redirect is blocked rather than followed with the Authorization header."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers={"Location": "https://evil.example.com/capture"},
        )

    mock_upstream(handler)
    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {consumer_key['api_key']}"},
        json={
            "model": "limited-default",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert resp.status_code == 502
    _assert_no_secret_in_response(resp)
    assert resp.json()["error"]["code"] == "upstream_redirect_blocked"
