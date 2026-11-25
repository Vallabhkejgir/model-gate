"""OpenAI compatibility tests for /v1/models and /v1/chat/completions (JSON + SSE streaming)."""

from __future__ import annotations

import json

import httpx
from fastapi.testclient import TestClient

from app.database import create_virtual_key_record, get_virtual_key_by_id


def test_v1_models_returns_only_allowed_models_for_key(client: TestClient):
    """GET /v1/models returns an OpenAI list containing only the consumer key's allowed_models."""
    key_a = create_virtual_key_record(
        name="alice", allowed_models=["limited-default"]
    )
    key_b = create_virtual_key_record(
        name="bob", allowed_models=["gemini-flash-lite", "gemini-flash"]
    )

    resp_a = client.get(
        "/v1/models", headers={"Authorization": f"Bearer {key_a['api_key']}"}
    )
    assert resp_a.status_code == 200
    data_a = resp_a.json()
    assert data_a["object"] == "list"
    assert [m["id"] for m in data_a["data"]] == ["limited-default"]

    resp_b = client.get(
        "/v1/models", headers={"x-api-key": key_b["api_key"]}
    )
    assert resp_b.status_code == 200
    data_b = resp_b.json()
    assert [m["id"] for m in data_b["data"]] == [
        "gemini-flash-lite",
        "gemini-flash",
    ]


def test_disallowed_model_returns_403_openai_error(
    client: TestClient, consumer_key
):
    """Requesting a model not in key.allowed_models returns 403 Forbidden with code=model_not_allowed."""
    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {consumer_key['api_key']}"},
        json={
            "model": "gemini-2.5-pro-ultra-unrestricted",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert resp.status_code == 403
    err = resp.json()["error"]
    assert err["type"] == "permission_error"
    assert err["code"] == "model_not_allowed"
    assert "gemini-2.5-pro-ultra-unrestricted" in err["message"]


def test_model_alias_mapping_max_tokens_clamping_and_system_prompt_injection(
    client: TestClient, mock_upstream
):
    """Verify alias resolution, max_tokens clamping, forced_system_prompt prepending, and response model rewrite."""
    custom_key = create_virtual_key_record(
        name="clamped-user",
        allowed_models=["gemini-flash-lite"],
        forced_system_prompt="You are a concise assistant.",
        max_tokens_per_request=128,
        max_input_chars=500,
    )

    captured_payload = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-999",
                "object": "chat.completion",
                "created": 1700000100,
                "model": "gemini-2.5-flash-lite",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Concise reply.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 4,
                    "total_tokens": 16,
                },
            },
        )

    mock_upstream(handler)

    # Consumer requests max_tokens=4096, which exceeds key policy of 128
    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {custom_key['api_key']}"},
        json={
            "model": "gemini-flash-lite",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": "Tell me a story"}],
        },
    )
    assert resp.status_code == 200

    # Verify upstream received mapped model, clamped max_tokens, and forced system prompt
    assert captured_payload["model"] == "gemini-2.5-flash-lite"
    assert captured_payload["max_tokens"] == 128
    assert len(captured_payload["messages"]) == 2
    assert captured_payload["messages"][0] == {
        "role": "system",
        "content": "You are a concise assistant.",
    }

    # Verify consumer sees virtual model alias in response and usage was recorded
    body = resp.json()
    assert body["model"] == "gemini-flash-lite"
    assert body["usage"]["total_tokens"] == 16

    updated_key = get_virtual_key_by_id(custom_key["key_id"])
    assert updated_key is not None
    assert updated_key["tokens_used"] == 16
    assert updated_key["requests_count"] == 1


def test_max_input_chars_enforcement(client: TestClient):
    """Exceeding max_input_chars returns 400 invalid_request_error before calling upstream."""
    small_key = create_virtual_key_record(
        name="short-input-key",
        allowed_models=["limited-default"],
        max_input_chars=20,
    )
    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {small_key['api_key']}"},
        json={
            "model": "limited-default",
            "messages": [
                {
                    "role": "user",
                    "content": "This input string is definitely longer than twenty characters.",
                }
            ],
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "max_input_chars_exceeded"


def test_streaming_sse_openai_compatibility_and_token_accounting(
    client: TestClient, consumer_key, mock_upstream
):
    """Streaming chat completion yields OpenAI SSE chunks with virtual model alias and updates token usage."""
    sse_stream = (
        'data: {"id":"chatcmpl-s1","object":"chat.completion.chunk","model":"gemini-2.5-flash",'
        '"choices":[{"index":0,"delta":{"role":"assistant","content":"Hello "},"finish_reason":null}]}\n\n'
        'data: {"id":"chatcmpl-s1","object":"chat.completion.chunk","model":"gemini-2.5-flash",'
        '"choices":[{"index":0,"delta":{"content":"world!"},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=sse_stream.encode("utf-8"),
        )

    mock_upstream(handler)

    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {consumer_key['api_key']}"},
        json={
            "model": "limited-default",
            "stream": True,
            "messages": [{"role": "user", "content": "Say hello world"}],
        },
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")

    lines = [line for line in resp.text.splitlines() if line.startswith("data: ")]
    assert len(lines) == 3
    assert lines[-1] == "data: [DONE]"

    first_chunk = json.loads(lines[0][len("data: ") :])
    second_chunk = json.loads(lines[1][len("data: ") :])
    assert first_chunk["model"] == "limited-default"
    assert second_chunk["model"] == "limited-default"
    assert (
        first_chunk["choices"][0]["delta"]["content"]
        + second_chunk["choices"][0]["delta"]["content"]
        == "Hello world!"
    )

    updated = get_virtual_key_by_id(consumer_key["key_id"])
    assert updated is not None
    assert updated["tokens_used"] == 15
    assert updated["requests_count"] == 1
