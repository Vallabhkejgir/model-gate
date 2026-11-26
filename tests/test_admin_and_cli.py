"""Tests for virtual key lifecycle, admin REST control plane, rate/budget enforcement, and CLI."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import httpx
from fastapi.testclient import TestClient

import cli
from app.database import create_virtual_key_record
from tests.conftest import DUMMY_ADMIN_KEY


def _register_ok_upstream(mock_upstream, total_tokens: int = 25):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-ok",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "gemini-2.5-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "OK"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": total_tokens - 10,
                    "total_tokens": total_tokens,
                },
            },
        )

    mock_upstream(handler)


def test_invalid_revoked_and_expired_virtual_keys_return_401(client: TestClient):
    """Verify invalid format, tampered secret, revoked key, and expired key all return 401."""
    # Missing key
    r_missing = client.get("/v1/models")
    assert r_missing.status_code == 401

    # Invalid prefix
    r_bad = client.get(
        "/v1/models", headers={"Authorization": "Bearer sk-not-a-virtual-key"}
    )
    assert r_bad.status_code == 401

    # Tampered signature
    valid_key = create_virtual_key_record(name="tamper-test")
    tampered = valid_key["api_key"][:-4] + "XXXX"
    r_tampered = client.get(
        "/v1/models", headers={"Authorization": f"Bearer {tampered}"}
    )
    assert r_tampered.status_code == 401

    # Revoked key via Admin API
    rev_resp = client.delete(
        f"/admin/api/keys/{valid_key['key_id']}",
        headers={"X-Admin-Key": DUMMY_ADMIN_KEY},
    )
    assert rev_resp.status_code == 200
    r_revoked = client.get(
        "/v1/models",
        headers={"Authorization": f"Bearer {valid_key['api_key']}"},
    )
    assert r_revoked.status_code == 401
    assert "revoked" in r_revoked.json()["error"]["message"].lower()

    # Expired key
    past_iso = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    expired_key = create_virtual_key_record(
        name="expired-test", expires_at=past_iso
    )
    r_expired = client.get(
        "/v1/models",
        headers={"Authorization": f"Bearer {expired_key['api_key']}"},
    )
    assert r_expired.status_code == 401
    assert "expired" in r_expired.json()["error"]["message"].lower()


def test_rpm_rate_limit_and_token_budget_enforcement(
    client: TestClient, mock_upstream
):
    """Verify per-key RPM rate limit -> 429 and token budget exhaustion -> 429."""
    _register_ok_upstream(mock_upstream, total_tokens=30)

    # 1. Test RPM limit = 2 requests per minute
    rpm_key = create_virtual_key_record(
        name="rpm-limited",
        allowed_models=["limited-default"],
        rpm_limit=2,
        token_budget=1000,
    )
    for _ in range(2):
        r_ok = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {rpm_key['api_key']}"},
            json={
                "model": "limited-default",
                "messages": [{"role": "user", "content": "Ping"}],
            },
        )
        assert r_ok.status_code == 200

    r_limited = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {rpm_key['api_key']}"},
        json={
            "model": "limited-default",
            "messages": [{"role": "user", "content": "Ping 3"}],
        },
    )
    assert r_limited.status_code == 429
    assert r_limited.json()["error"]["code"] == "rate_limit_exceeded"

    # 2. Test Token budget = 50 tokens (first request uses 30, second uses 30 -> 60 >= 50, third blocked)
    budget_key = create_virtual_key_record(
        name="budget-limited",
        allowed_models=["limited-default"],
        rpm_limit=50,
        token_budget=50,
    )
    r1 = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {budget_key['api_key']}"},
        json={
            "model": "limited-default",
            "messages": [{"role": "user", "content": "Req 1"}],
        },
    )
    assert r1.status_code == 200

    r2 = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {budget_key['api_key']}"},
        json={
            "model": "limited-default",
            "messages": [{"role": "user", "content": "Req 2"}],
        },
    )
    assert r2.status_code == 200

    r3 = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {budget_key['api_key']}"},
        json={
            "model": "limited-default",
            "messages": [{"role": "user", "content": "Req 3 blocked"}],
        },
    )
    assert r3.status_code == 429
    assert r3.json()["error"]["code"] == "token_budget_exhausted"


def test_admin_api_lifecycle_and_dashboard(client: TestClient, mock_upstream):
    """Test admin authentication, key creation, listing (no hash/secret leak), policy update, and dashboard HTML."""
    # Unauthorized admin call fails with 401
    unauth = client.get("/admin/api/keys", headers={"X-Admin-Key": "wrong-key"})
    assert unauth.status_code == 401

    # Web dashboard loads
    dash = client.get("/admin")
    assert dash.status_code == 200
    assert "Model-Gate Operator Console" in dash.text

    # Create key via POST /admin/api/keys
    create_resp = client.post(
        "/admin/api/keys",
        headers={"X-Admin-Key": DUMMY_ADMIN_KEY},
        json={
            "name": "frontend-team",
            "allowed_models": ["limited-default"],
            "max_tokens_per_request": 512,
            "rpm_limit": 20,
            "token_budget": 5000,
        },
    )
    assert create_resp.status_code == 201
    created = create_resp.json()
    assert created["api_key"].startswith("vk_live_")
    assert "key_hash" not in created
    key_id = created["key_id"]

    # List keys via GET /admin/api/keys — plaintext api_key and key_hash must NOT be present
    list_resp = client.get(
        "/admin/api/keys", headers={"X-Admin-Key": DUMMY_ADMIN_KEY}
    )
    assert list_resp.status_code == 200
    listed = list_resp.json()["data"]
    matching = [k for k in listed if k["key_id"] == key_id][0]
    assert "api_key" not in matching
    assert "key_hash" not in matching

    # Update policy via PATCH /admin/api/keys/{key_id}
    patch_resp = client.patch(
        f"/admin/api/keys/{key_id}",
        headers={"X-Admin-Key": DUMMY_ADMIN_KEY},
        json={
            "allowed_models": ["limited-default", "gemini-flash"],
            "max_tokens_per_request": 2048,
            "token_budget": 10000,
        },
    )
    assert patch_resp.status_code == 200
    patched = patch_resp.json()
    assert patched["allowed_models"] == ["limited-default", "gemini-flash"]
    assert patched["max_tokens_per_request"] == 2048
    assert patched["token_budget"] == 10000

    # Perform a request and verify GET /admin/api/usage records it
    _register_ok_upstream(mock_upstream, total_tokens=42)
    client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {created['api_key']}"},
        json={
            "model": "gemini-flash",
            "messages": [{"role": "user", "content": "Hello flash"}],
        },
    )

    usage_resp = client.get(
        f"/admin/api/usage?key_id={key_id}",
        headers={"X-Admin-Key": DUMMY_ADMIN_KEY},
    )
    assert usage_resp.status_code == 200
    usage_data = usage_resp.json()
    assert usage_data["summary"]["total_requests"] == 1
    assert usage_data["summary"]["total_tokens"] == 42


def test_cli_commands_create_list_usage_and_revoke(isolated_env, capsys):
    """Test cli.py create-key, list-keys, usage, and revoke-key commands."""
    # 1. create-key
    rc_create = cli.main(
        [
            "--db",
            isolated_env,
            "--json",
            "create-key",
            "--name",
            "cli-alice",
            "--models",
            "limited-default,gemini-flash",
            "--max-tokens",
            "1024",
            "--rpm",
            "15",
            "--budget",
            "50000",
        ]
    )
    assert rc_create == 0
    created_out = json.loads(capsys.readouterr().out)
    assert created_out["name"] == "cli-alice"
    assert created_out["api_key"].startswith("vk_live_")
    key_id = created_out["key_id"]

    # 2. list-keys
    rc_list = cli.main(["--db", isolated_env, "--json", "list-keys"])
    assert rc_list == 0
    keys_list = json.loads(capsys.readouterr().out)
    assert any(k["key_id"] == key_id and k["is_active"] is True for k in keys_list)

    # 3. usage
    rc_usage = cli.main(["--db", isolated_env, "--json", "usage"])
    assert rc_usage == 0
    usage_out = json.loads(capsys.readouterr().out)
    assert "summary" in usage_out
    assert "logs" in usage_out

    # 4. revoke-key
    rc_revoke = cli.main(["--db", isolated_env, "--json", "revoke-key", key_id])
    assert rc_revoke == 0
    revoke_out = json.loads(capsys.readouterr().out)
    assert revoke_out["revoked"] is True

    # Verify key is now inactive in list-keys
    cli.main(["--db", isolated_env, "--json", "list-keys"])
    keys_after = json.loads(capsys.readouterr().out)
    target = [k for k in keys_after if k["key_id"] == key_id][0]
    assert target["is_active"] is False


def test_admin_session_cookie_persistence_across_reload_and_logout(
    client: TestClient,
):
    """Verify POST /admin/api/session sets HttpOnly cookie allowing Ctrl+R reloads without X-Admin-Key header."""
    # 1. Invalid key rejected
    bad_login = client.post(
        "/admin/api/session", json={"admin_key": "invalid-admin-key"}
    )
    assert bad_login.status_code == 401

    # 2. Valid key sets mg_admin_session HttpOnly cookie
    login_resp = client.post(
        "/admin/api/session", json={"admin_key": DUMMY_ADMIN_KEY}
    )
    assert login_resp.status_code == 200
    assert login_resp.json()["authenticated"] is True
    assert "mg_admin_session" in login_resp.cookies

    # 3. Subsequent admin API calls succeed using ONLY the session cookie (no X-Admin-Key header, simulating Ctrl+R)
    keys_resp = client.get("/admin/api/keys")
    assert keys_resp.status_code == 200
    assert keys_resp.json()["object"] == "list"

    cfg_resp = client.get("/admin/api/config")
    assert cfg_resp.status_code == 200
    assert cfg_resp.json()["gemini_key_configured"] is True

    # 4. Logout clears the session cookie and subsequent calls without header return 401
    logout_resp = client.post("/admin/api/logout")
    assert logout_resp.status_code == 200
    assert logout_resp.json()["authenticated"] is False

    client.cookies.clear()
    after_logout = client.get("/admin/api/keys")
    assert after_logout.status_code == 401
