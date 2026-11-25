"""Shared pytest fixtures for Model-Gate security, OpenAI compatibility, and admin tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import reload_settings
from app.database import create_virtual_key_record, init_db
from app.main import app
from app.rate_limit import rate_limiter
from app.upstream import set_custom_transport

DUMMY_GEMINI_KEY = "AIzaSyDUMMY_SUPER_SECRET_GEMINI_KEY_99999"
DUMMY_ADMIN_KEY = "adm_master_secret_test_key_7777777"
DUMMY_PEPPER = "pepper_secret_hmac_test_key_8888888"


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Configure an isolated SQLite DB and mock secrets for every test."""
    db_file = str(tmp_path / "test_model_gate.db")
    monkeypatch.setenv("GEMINI_API_KEY", DUMMY_GEMINI_KEY)
    monkeypatch.setenv(
        "UPSTREAM_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai",
    )
    monkeypatch.setenv("ADMIN_MASTER_KEY", DUMMY_ADMIN_KEY)
    monkeypatch.setenv("KEY_HASH_PEPPER", DUMMY_PEPPER)
    monkeypatch.setenv("DATABASE_PATH", db_file)

    reload_settings(
        gemini_api_key=DUMMY_GEMINI_KEY,
        upstream_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        admin_master_key=DUMMY_ADMIN_KEY,
        key_hash_pepper=DUMMY_PEPPER,
        database_path=db_file,
        default_limited_model="limited-default",
        model_map={
            "limited-default": "gemini-2.5-flash",
            "gemini-flash-lite": "gemini-2.5-flash-lite",
            "gemini-flash": "gemini-2.5-flash",
        },
    )
    init_db(db_file)
    rate_limiter.reset()
    set_custom_transport(None)
    yield db_file
    set_custom_transport(None)
    rate_limiter.reset()


@pytest.fixture
def client() -> TestClient:
    """FastAPI TestClient with raise_server_exceptions=False to verify L7 global exception handler."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def consumer_key() -> Dict[str, Any]:
    """Create a default active consumer virtual key with limited-default and gemini-flash allowed."""
    return create_virtual_key_record(
        name="test-consumer",
        allowed_models=["limited-default", "gemini-flash"],
        max_tokens_per_request=256,
        max_input_chars=2000,
        rpm_limit=10,
        token_budget=1000,
    )


@pytest.fixture
def mock_upstream():
    """Helper fixture to register an httpx.MockTransport handler for upstream calls."""

    def _register(handler: Callable[[httpx.Request], httpx.Response]):
        transport = httpx.MockTransport(handler)
        set_custom_transport(transport)
        return transport

    return _register
