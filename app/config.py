"""Configuration loader and validated settings for Model-Gate."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL_MAP: Dict[str, str] = {
    "limited-default": "gemini-2.5-flash",
    "gemini-flash-lite": "gemini-2.5-flash-lite",
    "gemini-flash": "gemini-2.5-flash",
}


def _parse_model_map(raw: str | None) -> Dict[str, str]:
    if not raw or not raw.strip():
        return dict(DEFAULT_MODEL_MAP)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and all(
            isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()
        ):
            return parsed
    except json.JSONDecodeError:
        pass
    return dict(DEFAULT_MODEL_MAP)


def _normalize_database_url(raw: str | None) -> str:
    """Normalize postgres:// URLs (from Render/Neon/Supabase) to standard postgresql://."""
    if not raw:
        return ""
    cleaned = raw.strip()
    if cleaned.startswith("postgres://"):
        return "postgresql://" + cleaned[len("postgres://") :]
    return cleaned


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    gemini_api_key: str = field(
        default_factory=lambda: os.environ.get("GEMINI_API_KEY", "").strip()
    )
    upstream_base_url: str = field(
        default_factory=lambda: os.environ.get(
            "UPSTREAM_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai",
        )
        .strip()
        .rstrip("/")
    )
    admin_master_key: str = field(
        default_factory=lambda: os.environ.get(
            "ADMIN_MASTER_KEY", "dev-admin-master-key-change-me"
        ).strip()
    )
    key_hash_pepper: str = field(
        default_factory=lambda: os.environ.get(
            "KEY_HASH_PEPPER", "dev-key-hash-pepper-change-me"
        ).strip()
    )
    default_limited_model: str = field(
        default_factory=lambda: os.environ.get(
            "DEFAULT_LIMITED_MODEL", "limited-default"
        ).strip()
    )
    model_map: Dict[str, str] = field(
        default_factory=lambda: _parse_model_map(os.environ.get("MODEL_MAP"))
    )
    database_url: str = field(
        default_factory=lambda: _normalize_database_url(
            os.environ.get("DATABASE_URL", "")
        )
    )
    database_path: str = field(
        default_factory=lambda: os.environ.get("DATABASE_PATH", "model_gate.db").strip()
    )
    key_cache_ttl_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("KEY_CACHE_TTL_SECONDS", "30.0")
        )
    )
    max_usage_log_rows: int = field(
        default_factory=lambda: int(
            os.environ.get("MAX_USAGE_LOG_ROWS", "5000")
        )
    )
    bootstrap_key_records: str = field(
        default_factory=lambda: os.environ.get("BOOTSTRAP_KEY_RECORDS", "").strip()
    )
    upstream_timeout_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("UPSTREAM_TIMEOUT_SECONDS", "60.0")
        )
    )
    upstream_stream_read_timeout_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("UPSTREAM_STREAM_READ_TIMEOUT_SECONDS", "300.0")
        )
    )
    sse_heartbeat_interval_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("SSE_HEARTBEAT_INTERVAL_SECONDS", "15.0")
        )
    )
    upstream_max_connections: int = field(
        default_factory=lambda: int(
            os.environ.get("UPSTREAM_MAX_CONNECTIONS", "100")
        )
    )
    upstream_max_keepalive_connections: int = field(
        default_factory=lambda: int(
            os.environ.get("UPSTREAM_MAX_KEEPALIVE_CONNECTIONS", "20")
        )
    )

    def __post_init__(self) -> None:
        self.database_url = _normalize_database_url(self.database_url)

    def storage_backend(self) -> str:
        """Return 'postgresql' when a Postgres DATABASE_URL is configured, else 'sqlite'."""
        url = (self.database_url or "").strip().lower()
        if url.startswith("postgresql://") or url.startswith("postgres://"):
            return "postgresql"
        return "sqlite"

    def resolve_upstream_model(self, requested_model: str) -> str:
        """Map a consumer-facing virtual model name to the actual upstream model."""
        return self.model_map.get(requested_model, requested_model)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the active Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings(**overrides) -> Settings:
    """Reload settings from environment and apply optional test overrides."""
    global _settings
    s = Settings()
    for key, value in overrides.items():
        if hasattr(s, key):
            if key == "database_url":
                value = _normalize_database_url(value)
            setattr(s, key, value)
    _settings = s
    return _settings
