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
    database_path: str = field(
        default_factory=lambda: os.environ.get("DATABASE_PATH", "model_gate.db").strip()
    )
    upstream_timeout_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("UPSTREAM_TIMEOUT_SECONDS", "60.0")
        )
    )

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
            setattr(s, key, value)
    _settings = s
    return _settings
