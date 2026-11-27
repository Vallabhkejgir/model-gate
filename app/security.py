"""Cryptographic virtual key issuance, verification, and 7-layer secret redaction."""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
import urllib.parse
from typing import Tuple

from app.config import Settings, get_settings

REDACTED_PLACEHOLDER = "[REDACTED_SECRET]"

# Pre-compiled patterns for common API credential formats and database URIs
_COMMON_SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z\-_]{20,}"),
    re.compile(r"sk-[0-9A-Za-z\-_]{16,}"),
    re.compile(r"(?i)Bearer\s+[0-9A-Za-z\-_\.]{16,}"),
    re.compile(r"vk_live_[0-9a-f]{8}_[0-9A-Za-z\-_]{16,}"),
    re.compile(r"(?i)postgres(?:ql)?://[^\s\"'<>]+"),
    re.compile(r"(?i)mongodb(?:\+srv)?://[^\s\"'<>]+"),
    re.compile(r"mg_snap_v1_[A-Za-z0-9_\-\.=]+"),
]


def generate_virtual_key(settings: Settings | None = None) -> Tuple[str, str, str, str]:
    """Generate a CSPRNG consumer virtual key.

    Format: vk_live_<8-hex-key-id>_<32-char-urlsafe-secret>
    Returns:
        (plaintext_key, key_id, key_prefix, key_hash)
    """
    cfg = settings or get_settings()
    key_id = secrets.token_hex(4)  # 8 hex characters
    raw_secret = secrets.token_urlsafe(24)[:32]  # 32 URL-safe chars
    key_prefix = f"vk_live_{key_id}"
    plaintext_key = f"{key_prefix}_{raw_secret}"
    key_hash = hash_virtual_key(plaintext_key, cfg.key_hash_pepper)
    return plaintext_key, key_id, key_prefix, key_hash


def hash_virtual_key(plaintext_key: str, pepper: str | None = None) -> str:
    """Compute HMAC-SHA256 digest of a full virtual key using the server pepper."""
    secret_pepper = pepper if pepper is not None else get_settings().key_hash_pepper
    return hmac.new(
        secret_pepper.encode("utf-8"),
        plaintext_key.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_virtual_key_hash(
    plaintext_key: str, stored_hash: str, pepper: str | None = None
) -> bool:
    """Constant-time verification of a plaintext virtual key against its HMAC hash."""
    computed = hash_virtual_key(plaintext_key, pepper)
    return hmac.compare_digest(computed, stored_hash)


def verify_admin_key(candidate_key: str, settings: Settings | None = None) -> bool:
    """Constant-time verification of the operator admin master key."""
    cfg = settings or get_settings()
    if not cfg.admin_master_key or not candidate_key:
        return False
    return hmac.compare_digest(
        candidate_key.encode("utf-8"),
        cfg.admin_master_key.encode("utf-8"),
    )


def create_admin_session_token(settings: Settings | None = None) -> str:
    """Compute an HMAC-SHA256 session cookie token bound to ADMIN_MASTER_KEY and KEY_HASH_PEPPER."""
    cfg = settings or get_settings()
    return hmac.new(
        cfg.key_hash_pepper.encode("utf-8"),
        f"admin_session:{cfg.admin_master_key}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_admin_session_token(
    token: str, settings: Settings | None = None
) -> bool:
    """Constant-time verification of an operator admin session cookie token."""
    cfg = settings or get_settings()
    if not token or not cfg.admin_master_key:
        return False
    expected = create_admin_session_token(cfg)
    return hmac.compare_digest(token.encode("utf-8"), expected.encode("utf-8"))


def parse_virtual_key_id(plaintext_key: str) -> str | None:
    """Extract the 8-hex key_id from a vk_live_<key_id>_<secret> token."""
    if not plaintext_key or not plaintext_key.startswith("vk_live_"):
        return None
    parts = plaintext_key.split("_", 3)
    if len(parts) != 4:
        return None
    _, _, key_id, secret_part = parts
    if len(key_id) != 8 or not secret_part:
        return None
    try:
        int(key_id, 16)
    except ValueError:
        return None
    return key_id


def redact_secrets(text: str | None, settings: Settings | None = None) -> str:
    """Scrub any occurrence of upstream API keys, master keys, peppers, DB URLs, or bearer tokens."""
    if not text:
        return ""
    cfg = settings or get_settings()
    scrubbed = str(text)

    raw_candidates = [
        cfg.gemini_api_key,
        cfg.admin_master_key,
        cfg.key_hash_pepper,
        cfg.database_url,
    ]
    if cfg.database_url:
        if cfg.database_url.startswith("postgresql://"):
            raw_candidates.append(
                "postgres://" + cfg.database_url[len("postgresql://") :]
            )
        try:
            parsed_db = urllib.parse.urlsplit(cfg.database_url)
            if parsed_db.password:
                raw_candidates.append(parsed_db.password)
                unquoted_pw = urllib.parse.unquote(parsed_db.password)
                if unquoted_pw != parsed_db.password:
                    raw_candidates.append(unquoted_pw)
        except Exception:
            pass

    sensitive_literals = []
    for raw_secret in raw_candidates:
        if raw_secret and len(raw_secret) >= 4:
            sensitive_literals.append(raw_secret)
            url_enc = urllib.parse.quote(raw_secret, safe="")
            if url_enc != raw_secret:
                sensitive_literals.append(url_enc)
            b64_enc = base64.b64encode(raw_secret.encode("utf-8")).decode("ascii")
            sensitive_literals.append(b64_enc)
            b64_stripped = b64_enc.rstrip("=")
            if len(b64_stripped) >= 6:
                sensitive_literals.append(b64_stripped)

    for literal in sensitive_literals:
        if literal and literal in scrubbed:
            scrubbed = scrubbed.replace(literal, REDACTED_PLACEHOLDER)

    for pattern in _COMMON_SECRET_PATTERNS:
        scrubbed = pattern.sub(REDACTED_PLACEHOLDER, scrubbed)

    return scrubbed
