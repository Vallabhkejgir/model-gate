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

# Pre-compiled patterns for common API credential formats
_COMMON_SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z\-_]{20,}"),
    re.compile(r"sk-[0-9A-Za-z\-_]{16,}"),
    re.compile(r"(?i)Bearer\s+[0-9A-Za-z\-_\.]{16,}"),
    re.compile(r"vk_live_[0-9a-f]{8}_[0-9A-Za-z\-_]{16,}"),
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
    """Scrub any occurrence of upstream API keys, master keys, peppers, or bearer tokens."""
    if not text:
        return ""
    cfg = settings or get_settings()
    scrubbed = str(text)

    sensitive_literals = []
    for raw_secret in (
        cfg.gemini_api_key,
        cfg.admin_master_key,
        cfg.key_hash_pepper,
    ):
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
