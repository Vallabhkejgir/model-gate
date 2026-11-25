"""SQLite persistence layer (WAL mode) for virtual consumer keys and usage audit logs."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

from app.config import get_settings
from app.security import (
    generate_virtual_key,
    parse_virtual_key_id,
    verify_virtual_key_hash,
)


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_db_connection(db_path: str | None = None) -> Generator[sqlite3.Connection, None, None]:
    """Open a SQLite connection configured with WAL mode and Row factory."""
    resolved_path = db_path or get_settings().database_path
    conn = sqlite3.connect(resolved_path, timeout=15.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str | None = None) -> None:
    """Initialize required database tables and indexes."""
    with get_db_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS virtual_keys (
                key_id TEXT PRIMARY KEY,
                key_prefix TEXT NOT NULL,
                key_hash TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                allowed_models TEXT NOT NULL,
                forced_system_prompt TEXT,
                max_tokens_per_request INTEGER NOT NULL DEFAULT 1024,
                max_input_chars INTEGER NOT NULL DEFAULT 16000,
                rpm_limit INTEGER NOT NULL DEFAULT 15,
                token_budget INTEGER,
                tokens_used INTEGER NOT NULL DEFAULT 0,
                prompt_tokens_used INTEGER NOT NULL DEFAULT 0,
                completion_tokens_used INTEGER NOT NULL DEFAULT 0,
                requests_count INTEGER NOT NULL DEFAULT 0,
                is_active INTEGER NOT NULL DEFAULT 1,
                expires_at TEXT,
                created_at TEXT NOT NULL,
                last_used_at TEXT
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_virtual_keys_key_id ON virtual_keys(key_id);"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                key_id TEXT NOT NULL,
                requested_model TEXT NOT NULL,
                upstream_model TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                status_code INTEGER NOT NULL,
                latency_ms REAL NOT NULL DEFAULT 0.0,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_usage_logs_key_id ON usage_logs(key_id);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_usage_logs_created_at ON usage_logs(created_at DESC);"
        )


def _row_to_key_dict(row: sqlite3.Row) -> Dict[str, Any]:
    data = dict(row)
    try:
        data["allowed_models"] = json.loads(data["allowed_models"])
    except (TypeError, json.JSONDecodeError):
        data["allowed_models"] = []
    data["is_active"] = bool(data["is_active"])
    # Never expose key_hash in API responses
    data.pop("key_hash", None)
    return data


def create_virtual_key_record(
    name: str,
    allowed_models: List[str] | None = None,
    forced_system_prompt: Optional[str] = None,
    max_tokens_per_request: int = 1024,
    max_input_chars: int = 16000,
    rpm_limit: int = 15,
    token_budget: Optional[int] = None,
    expires_at: Optional[str] = None,
    db_path: str | None = None,
) -> Dict[str, Any]:
    """Create and persist a new consumer virtual key; returns record with one-time plaintext key."""
    settings = get_settings()
    models = allowed_models or [settings.default_limited_model]
    plaintext_key, key_id, key_prefix, key_hash = generate_virtual_key(settings)
    now = utc_now_iso()
    normalized_budget = token_budget if (token_budget and token_budget > 0) else None

    with get_db_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO virtual_keys (
                key_id, key_prefix, key_hash, name, allowed_models,
                forced_system_prompt, max_tokens_per_request, max_input_chars,
                rpm_limit, token_budget, tokens_used, prompt_tokens_used,
                completion_tokens_used, requests_count, is_active,
                expires_at, created_at, last_used_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 0, 1, ?, ?, NULL)
            """,
            (
                key_id,
                key_prefix,
                key_hash,
                name,
                json.dumps(models),
                forced_system_prompt,
                int(max_tokens_per_request),
                int(max_input_chars),
                int(rpm_limit),
                normalized_budget,
                expires_at,
                now,
            ),
        )
        row = conn.execute(
            "SELECT * FROM virtual_keys WHERE key_id = ?", (key_id,)
        ).fetchone()

    record = _row_to_key_dict(row)
    record["api_key"] = plaintext_key
    return record


def list_virtual_keys(db_path: str | None = None) -> List[Dict[str, Any]]:
    """List all consumer virtual keys without exposing key hashes."""
    with get_db_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM virtual_keys ORDER BY created_at DESC"
        ).fetchall()
    return [_row_to_key_dict(r) for r in rows]


def get_virtual_key_by_id(
    key_id: str, db_path: str | None = None
) -> Optional[Dict[str, Any]]:
    """Fetch a single consumer key record by its 8-char hex key_id."""
    with get_db_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM virtual_keys WHERE key_id = ?", (key_id,)
        ).fetchone()
    if not row:
        return None
    return _row_to_key_dict(row)


def authenticate_virtual_key(
    plaintext_key: str, db_path: str | None = None
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Verify a plaintext virtual key and check active status and expiration.

    Returns:
        (key_record_dict, error_reason_or_none)
    """
    key_id = parse_virtual_key_id(plaintext_key)
    if not key_id:
        return None, "Invalid virtual API key format."

    with get_db_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM virtual_keys WHERE key_id = ?", (key_id,)
        ).fetchone()

    if not row:
        return None, "Invalid or unknown virtual API key."

    stored_hash = row["key_hash"]
    if not verify_virtual_key_hash(plaintext_key, stored_hash):
        return None, "Invalid virtual API key signature."

    record = _row_to_key_dict(row)
    if not record["is_active"]:
        return None, "Virtual API key has been revoked."

    expires_at = record.get("expires_at")
    if expires_at:
        try:
            exp_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if exp_dt.tzinfo is None:
                exp_dt = exp_dt.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) >= exp_dt:
                return None, "Virtual API key has expired."
        except ValueError:
            return None, "Virtual API key has an invalid expiration timestamp."

    return record, None


def update_virtual_key_policy(
    key_id: str,
    updates: Dict[str, Any],
    db_path: str | None = None,
) -> Optional[Dict[str, Any]]:
    """Update policy fields or reset usage on an existing consumer virtual key."""
    with get_db_connection(db_path) as conn:
        existing = conn.execute(
            "SELECT * FROM virtual_keys WHERE key_id = ?", (key_id,)
        ).fetchone()
        if not existing:
            return None

        clauses: List[str] = []
        params: List[Any] = []

        if "name" in updates and updates["name"] is not None:
            clauses.append("name = ?")
            params.append(str(updates["name"]))

        if "allowed_models" in updates and updates["allowed_models"] is not None:
            clauses.append("allowed_models = ?")
            params.append(json.dumps(list(updates["allowed_models"])))

        if "forced_system_prompt" in updates:
            clauses.append("forced_system_prompt = ?")
            params.append(updates["forced_system_prompt"])

        if (
            "max_tokens_per_request" in updates
            and updates["max_tokens_per_request"] is not None
        ):
            clauses.append("max_tokens_per_request = ?")
            params.append(int(updates["max_tokens_per_request"]))

        if "max_input_chars" in updates and updates["max_input_chars"] is not None:
            clauses.append("max_input_chars = ?")
            params.append(int(updates["max_input_chars"]))

        if "rpm_limit" in updates and updates["rpm_limit"] is not None:
            clauses.append("rpm_limit = ?")
            params.append(int(updates["rpm_limit"]))

        if "token_budget" in updates:
            budget = updates["token_budget"]
            normalized = int(budget) if (budget is not None and int(budget) > 0) else None
            clauses.append("token_budget = ?")
            params.append(normalized)

        if "is_active" in updates and updates["is_active"] is not None:
            clauses.append("is_active = ?")
            params.append(1 if updates["is_active"] else 0)

        if "expires_at" in updates:
            clauses.append("expires_at = ?")
            params.append(updates["expires_at"])

        if updates.get("reset_usage"):
            clauses.extend(
                [
                    "tokens_used = 0",
                    "prompt_tokens_used = 0",
                    "completion_tokens_used = 0",
                    "requests_count = 0",
                ]
            )

        if clauses:
            params.append(key_id)
            sql = f"UPDATE virtual_keys SET {', '.join(clauses)} WHERE key_id = ?"
            conn.execute(sql, tuple(params))

        updated_row = conn.execute(
            "SELECT * FROM virtual_keys WHERE key_id = ?", (key_id,)
        ).fetchone()
    return _row_to_key_dict(updated_row)


def revoke_virtual_key(
    key_id: str, hard_delete: bool = False, db_path: str | None = None
) -> bool:
    """Revoke (deactivate) or permanently delete a consumer virtual key."""
    with get_db_connection(db_path) as conn:
        row = conn.execute(
            "SELECT key_id FROM virtual_keys WHERE key_id = ?", (key_id,)
        ).fetchone()
        if not row:
            return False
        if hard_delete:
            conn.execute("DELETE FROM virtual_keys WHERE key_id = ?", (key_id,))
        else:
            conn.execute(
                "UPDATE virtual_keys SET is_active = 0 WHERE key_id = ?", (key_id,)
            )
    return True


def record_request_usage(
    request_id: str,
    key_id: str,
    requested_model: str,
    upstream_model: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    status_code: int,
    latency_ms: float,
    db_path: str | None = None,
) -> None:
    """Atomically increment key token counters and append a usage_logs audit entry."""
    now = utc_now_iso()
    p_tok = max(0, int(prompt_tokens))
    c_tok = max(0, int(completion_tokens))
    t_tok = max(0, int(total_tokens)) or (p_tok + c_tok)

    with get_db_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE virtual_keys
            SET tokens_used = tokens_used + ?,
                prompt_tokens_used = prompt_tokens_used + ?,
                completion_tokens_used = completion_tokens_used + ?,
                requests_count = requests_count + 1,
                last_used_at = ?
            WHERE key_id = ?
            """,
            (t_tok, p_tok, c_tok, now, key_id),
        )
        conn.execute(
            """
            INSERT INTO usage_logs (
                request_id, key_id, requested_model, upstream_model,
                prompt_tokens, completion_tokens, total_tokens,
                status_code, latency_ms, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                key_id,
                requested_model,
                upstream_model,
                p_tok,
                c_tok,
                t_tok,
                int(status_code),
                round(float(latency_ms), 2),
                now,
            ),
        )


def get_usage_summary(
    limit: int = 100, key_id: Optional[str] = None, db_path: str | None = None
) -> Dict[str, Any]:
    """Return aggregated usage statistics and recent audit log entries."""
    with get_db_connection(db_path) as conn:
        if key_id:
            logs_rows = conn.execute(
                """
                SELECT * FROM usage_logs
                WHERE key_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (key_id, int(limit)),
            ).fetchall()
            agg_row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_requests,
                    COALESCE(SUM(prompt_tokens), 0) AS total_prompt_tokens,
                    COALESCE(SUM(completion_tokens), 0) AS total_completion_tokens,
                    COALESCE(SUM(total_tokens), 0) AS total_tokens
                FROM usage_logs
                WHERE key_id = ?
                """,
                (key_id,),
            ).fetchone()
        else:
            logs_rows = conn.execute(
                "SELECT * FROM usage_logs ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
            agg_row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_requests,
                    COALESCE(SUM(prompt_tokens), 0) AS total_prompt_tokens,
                    COALESCE(SUM(completion_tokens), 0) AS total_completion_tokens,
                    COALESCE(SUM(total_tokens), 0) AS total_tokens
                FROM usage_logs
                """
            ).fetchone()

    return {
        "summary": {
            "total_requests": int(agg_row["total_requests"] or 0),
            "total_prompt_tokens": int(agg_row["total_prompt_tokens"] or 0),
            "total_completion_tokens": int(agg_row["total_completion_tokens"] or 0),
            "total_tokens": int(agg_row["total_tokens"] or 0),
        },
        "logs": [dict(r) for r in logs_rows],
    }
