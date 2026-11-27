"""Dual-engine persistence layer (Cloud PostgreSQL + Local SQLite WAL fallback) with sub-millisecond write-through key cache."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from app.config import _normalize_database_url, get_settings
from app.security import (
    generate_virtual_key,
    parse_virtual_key_id,
    verify_virtual_key_hash,
)

try:
    import psycopg
    import psycopg.rows
    import psycopg_pool

    _HAS_PSYCOPG = True
except ImportError:  # pragma: no cover
    psycopg = None  # type: ignore[assignment]
    psycopg_pool = None  # type: ignore[assignment]
    _HAS_PSYCOPG = False


# ---------------------------------------------------------------------------
# Sub-Millisecond Write-Through In-Memory Key Cache
# ---------------------------------------------------------------------------
_KEY_CACHE: Dict[Tuple[str, str], Tuple[float, Dict[str, Any]]] = {}
_KEY_CACHE_LOCK = threading.Lock()
_CACHE_HITS = 0
_CACHE_MISSES = 0

# Counter for periodic log retention pruning
_USAGE_WRITE_COUNTER = 0
_USAGE_WRITE_LOCK = threading.Lock()

# PostgreSQL connection pool state
_pg_pool: Any = None
_pg_pool_url: str = ""
_pg_pool_lock = threading.Lock()
_custom_pg_connection_factory: Optional[Callable[[str], Any]] = None


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def set_pg_connection_factory(
    factory: Optional[Callable[[str], Any]]
) -> None:
    """Register or clear a custom PostgreSQL connection factory (used in unit tests)."""
    global _custom_pg_connection_factory
    _custom_pg_connection_factory = factory


def _use_postgres(db_path: str | None = None) -> bool:
    """Determine whether to use PostgreSQL or SQLite for this operation."""
    if db_path:
        lower = db_path.strip().lower()
        return lower.startswith("postgresql://") or lower.startswith("postgres://")
    return get_settings().storage_backend() == "postgresql"


def _resolve_postgres_url(db_path: str | None = None) -> str:
    """Return normalized postgresql:// connection URI."""
    if db_path:
        return _normalize_database_url(db_path)
    return get_settings().database_url


def _cache_scope(db_path: str | None = None) -> str:
    """Compute an isolated cache namespace key for the active database."""
    if db_path:
        return _normalize_database_url(db_path) if _use_postgres(db_path) else str(db_path)
    cfg = get_settings()
    return cfg.database_url if cfg.storage_backend() == "postgresql" else cfg.database_path


def clear_key_cache() -> None:
    """Clear all entries from the in-memory virtual key cache."""
    global _CACHE_HITS, _CACHE_MISSES
    with _KEY_CACHE_LOCK:
        _KEY_CACHE.clear()
        _CACHE_HITS = 0
        _CACHE_MISSES = 0


def get_key_cache_stats() -> Dict[str, int]:
    """Return diagnostic cache statistics (size, hits, misses)."""
    with _KEY_CACHE_LOCK:
        return {
            "entries": len(_KEY_CACHE),
            "hits": _CACHE_HITS,
            "misses": _CACHE_MISSES,
        }


def _cache_put(scope: str, row: Any) -> None:
    """Store or refresh a virtual_keys row in the in-memory cache."""
    if not row:
        return
    ttl = get_settings().key_cache_ttl_seconds
    if ttl <= 0:
        return
    row_dict = dict(row)
    key_id = row_dict.get("key_id")
    if not key_id:
        return
    with _KEY_CACHE_LOCK:
        _KEY_CACHE[(scope, str(key_id))] = (time.monotonic(), row_dict)


def _cache_get(scope: str, key_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a cached virtual_keys row if present and within TTL."""
    global _CACHE_HITS, _CACHE_MISSES
    ttl = get_settings().key_cache_ttl_seconds
    if ttl <= 0:
        with _KEY_CACHE_LOCK:
            _CACHE_MISSES += 1
        return None
    now_mono = time.monotonic()
    with _KEY_CACHE_LOCK:
        entry = _KEY_CACHE.get((scope, key_id))
        if entry is None:
            _CACHE_MISSES += 1
            return None
        cached_at, row_dict = entry
        if (now_mono - cached_at) > ttl:
            _KEY_CACHE.pop((scope, key_id), None)
            _CACHE_MISSES += 1
            return None
        _CACHE_HITS += 1
        return dict(row_dict)


def _cache_evict(scope: str, key_id: str) -> None:
    """Immediately remove a virtual key from the in-memory cache."""
    with _KEY_CACHE_LOCK:
        _KEY_CACHE.pop((scope, key_id), None)


def _cache_increment_usage(
    scope: str,
    key_id: str,
    total_tokens: int,
    prompt_tokens: int,
    completion_tokens: int,
    used_at: str,
) -> None:
    """Write-through increment of token and request counters on a cached virtual key."""
    with _KEY_CACHE_LOCK:
        entry = _KEY_CACHE.get((scope, key_id))
        if entry is None:
            return
        cached_at, row_dict = entry
        updated = dict(row_dict)
        updated["tokens_used"] = int(updated.get("tokens_used") or 0) + total_tokens
        updated["prompt_tokens_used"] = (
            int(updated.get("prompt_tokens_used") or 0) + prompt_tokens
        )
        updated["completion_tokens_used"] = (
            int(updated.get("completion_tokens_used") or 0) + completion_tokens
        )
        updated["requests_count"] = int(updated.get("requests_count") or 0) + 1
        updated["last_used_at"] = used_at
        _KEY_CACHE[(scope, key_id)] = (cached_at, updated)


def _adapt_sql(sql: str, is_pg: bool) -> str:
    """Translate SQLite '?' parameter placeholders to PostgreSQL '%s' placeholders."""
    if not is_pg:
        return sql
    return sql.replace("?", "%s")


class _PgConnectionAdapter:
    """Unified connection adapter wrapping a PostgreSQL connection with '?' -> '%s' placeholder translation."""

    is_postgres: bool = True

    def __init__(self, raw_conn: Any) -> None:
        self._raw_conn = raw_conn

    def execute(
        self, sql: str, params: Tuple[Any, ...] | List[Any] | None = None
    ) -> Any:
        pg_sql = _adapt_sql(sql, is_pg=True)
        if hasattr(self._raw_conn, "execute"):
            if params is not None:
                return self._raw_conn.execute(pg_sql, params)
            return self._raw_conn.execute(pg_sql)
        cur = self._raw_conn.cursor()
        if params is not None:
            cur.execute(pg_sql, params)
        else:
            cur.execute(pg_sql)
        return cur

    def commit(self) -> None:
        if hasattr(self._raw_conn, "commit"):
            self._raw_conn.commit()

    def rollback(self) -> None:
        if hasattr(self._raw_conn, "rollback"):
            self._raw_conn.rollback()


def _get_or_create_pg_pool(db_url: str) -> Any:
    """Return or initialize the shared psycopg_pool.ConnectionPool for db_url."""
    global _pg_pool, _pg_pool_url
    if not _HAS_PSYCOPG:  # pragma: no cover
        raise RuntimeError(
            "psycopg[binary,pool] is required when DATABASE_URL is set to PostgreSQL."
        )
    with _pg_pool_lock:
        if _pg_pool is not None and _pg_pool_url == db_url:
            return _pg_pool
        if _pg_pool is not None:
            try:
                _pg_pool.close()
            except Exception:
                pass
            _pg_pool = None
        _pg_pool = psycopg_pool.ConnectionPool(
            conninfo=db_url,
            min_size=1,
            max_size=10,
            open=True,
            kwargs={
                "row_factory": psycopg.rows.dict_row,
                "autocommit": False,
                "connect_timeout": 10,
            },
        )
        _pg_pool_url = db_url
        return _pg_pool


def close_db_pool() -> None:
    """Close the active PostgreSQL connection pool on application shutdown."""
    global _pg_pool, _pg_pool_url
    with _pg_pool_lock:
        if _pg_pool is not None:
            try:
                _pg_pool.close()
            except Exception:
                pass
            _pg_pool = None
            _pg_pool_url = ""


@contextmanager
def get_db_connection(
    db_path: str | None = None,
) -> Generator[Any, None, None]:
    """Open a pooled PostgreSQL connection (when DATABASE_URL is configured) or a local SQLite WAL connection."""
    if _use_postgres(db_path):
        db_url = _resolve_postgres_url(db_path)
        if _custom_pg_connection_factory is not None:
            raw_conn = _custom_pg_connection_factory(db_url)
            adapter = _PgConnectionAdapter(raw_conn)
            try:
                yield adapter
                adapter.commit()
            except Exception:
                adapter.rollback()
                raise
            finally:
                if hasattr(raw_conn, "close"):
                    raw_conn.close()
            return

        pool = _get_or_create_pg_pool(db_url)
        with pool.connection() as raw_conn:
            adapter = _PgConnectionAdapter(raw_conn)
            try:
                yield adapter
                adapter.commit()
            except Exception:
                adapter.rollback()
                raise
        return

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
    """Initialize required database tables and indexes on PostgreSQL or SQLite, and seed bootstrap records."""
    clear_key_cache()
    is_pg = _use_postgres(db_path)
    usage_id_col = (
        "id BIGSERIAL PRIMARY KEY"
        if is_pg
        else "id INTEGER PRIMARY KEY AUTOINCREMENT"
    )

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
            f"""
            CREATE TABLE IF NOT EXISTS usage_logs (
                {usage_id_col},
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

    cfg = get_settings()
    if cfg.bootstrap_key_records and cfg.bootstrap_key_records.strip():
        seed_bootstrap_key_records(cfg.bootstrap_key_records, db_path=db_path)


def _row_to_key_dict(row: Any) -> Dict[str, Any]:
    data = dict(row)
    raw_models = data.get("allowed_models")
    if isinstance(raw_models, list):
        data["allowed_models"] = raw_models
    else:
        try:
            data["allowed_models"] = json.loads(raw_models or "[]")
        except (TypeError, json.JSONDecodeError):
            data["allowed_models"] = []
    data["is_active"] = bool(data.get("is_active"))
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

    _cache_put(_cache_scope(db_path), row)
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
    _cache_put(_cache_scope(db_path), row)
    return _row_to_key_dict(row)


def authenticate_virtual_key(
    plaintext_key: str, db_path: str | None = None
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Verify a plaintext virtual key using the sub-millisecond write-through cache or database.

    Returns:
        (key_record_dict, error_reason_or_none)
    """
    key_id = parse_virtual_key_id(plaintext_key)
    if not key_id:
        return None, "Invalid virtual API key format."

    scope = _cache_scope(db_path)
    row: Any = _cache_get(scope, key_id)
    if row is None:
        with get_db_connection(db_path) as conn:
            row = conn.execute(
                "SELECT * FROM virtual_keys WHERE key_id = ?", (key_id,)
            ).fetchone()
        if row:
            _cache_put(scope, row)

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
    """Update policy fields or reset usage on an existing consumer virtual key and refresh cache."""
    scope = _cache_scope(db_path)
    with get_db_connection(db_path) as conn:
        existing = conn.execute(
            "SELECT * FROM virtual_keys WHERE key_id = ?", (key_id,)
        ).fetchone()
        if not existing:
            _cache_evict(scope, key_id)
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

    _cache_put(scope, updated_row)
    return _row_to_key_dict(updated_row)


def revoke_virtual_key(
    key_id: str, hard_delete: bool = False, db_path: str | None = None
) -> bool:
    """Revoke (deactivate) or permanently delete a consumer virtual key and evict from cache immediately."""
    scope = _cache_scope(db_path)
    with get_db_connection(db_path) as conn:
        row = conn.execute(
            "SELECT key_id FROM virtual_keys WHERE key_id = ?", (key_id,)
        ).fetchone()
        if not row:
            _cache_evict(scope, key_id)
            return False
        if hard_delete:
            conn.execute("DELETE FROM virtual_keys WHERE key_id = ?", (key_id,))
        else:
            conn.execute(
                "UPDATE virtual_keys SET is_active = 0 WHERE key_id = ?", (key_id,)
            )
    _cache_evict(scope, key_id)
    return True


def _prune_usage_logs_in_conn(conn: Any, max_rows: int) -> None:
    """Delete oldest usage_logs entries exceeding max_rows."""
    if max_rows <= 0:
        return
    conn.execute(
        """
        DELETE FROM usage_logs
        WHERE id NOT IN (
            SELECT id FROM usage_logs ORDER BY id DESC LIMIT ?
        )
        """,
        (int(max_rows),),
    )


def prune_usage_logs(
    max_rows: Optional[int] = None, db_path: str | None = None
) -> None:
    """Explicitly prune usage_logs beyond max_rows (or settings.max_usage_log_rows)."""
    limit = max_rows if max_rows is not None else get_settings().max_usage_log_rows
    if limit <= 0:
        return
    with get_db_connection(db_path) as conn:
        _prune_usage_logs_in_conn(conn, int(limit))


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
    """Atomically increment key token counters, update write-through cache, append metadata-only usage_logs, and enforce retention cap."""
    global _USAGE_WRITE_COUNTER
    now = utc_now_iso()
    p_tok = max(0, int(prompt_tokens))
    c_tok = max(0, int(completion_tokens))
    t_tok = max(0, int(total_tokens)) or (p_tok + c_tok)
    scope = _cache_scope(db_path)
    max_rows = get_settings().max_usage_log_rows

    with _USAGE_WRITE_LOCK:
        _USAGE_WRITE_COUNTER += 1
        write_num = _USAGE_WRITE_COUNTER

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
        if max_rows > 0 and (max_rows <= 500 or write_num % 25 == 0):
            _prune_usage_logs_in_conn(conn, max_rows)

    _cache_increment_usage(
        scope=scope,
        key_id=key_id,
        total_tokens=t_tok,
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        used_at=now,
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


def list_bootstrap_key_records(db_path: str | None = None) -> List[Dict[str, Any]]:
    """Export all virtual key records with their HMAC key_hash and policy fields (zero plaintext secrets)."""
    with get_db_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM virtual_keys ORDER BY created_at ASC"
        ).fetchall()

    exported: List[Dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        raw_models = d.get("allowed_models")
        if isinstance(raw_models, list):
            models = raw_models
        else:
            try:
                models = json.loads(raw_models or "[]")
            except (TypeError, json.JSONDecodeError):
                models = []
        exported.append(
            {
                "key_id": d["key_id"],
                "key_prefix": d["key_prefix"],
                "key_hash": d["key_hash"],
                "name": d["name"],
                "allowed_models": models,
                "forced_system_prompt": d.get("forced_system_prompt"),
                "max_tokens_per_request": int(d.get("max_tokens_per_request") or 1024),
                "max_input_chars": int(d.get("max_input_chars") or 16000),
                "rpm_limit": int(d.get("rpm_limit") or 15),
                "token_budget": d.get("token_budget"),
                "tokens_used": int(d.get("tokens_used") or 0),
                "prompt_tokens_used": int(d.get("prompt_tokens_used") or 0),
                "completion_tokens_used": int(d.get("completion_tokens_used") or 0),
                "requests_count": int(d.get("requests_count") or 0),
                "is_active": bool(d.get("is_active", 1)),
                "expires_at": d.get("expires_at"),
                "created_at": d.get("created_at") or utc_now_iso(),
                "last_used_at": d.get("last_used_at"),
            }
        )
    return exported


def export_bootstrap_key_records(db_path: str | None = None) -> str:
    """Return a compact JSON string of pre-hashed virtual key records suitable for BOOTSTRAP_KEY_RECORDS."""
    records = list_bootstrap_key_records(db_path=db_path)
    return json.dumps(records, separators=(",", ":"))


def seed_bootstrap_key_records(
    raw_records: str | List[Dict[str, Any]] | None = None,
    db_path: str | None = None,
) -> int:
    """Idempotently seed pre-hashed virtual key records from BOOTSTRAP_KEY_RECORDS JSON."""
    source = (
        raw_records
        if raw_records is not None
        else get_settings().bootstrap_key_records
    )
    if not source:
        return 0

    if isinstance(source, str):
        cleaned = source.strip()
        if not cleaned:
            return 0
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return 0
    else:
        parsed = source

    if isinstance(parsed, dict) and isinstance(parsed.get("records"), list):
        items = parsed["records"]
    elif isinstance(parsed, list):
        items = parsed
    else:
        return 0

    inserted = 0
    now = utc_now_iso()
    with get_db_connection(db_path) as conn:
        for item in items:
            if not isinstance(item, dict):
                continue
            key_id = str(item.get("key_id") or "").strip()
            key_prefix = str(item.get("key_prefix") or "").strip()
            key_hash = str(item.get("key_hash") or "").strip()
            name = str(item.get("name") or "").strip()
            if not (key_id and key_prefix and key_hash and name):
                continue

            allowed_models = item.get("allowed_models") or [
                get_settings().default_limited_model
            ]
            if isinstance(allowed_models, list):
                allowed_models_json = json.dumps(allowed_models)
            else:
                allowed_models_json = str(allowed_models)

            conn.execute(
                """
                INSERT INTO virtual_keys (
                    key_id, key_prefix, key_hash, name, allowed_models,
                    forced_system_prompt, max_tokens_per_request, max_input_chars,
                    rpm_limit, token_budget, tokens_used, prompt_tokens_used,
                    completion_tokens_used, requests_count, is_active,
                    expires_at, created_at, last_used_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (key_id) DO NOTHING
                """,
                (
                    key_id,
                    key_prefix,
                    key_hash,
                    name,
                    allowed_models_json,
                    item.get("forced_system_prompt"),
                    int(item.get("max_tokens_per_request") or 1024),
                    int(item.get("max_input_chars") or 16000),
                    int(item.get("rpm_limit") or 15),
                    item.get("token_budget"),
                    int(item.get("tokens_used") or 0),
                    int(item.get("prompt_tokens_used") or 0),
                    int(item.get("completion_tokens_used") or 0),
                    int(item.get("requests_count") or 0),
                    1 if item.get("is_active", True) else 0,
                    item.get("expires_at"),
                    item.get("created_at") or now,
                    item.get("last_used_at"),
                ),
            )
            inserted += 1

    clear_key_cache()
    return inserted
