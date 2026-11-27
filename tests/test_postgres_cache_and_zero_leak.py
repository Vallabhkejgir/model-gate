"""Tests for dual-engine PostgreSQL/SQLite persistence, sub-millisecond write-through key cache, zero-leak DB URL redaction, bootstrap export/restore, and log retention pruning."""

from __future__ import annotations

import base64
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, List, Tuple

from fastapi.testclient import TestClient

from app.config import Settings, reload_settings
from app.database import (
    _adapt_sql,
    authenticate_virtual_key,
    clear_key_cache,
    create_virtual_key_record,
    export_bootstrap_key_records,
    get_key_cache_stats,
    get_usage_summary,
    get_virtual_key_by_id,
    init_db,
    list_virtual_keys,
    record_request_usage,
    revoke_virtual_key,
    seed_bootstrap_key_records,
    set_pg_connection_factory,
    update_virtual_key_policy,
)
from app.security import redact_secrets
from tests.conftest import DUMMY_ADMIN_KEY, DUMMY_GEMINI_KEY, DUMMY_PEPPER


class FakePsycopgCursor:
    """Simulates a psycopg v3 dict_row cursor executing Postgres-adapted (%s) SQL."""

    def __init__(self, rows: List[dict]):
        self._rows = rows

    def fetchone(self) -> dict | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> List[dict]:
        return list(self._rows)


class FakePsycopgConnection:
    """Simulates a psycopg v3 Connection that enforces %s placeholders and BIGSERIAL DDL while executing against an isolated backing store."""

    def __init__(
        self,
        backing_sqlite_path: str,
        executed_sql_log: List[Tuple[str, Any]],
    ) -> None:
        self._sqlite = sqlite3.connect(backing_sqlite_path, check_same_thread=False)
        self._sqlite.row_factory = sqlite3.Row
        self._log = executed_sql_log
        self.closed = False

    def execute(self, pg_sql: str, params: Any = None) -> FakePsycopgCursor:
        self._log.append((pg_sql, params))
        # Assert that _PgConnectionAdapter translated all '?' placeholders to '%s'
        assert "?" not in pg_sql, f"Untranslated '?' placeholder found in Postgres SQL: {pg_sql}"

        # Translate Postgres DDL/placeholders back to SQLite for in-memory execution verification
        sqlite_sql = pg_sql.replace(
            "id BIGSERIAL PRIMARY KEY", "id INTEGER PRIMARY KEY AUTOINCREMENT"
        ).replace("%s", "?")

        if params is not None:
            cur = self._sqlite.execute(sqlite_sql, tuple(params))
        else:
            cur = self._sqlite.execute(sqlite_sql)
        fetched = [dict(r) for r in cur.fetchall()]
        return FakePsycopgCursor(fetched)

    def commit(self) -> None:
        self._sqlite.commit()

    def rollback(self) -> None:
        self._sqlite.rollback()

    def close(self) -> None:
        self.closed = True
        self._sqlite.close()


def test_postgres_url_normalization_and_storage_backend_detection():
    """Verify Settings normalizes postgres:// to postgresql:// and reports storage_backend accurately."""
    s_sqlite = Settings(database_url="", database_path="local.db")
    assert s_sqlite.storage_backend() == "sqlite"

    s_pg_short = Settings(
        database_url="postgres://neon_user:secret123@ep-test.aws.neon.tech/neondb?sslmode=require"
    )
    assert (
        s_pg_short.database_url
        == "postgresql://neon_user:secret123@ep-test.aws.neon.tech/neondb?sslmode=require"
    )
    assert s_pg_short.storage_backend() == "postgresql"

    s_pg_full = reload_settings(
        database_url="postgresql://supa_user:pass456@db.supabase.co:5432/postgres"
    )
    assert s_pg_full.storage_backend() == "postgresql"
    assert _adapt_sql("SELECT * FROM virtual_keys WHERE key_id = ?", is_pg=True) == (
        "SELECT * FROM virtual_keys WHERE key_id = %s"
    )
    assert _adapt_sql("SELECT * FROM virtual_keys WHERE key_id = ?", is_pg=False) == (
        "SELECT * FROM virtual_keys WHERE key_id = ?"
    )


def test_postgresql_execution_layer_crud_end_to_end(tmp_path: Path, client: TestClient):
    """Verify full PostgreSQL CRUD lifecycle, BIGSERIAL DDL, %s placeholders, and /health + /admin/api/config indicators."""
    pg_backing_file = str(tmp_path / "pg_backing_store.db")
    executed_queries: List[Tuple[str, Any]] = []

    def fake_pg_factory(db_url: str) -> FakePsycopgConnection:
        assert db_url.startswith("postgresql://")
        return FakePsycopgConnection(pg_backing_file, executed_queries)

    set_pg_connection_factory(fake_pg_factory)
    reload_settings(
        gemini_api_key=DUMMY_GEMINI_KEY,
        admin_master_key=DUMMY_ADMIN_KEY,
        key_hash_pepper=DUMMY_PEPPER,
        database_url="postgres://mg_owner:super_secret_pg_pass@ep-mute-hall.us-east-2.aws.neon.tech/modelgate?sslmode=require",
        key_cache_ttl_seconds=30.0,
    )

    # 1. init_db uses BIGSERIAL PRIMARY KEY on PostgreSQL
    init_db()
    assert any("BIGSERIAL PRIMARY KEY" in sql for sql, _ in executed_queries)

    # 2. Check /health and /admin/api/config reflect PostgreSQL cloud storage
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    h_json = health_resp.json()
    assert h_json["storage_backend"] == "postgresql"
    assert h_json["persistent_cloud_db"] is True
    assert h_json["key_cache_ttl_seconds"] == 30.0

    cfg_resp = client.get(
        "/admin/api/config", headers={"X-Admin-Key": DUMMY_ADMIN_KEY}
    )
    assert cfg_resp.status_code == 200
    c_json = cfg_resp.json()
    assert c_json["storage_backend"] == "postgresql"
    assert c_json["persistent_cloud_db"] is True

    # 3. Create virtual key on Postgres backend
    created = create_virtual_key_record(
        name="pg-cloud-consumer",
        allowed_models=["limited-default", "gemini-flash"],
        max_tokens_per_request=512,
        rpm_limit=25,
        token_budget=1000,
    )
    key_id = created["key_id"]
    plaintext_key = created["api_key"]

    # 4. Authenticate key (cache hit on first check after create, then force cache clear to test Postgres SELECT %s)
    clear_key_cache()
    authed, err = authenticate_virtual_key(plaintext_key)
    assert err is None
    assert authed is not None
    assert authed["key_id"] == key_id
    assert authed["name"] == "pg-cloud-consumer"

    # 5. List and get by ID on Postgres
    all_keys = list_virtual_keys()
    assert len(all_keys) == 1
    assert all_keys[0]["key_id"] == key_id
    fetched = get_virtual_key_by_id(key_id)
    assert fetched is not None
    assert fetched["rpm_limit"] == 25

    # 6. Update policy on Postgres
    updated = update_virtual_key_policy(
        key_id, {"rpm_limit": 40, "token_budget": 2500}
    )
    assert updated is not None
    assert updated["rpm_limit"] == 40
    assert updated["token_budget"] == 2500

    # 7. Record usage and fetch summary on Postgres
    record_request_usage(
        request_id="req_pg_1",
        key_id=key_id,
        requested_model="limited-default",
        upstream_model="gemini-2.5-flash",
        prompt_tokens=15,
        completion_tokens=35,
        total_tokens=50,
        status_code=200,
        latency_ms=18.4,
    )
    usage = get_usage_summary(key_id=key_id)
    assert usage["summary"]["total_requests"] == 1
    assert usage["summary"]["total_tokens"] == 50

    # 8. Revoke key on Postgres
    assert revoke_virtual_key(key_id) is True
    authed_after, err_after = authenticate_virtual_key(plaintext_key)
    assert authed_after is None
    assert err_after is not None and "revoked" in err_after.lower()


def test_in_memory_key_cache_zero_db_roundtrip_and_write_through_invalidation(
    isolated_env: str,
):
    """Verify _KEY_CACHE serves hot auth with zero DB queries and immediately updates on usage, policy change, and revocation."""
    clear_key_cache()
    created = create_virtual_key_record(
        name="cache-speed-user",
        allowed_models=["limited-default"],
        token_budget=500,
        db_path=isolated_env,
    )
    key_id = created["key_id"]
    plaintext_key = created["api_key"]

    # After create_virtual_key_record, the key is already pre-warmed in _KEY_CACHE
    stats_before = get_key_cache_stats()
    assert stats_before["entries"] == 1

    # Authenticate 5 times: all 5 should be served from in-memory cache (hits += 5, misses == 0)
    for _ in range(5):
        rec, err = authenticate_virtual_key(plaintext_key, db_path=isolated_env)
        assert err is None
        assert rec is not None
        assert rec["key_id"] == key_id

    stats_after = get_key_cache_stats()
    assert stats_after["hits"] == stats_before["hits"] + 5
    assert stats_after["misses"] == 0

    # Record usage (120 tokens) -> write-through cache must immediately reflect tokens_used == 120 without cache eviction
    record_request_usage(
        request_id="req_cache_1",
        key_id=key_id,
        requested_model="limited-default",
        upstream_model="gemini-2.5-flash",
        prompt_tokens=50,
        completion_tokens=70,
        total_tokens=120,
        status_code=200,
        latency_ms=9.2,
        db_path=isolated_env,
    )
    rec_after_usage, err = authenticate_virtual_key(
        plaintext_key, db_path=isolated_env
    )
    assert err is None
    assert rec_after_usage is not None
    assert rec_after_usage["tokens_used"] == 120
    assert rec_after_usage["requests_count"] == 1

    # Update policy -> write-through cache must immediately reflect new max_tokens_per_request
    update_virtual_key_policy(
        key_id, {"max_tokens_per_request": 4096}, db_path=isolated_env
    )
    rec_after_policy, _ = authenticate_virtual_key(
        plaintext_key, db_path=isolated_env
    )
    assert rec_after_policy is not None
    assert rec_after_policy["max_tokens_per_request"] == 4096

    # Revoke key -> cache must evict immediately so auth fails at 0ms delay
    revoke_virtual_key(key_id, db_path=isolated_env)
    rec_revoked, err_revoked = authenticate_virtual_key(
        plaintext_key, db_path=isolated_env
    )
    assert rec_revoked is None
    assert err_revoked is not None and "revoked" in err_revoked.lower()

    # Re-enable and test TTL expiration
    update_virtual_key_policy(key_id, {"is_active": True}, db_path=isolated_env)
    reload_settings(
        gemini_api_key=DUMMY_GEMINI_KEY,
        admin_master_key=DUMMY_ADMIN_KEY,
        key_hash_pepper=DUMMY_PEPPER,
        database_path=isolated_env,
        key_cache_ttl_seconds=0.02,
    )
    time.sleep(0.04)
    misses_before_expiry = get_key_cache_stats()["misses"]
    rec_reloaded, err_reloaded = authenticate_virtual_key(
        plaintext_key, db_path=isolated_env
    )
    assert err_reloaded is None
    assert rec_reloaded is not None
    assert get_key_cache_stats()["misses"] == misses_before_expiry + 1


def test_zero_leak_database_url_and_password_redaction():
    """Verify DATABASE_URL (postgres://, postgresql://, mongodb+srv://) and embedded password are completely scrubbed."""
    pg_url = "postgresql://neondb_owner:SuperSecretPgPass_987654@ep-cool-forest-123456.us-east-2.aws.neon.tech/neondb?sslmode=require"
    cfg = reload_settings(
        gemini_api_key=DUMMY_GEMINI_KEY,
        admin_master_key=DUMMY_ADMIN_KEY,
        key_hash_pepper=DUMMY_PEPPER,
        database_url=pg_url,
    )

    b64_url = base64.b64encode(pg_url.encode("utf-8")).decode("ascii")
    raw_error = (
        f"OperationalError: connection to {pg_url} failed; "
        f"password authentication failed for password 'SuperSecretPgPass_987654'; "
        f"fallback URI postgres://other_user:OtherPass123@db.internal:5432/app "
        f"and mongodb+srv://mongo_admin:MongoPass999@cluster0.mongodb.net/prod "
        f"and snapshot mg_snap_v1_eyJrZXlfaWQiOiJhMWIyYzNkNCJ9 "
        f"and b64={b64_url}"
    )

    scrubbed = redact_secrets(raw_error, cfg)
    assert "SuperSecretPgPass_987654" not in scrubbed
    assert "ep-cool-forest-123456.us-east-2.aws.neon.tech" not in scrubbed
    assert "OtherPass123" not in scrubbed
    assert "MongoPass999" not in scrubbed
    assert "mg_snap_v1_" not in scrubbed
    assert b64_url not in scrubbed
    assert "[REDACTED_SECRET]" in scrubbed


def test_zero_plaintext_bootstrap_export_and_disaster_recovery_restore(
    tmp_path: Path, client: TestClient
):
    """Verify GET /admin/api/bootstrap-export contains zero plaintext keys and BOOTSTRAP_KEY_RECORDS restores keys on a fresh DB."""
    created = create_virtual_key_record(
        name="bootstrap-survivor",
        allowed_models=["limited-default", "gemini-flash"],
        forced_system_prompt="Always answer concisely.",
        max_tokens_per_request=768,
        rpm_limit=30,
        token_budget=20000,
    )
    plaintext_key = created["api_key"]
    key_id = created["key_id"]

    # 1. Export via Admin API
    export_resp = client.get(
        "/admin/api/bootstrap-export",
        headers={"X-Admin-Key": DUMMY_ADMIN_KEY},
    )
    assert export_resp.status_code == 200
    export_data = export_resp.json()
    assert export_data["count"] >= 1
    bootstrap_json = export_data["bootstrap_key_records_json"]

    # Crucial zero-leak assertion: the plaintext vk_live_<id>_<secret> MUST NEVER appear in the export
    assert plaintext_key not in bootstrap_json
    assert plaintext_key not in json.dumps(export_data)
    exported_rec = [r for r in export_data["records"] if r["key_id"] == key_id][0]
    assert "api_key" not in exported_rec
    assert len(exported_rec["key_hash"]) == 64

    # 2. Simulate brand-new database + BOOTSTRAP_KEY_RECORDS env var on startup
    fresh_db = str(tmp_path / "fresh_disaster_recovery.db")
    reload_settings(
        gemini_api_key=DUMMY_GEMINI_KEY,
        admin_master_key=DUMMY_ADMIN_KEY,
        key_hash_pepper=DUMMY_PEPPER,
        database_url="",
        database_path=fresh_db,
        bootstrap_key_records=bootstrap_json,
    )
    init_db(fresh_db)

    # 3. Verify the consumer's original plaintext_key authenticates seamlessly against the restored DB
    restored_rec, err = authenticate_virtual_key(plaintext_key, db_path=fresh_db)
    assert err is None
    assert restored_rec is not None
    assert restored_rec["key_id"] == key_id
    assert restored_rec["name"] == "bootstrap-survivor"
    assert restored_rec["allowed_models"] == ["limited-default", "gemini-flash"]
    assert restored_rec["forced_system_prompt"] == "Always answer concisely."
    assert restored_rec["max_tokens_per_request"] == 768
    assert restored_rec["rpm_limit"] == 30
    assert restored_rec["token_budget"] == 20000

    # 4. Calling seed_bootstrap_key_records again is idempotent (ON CONFLICT DO NOTHING)
    count_again = seed_bootstrap_key_records(bootstrap_json, db_path=fresh_db)
    assert count_again >= 1
    assert len(list_virtual_keys(db_path=fresh_db)) == export_data["count"]


def test_automatic_usage_log_retention_pruning(isolated_env: str):
    """Verify max_usage_log_rows prunes oldest usage_logs entries while preserving cumulative virtual_keys.tokens_used."""
    reload_settings(
        gemini_api_key=DUMMY_GEMINI_KEY,
        admin_master_key=DUMMY_ADMIN_KEY,
        key_hash_pepper=DUMMY_PEPPER,
        database_path=isolated_env,
        max_usage_log_rows=3,
    )
    created = create_virtual_key_record(
        name="prune-test-key", db_path=isolated_env
    )
    key_id = created["key_id"]

    # Record 6 requests of 10 tokens each (60 tokens total)
    for i in range(1, 7):
        record_request_usage(
            request_id=f"req_prune_{i}",
            key_id=key_id,
            requested_model="limited-default",
            upstream_model="gemini-2.5-flash",
            prompt_tokens=4,
            completion_tokens=6,
            total_tokens=10,
            status_code=200,
            latency_ms=5.0,
            db_path=isolated_env,
        )

    # 1. usage_logs table must be capped at max_usage_log_rows (3 most recent entries: req_prune_4, 5, 6)
    summary = get_usage_summary(limit=100, key_id=key_id, db_path=isolated_env)
    assert len(summary["logs"]) == 3
    remaining_req_ids = [row["request_id"] for row in summary["logs"]]
    assert remaining_req_ids == ["req_prune_6", "req_prune_5", "req_prune_4"]

    # 2. Cumulative tokens_used and requests_count on virtual_keys must still reflect all 6 requests (60 tokens)
    key_rec = get_virtual_key_by_id(key_id, db_path=isolated_env)
    assert key_rec is not None
    assert key_rec["tokens_used"] == 60
    assert key_rec["requests_count"] == 6
