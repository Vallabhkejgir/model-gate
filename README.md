# Model-Gate

**Zero-Leak OpenAI-Compatible Gemini API Proxy with Virtual Consumer Keys, Model Allowlists, and Token Quotas**

`model-gate` lets an operator host a single private upstream Gemini API key (`GEMINI_API_KEY`) on their server and issue scoped, revocable virtual API keys (`vk_live_<id>_<secret>`) to consumers. Consumers connect using standard OpenAI SDKs or tools (`/v1/models` and `/v1/chat/completions`) while only ever accessing the specific model alias(es), token budgets, and rate limits allowed for their key.

---

## Key Capabilities

### 1. 7-Layer Zero-Leak Upstream Key Isolation
- **L1 — Credential Stripping & In-Memory Swap**: Consumer `Authorization` / `x-api-key` headers are stripped at ingress. `Authorization: Bearer <GEMINI_API_KEY>` is injected exclusively on outbound `httpx` requests to the pinned `UPSTREAM_BASE_URL`.
- **L2 — Outbound Secret Scrubber (`redact_secrets`)**: Every response payload, error message, exception string, and SSE stream line is scrubbed for the exact `GEMINI_API_KEY` (raw, URL-encoded, and Base64-encoded), `ADMIN_MASTER_KEY`, `KEY_HASH_PEPPER`, and common credential patterns (`AIza...`, `sk-...`, `Bearer ...`).
- **L3 — Upstream Error Reflection Blocker**: Raw upstream `401`, `403`, `429`, and `5xx` payloads (which can contain key hints, Google project numbers, or billing identifiers) are never forwarded to consumers.
- **L4 — Strict Response Header Allowlist**: Upstream headers (`x-goog-*`, `alt-svc`, `server`, etc.) are dropped. Only proxy-controlled headers (`Content-Type`, `Cache-Control`, `X-Proxy-Request-Id`, `X-Consumer-Key-Prefix`, `X-RateLimit-Limit-RPM`, `X-RateLimit-Remaining-RPM`, `X-Quota-Remaining-Tokens`) are emitted.
- **L5 — Explicit Route Surface Only**: Exposes only `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`, and `/admin/*`. No wildcard pass-through routes exist.
- **L6 — SSRF & Redirect Immunity**: Outbound `httpx.AsyncClient(follow_redirects=False)` refuses redirects so the upstream `Authorization` header can never follow a `301/302` to an external host.
- **L7 — Global Exception Handler**: Any unhandled exception is intercepted, scrubbed through `redact_secrets()`, and returned as a clean OpenAI-formatted `500` JSON response without stack traces.

### 2. Scoped Virtual Consumer Keys (`vk_live_...`) & Dual-Engine Persistence
- Generated via CSPRNG (`vk_live_<8-hex-id>_<32-char-secret>`).
- Plaintext key is displayed **once** upon creation; only `HMAC-SHA256(KEY_HASH_PEPPER, full_key)` is stored in the database.
- **Dual-Engine Cloud Persistence (`DATABASE_URL`) + Local SQLite Fallback**:
  - Set `DATABASE_URL=postgresql://...` (Neon, Supabase, or Render PostgreSQL) so issued virtual keys and usage counters survive ephemeral container redeploys and cold starts.
  - Omit `DATABASE_URL` to automatically fall back to local SQLite in WAL mode (`DATABASE_PATH=model_gate.db`).
- **Sub-Millisecond Write-Through Key Cache (`KEY_CACHE_TTL_SECONDS`)**:
  - Verified key records are cached in memory (`~0.01ms` HMAC verification with zero network round-trip on hot `/v1/chat/completions` and `/v1/models` requests) and immediately updated or evicted on key creation, policy edits, token increments, or revocation.
- **Zero-Leak Data Hardening**:
  - `DATABASE_URL` credentials and `postgres://` / `postgresql://` / `mongodb://` URIs are scrubbed by the L2 outbound redaction engine.
  - `MAX_USAGE_LOG_ROWS` (default `5000`) automatically prunes the oldest metadata-only `usage_logs` entries while preserving cumulative token counters on `virtual_keys`.
  - `GET /admin/api/bootstrap-export` exports a zero-plaintext JSON snapshot of pre-hashed keys (`key_id`, `key_prefix`, `key_hash`, and policy limits) that can be placed in `BOOTSTRAP_KEY_RECORDS` for instant disaster-recovery seeding on startup.
- Per-key policy controls:
  - `allowed_models`: Virtual model aliases or explicit models this key can query (e.g. `["limited-default", "gemini-flash-lite"]`).
  - `max_tokens_per_request`: Hard ceiling on output tokens per completion request.
  - `max_input_chars`: Maximum total character length across input messages.
  - `rpm_limit`: Sliding-window requests-per-minute quota.
  - `token_budget`: Cumulative token ceiling across all requests (`null` or `0` = unlimited).
  - `forced_system_prompt`: Optional operator-enforced system instruction prepended to every request.
  - `is_active` & `expires_at`: Instant revocation and automatic time-based expiration.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env` and set your secrets:
```bash
cp .env.example .env
```

Edit `.env`:
```ini
GEMINI_API_KEY=AIzaSyYourRealGeminiKeyHere
UPSTREAM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
ADMIN_MASTER_KEY=replace_with_a_strong_operator_master_key
KEY_HASH_PEPPER=replace_with_a_random_32_byte_pepper
DEFAULT_LIMITED_MODEL=limited-default
MODEL_MAP={"limited-default":"gemini-2.5-flash","gemini-flash-lite":"gemini-2.5-flash-lite","gemini-flash":"gemini-2.5-flash"}
```

### 3. Run the Proxy Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open the built-in Operator Console at:
`http://localhost:8000/admin`

---

## Operator CLI (`cli.py`)

Manage consumer keys and inspect token usage directly from the command line:

```bash
# Issue a new virtual key for a consumer
python cli.py create-key --name "alice" --models "limited-default,gemini-flash" --max-tokens 1024 --rpm 15 --budget 50000

# List all virtual keys and their current usage
python cli.py list-keys

# View aggregated token metrics and recent request logs
python cli.py usage

# Revoke a virtual key immediately
python cli.py revoke-key <key_id>
```

---

## Consumer Usage (OpenAI Python SDK)

Consumers use standard OpenAI client libraries pointed at the proxy's `/v1` base URL with their `vk_live_...` virtual key:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="vk_live_a1b2c3d4_your_issued_virtual_key_secret",
)

# List models permitted for this key
models = client.models.list()
print([m.id for m in models.data])

# Non-streaming chat completion
response = client.chat.completions.create(
    model="limited-default",
    messages=[{"role": "user", "content": "Explain quantum tunneling in two sentences."}],
)
print(response.choices[0].message.content)

# Streaming chat completion
stream = client.chat.completions.create(
    model="limited-default",
    messages=[{"role": "user", "content": "Write a haiku about API gateways."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

---

## Running Tests

```bash
pytest -v
```
