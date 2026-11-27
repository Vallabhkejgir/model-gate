"""Operator CLI for Model-Gate virtual consumer key lifecycle and usage audit."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from app.config import get_settings
from app.database import (
    create_virtual_key_record,
    get_usage_summary,
    init_db,
    list_virtual_keys,
    revoke_virtual_key,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Model-Gate Operator CLI for managing virtual consumer API keys.",
    )
    parser.add_argument(
        "--db",
        dest="db_path",
        default=None,
        help="Optional override path for the SQLite database file.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # create-key
    create_p = subparsers.add_parser(
        "create-key", help="Issue a new consumer virtual key (vk_live_...)"
    )
    create_p.add_argument(
        "--name", required=True, help="Consumer label (e.g. 'alice')"
    )
    create_p.add_argument(
        "--models",
        default=None,
        help="Comma-separated allowed virtual models (e.g. 'limited-default,gemini-flash')",
    )
    create_p.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max output tokens per request (default: 1024)",
    )
    create_p.add_argument(
        "--max-input-chars",
        type=int,
        default=16000,
        help="Max input message characters per request (default: 16000)",
    )
    create_p.add_argument(
        "--rpm",
        type=int,
        default=15,
        help="Requests per minute limit (default: 15)",
    )
    create_p.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Total token budget across requests (0 or omitted = unlimited)",
    )
    create_p.add_argument(
        "--system-prompt",
        default=None,
        help="Optional forced system prompt injected into every request",
    )
    create_p.add_argument(
        "--expires-at",
        default=None,
        help="Optional expiration timestamp in ISO-8601 UTC format",
    )

    # list-keys
    subparsers.add_parser(
        "list-keys", help="List all consumer virtual keys and quota status"
    )

    # revoke-key
    revoke_p = subparsers.add_parser(
        "revoke-key", help="Revoke (or delete) a consumer virtual key by key_id"
    )
    revoke_p.add_argument("key_id", help="8-character hex key_id to revoke")
    revoke_p.add_argument(
        "--hard",
        action="store_true",
        help="Permanently delete the key record from the database",
    )

    # usage
    usage_p = subparsers.add_parser(
        "usage", help="Inspect aggregated token usage and recent request audit logs"
    )
    usage_p.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Number of recent audit log rows to display (default: 25)",
    )
    usage_p.add_argument(
        "--key-id",
        default=None,
        help="Filter usage statistics by a specific 8-char key_id",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = get_settings()
    db_path = args.db_path

    init_db(db_path)

    if args.command == "create-key":
        if args.models:
            allowed_models = [
                m.strip() for m in args.models.split(",") if m.strip()
            ]
        else:
            allowed_models = [settings.default_limited_model]

        record = create_virtual_key_record(
            name=args.name.strip(),
            allowed_models=allowed_models,
            forced_system_prompt=args.system_prompt,
            max_tokens_per_request=args.max_tokens,
            max_input_chars=args.max_input_chars,
            rpm_limit=args.rpm,
            token_budget=args.budget,
            expires_at=args.expires_at,
            db_path=db_path,
        )
        if args.json:
            print(json.dumps(record, indent=2))
        else:
            print("=== Virtual Consumer Key Created ===")
            print(f"Key ID         : {record['key_id']}")
            print(f"Consumer Name  : {record['name']}")
            print(f"Virtual Key    : {record['api_key']}")
            print(f"Allowed Models : {', '.join(record['allowed_models'])}")
            print(f"Max Tokens/Req : {record['max_tokens_per_request']}")
            print(f"RPM Limit      : {record['rpm_limit']}")
            print(
                f"Token Budget   : {record['token_budget'] if record['token_budget'] else 'unlimited'}"
            )
            print(
                "NOTE: Save the Virtual Key now. Only its HMAC-SHA256 hash is stored."
            )
        return 0

    if args.command == "list-keys":
        keys = list_virtual_keys(db_path=db_path)
        if args.json:
            print(json.dumps(keys, indent=2))
        else:
            if not keys:
                print("No virtual consumer keys found.")
                return 0
            print(
                f"{'KEY_ID':<10} {'NAME':<18} {'STATUS':<10} {'MODELS':<28} {'TOKENS (USED/BUDGET)':<22} {'RPM':<5}"
            )
            print("-" * 98)
            for k in keys:
                status = "ACTIVE" if k["is_active"] else "REVOKED"
                models_str = ",".join(k["allowed_models"])[:26]
                budget_str = (
                    f"{k['tokens_used']}/{k['token_budget']}"
                    if k["token_budget"]
                    else f"{k['tokens_used']}/inf"
                )
                print(
                    f"{k['key_id']:<10} {k['name'][:17]:<18} {status:<10} "
                    f"{models_str:<28} {budget_str:<22} {k['rpm_limit']:<5}"
                )
        return 0

    if args.command == "revoke-key":
        ok = revoke_virtual_key(
            key_id=args.key_id.strip(),
            hard_delete=bool(args.hard),
            db_path=db_path,
        )
        if not ok:
            if args.json:
                print(json.dumps({"error": "not_found", "key_id": args.key_id}))
            else:
                print(
                    f"Error: Virtual key '{args.key_id}' not found.",
                    file=sys.stderr,
                )
            return 1
        if args.json:
            print(
                json.dumps(
                    {
                        "key_id": args.key_id,
                        "revoked": True,
                        "deleted": bool(args.hard),
                    }
                )
            )
        else:
            action = "Deleted" if args.hard else "Revoked"
            print(f"{action} virtual key '{args.key_id}'.")
        return 0

    if args.command == "usage":
        report = get_usage_summary(
            limit=args.limit, key_id=args.key_id, db_path=db_path
        )
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            s = report["summary"]
            print("=== Aggregated Proxy Usage ===")
            print(f"Total Requests          : {s['total_requests']}")
            print(f"Total Prompt Tokens     : {s['total_prompt_tokens']}")
            print(f"Total Completion Tokens : {s['total_completion_tokens']}")
            print(f"Total Tokens            : {s['total_tokens']}")
            logs = report["logs"]
            if logs:
                print("\nRecent Requests:")
                for row in logs:
                    print(
                        f"  [{row['created_at']}] req={row['request_id']} "
                        f"key={row['key_id']} model={row['requested_model']} "
                        f"tokens={row['total_tokens']} status={row['status_code']} "
                        f"latency={row['latency_ms']}ms"
                    )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
