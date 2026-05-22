#!/usr/bin/env python3
"""CLI helper for AEROS runtime benchmark sessions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Optional
from urllib import error, request


def api_request(
    *,
    api_base: str,
    method: str,
    path: str,
    payload: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Send a JSON request to the local AEROS API."""
    url = f"{api_base.rstrip('/')}{path}"
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"status": "error", "message": body or str(exc)}
        raise RuntimeError(parsed.get("message", str(exc))) from exc
    except error.URLError as exc:
        raise RuntimeError(f"Failed to reach API at {url}: {exc}") from exc


def print_json(payload: dict[str, Any]) -> None:
    """Pretty-print a JSON payload."""
    print(json.dumps(payload, indent=2))


def handle_start(args: argparse.Namespace) -> int:
    response = api_request(
        api_base=args.api_base,
        method="POST",
        path="/benchmark/start",
        payload={
            "name": args.name,
            "scenario": args.scenario,
            "notes": args.notes,
        },
    )
    print_json(response)
    return 0


def handle_stop(args: argparse.Namespace) -> int:
    query = "?export=true" if args.export else "?export=false"
    response = api_request(
        api_base=args.api_base,
        method="POST",
        path=f"/benchmark/stop{query}",
    )
    print_json(response)
    return 0


def handle_status(args: argparse.Namespace) -> int:
    response = api_request(
        api_base=args.api_base,
        method="GET",
        path="/benchmark/status",
    )
    print_json(response)
    return 0


def handle_latest(args: argparse.Namespace) -> int:
    response = api_request(
        api_base=args.api_base,
        method="GET",
        path="/benchmark/latest",
    )
    print_json(response)
    return 0


def handle_run(args: argparse.Namespace) -> int:
    start_response = api_request(
        api_base=args.api_base,
        method="POST",
        path="/benchmark/start",
        payload={
            "name": args.name,
            "scenario": args.scenario,
            "notes": args.notes,
        },
    )
    print("Benchmark started:")
    print_json(start_response)

    try:
        time.sleep(args.duration)
    finally:
        query = "?export=true" if args.export else "?export=false"
        stop_response = api_request(
            api_base=args.api_base,
            method="POST",
            path=f"/benchmark/stop{query}",
        )
        print("Benchmark stopped:")
        print_json(stop_response)

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(description="AEROS runtime benchmark helper")
    parser.add_argument(
        "--api-base",
        default="http://127.0.0.1:8000",
        help="Base URL for the AEROS API (default: http://127.0.0.1:8000)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Start a benchmark, wait, and stop it")
    run_parser.add_argument("--duration", type=float, default=30.0, help="Duration in seconds")
    run_parser.add_argument("--name", default=None, help="Optional benchmark name")
    run_parser.add_argument("--scenario", default=None, help="Scenario label")
    run_parser.add_argument("--notes", default=None, help="Optional notes")
    run_parser.add_argument(
        "--export",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to export the run to benchmarks/runs",
    )
    run_parser.set_defaults(handler=handle_run)

    start_parser = subparsers.add_parser("start", help="Start a benchmark session")
    start_parser.add_argument("--name", default=None, help="Optional benchmark name")
    start_parser.add_argument("--scenario", default=None, help="Scenario label")
    start_parser.add_argument("--notes", default=None, help="Optional notes")
    start_parser.set_defaults(handler=handle_start)

    stop_parser = subparsers.add_parser("stop", help="Stop the active benchmark session")
    stop_parser.add_argument(
        "--export",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to export the run to benchmarks/runs",
    )
    stop_parser.set_defaults(handler=handle_stop)

    status_parser = subparsers.add_parser("status", help="Show benchmark session status")
    status_parser.set_defaults(handler=handle_status)

    latest_parser = subparsers.add_parser("latest", help="Show the latest benchmark result")
    latest_parser.set_defaults(handler=handle_latest)

    return parser


def main() -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        return int(args.handler(args))
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
