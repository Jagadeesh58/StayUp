"""Verify a real, uncached StayUp AI request against a running app.

The script deliberately requires source='live'. A calculated fallback or cache
hit is not evidence that the provider call was made for this verification.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=None, help="Base URL of running StayUp app (e.g. http://127.0.0.1:8000). If omitted, tests in-process.")
    args = parser.parse_args()
    payload = {
        "intent": "current_state_context",
        "question": "Explain this chase situation.",
        "state": {
            "score": 150,
            "wickets_lost": 5,
            "completed_overs": 16,
            "balls_in_over": 0,
            "target": 205,
            "overs_limit": 20,
        },
        "force_refresh": True,
    }
    if args.base_url:
        url = args.base_url.rstrip("/") + "/api/ai/explain"
        try:
            response = httpx.post(url, json=payload, timeout=12.0)
            response.raise_for_status()
            result = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            print(f"AI verification failed: {type(exc).__name__}: request could not be verified.", file=sys.stderr)
            return 1
    else:
        url = "in-process /api/ai/explain"
        try:
            from fastapi.testclient import TestClient
            from app.main import app
            client = TestClient(app)
            response = client.post("/api/ai/explain", json=payload)
            response.raise_for_status()
            result = response.json()
        except Exception as exc:
            print(f"AI verification failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

    print(f"Endpoint: {url}")
    print(f"Response source: {result.get('source', 'unknown')}")
    print(f"Provider/model: {result.get('provider') or 'not confirmed'} / {result.get('model') or 'not confirmed'}")
    print(f"Relationship/focus: {result.get('relationship_id')} / {result.get('focus_id')}")
    print(f"AI status: {result.get('ai_status', {}).get('status', 'unknown')}")
    print(f"Answer: {result.get('answer', '(no answer)')}")
    if result.get("source") != "live":
        print("FAILED: this response was cached or calculated; no new provider call was proven.", file=sys.stderr)
        return 1
    print("PASS: the application reports a validated, uncached provider response for this request.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
