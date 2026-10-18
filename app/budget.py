"""SQLite-backed budget ledger and rate limiting for StayUp Cricket.

Enforces global daily AI quotas and sliding-window per-client burst limits using
an embedded SQLite WAL-mode transaction store. Safeguards against runaway API spend
during live match spikes while gracefully falling back to deterministic calculations.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import logging
import os
import secrets
import sqlite3
import time

from fastapi import HTTPException, Request

logger = logging.getLogger("stayup.budget")

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("APP_DB_PATH", str(BASE_DIR.parent / "stayup_state.sqlite3")))
DAILY_AI_BUDGET = max(1, int(os.getenv("AI_DAILY_BUDGET", "200")))
BURST_PER_MINUTE = max(1, int(os.getenv("AI_BURST_PER_MINUTE", "4")))
_LOCAL_RATE_LIMIT_SALT = secrets.token_hex(32)


def current_day_utc() -> str:
    """Return today's date in UTC (YYYY-MM-DD)."""
    return datetime.now(timezone.utc).date().isoformat()


def initialize_budget_db(db_path: Path | None = None) -> None:
    """Initialize SQLite schema for the daily AI spend and rate-limit ledger."""
    target_path = db_path or DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(target_path, timeout=3) as conn:
        # WAL mode permits concurrent readers without blocking write transactions during traffic spikes
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS ai_budget ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "day_utc TEXT NOT NULL, "
            "client_hash TEXT NOT NULL, "
            "created_at REAL NOT NULL)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ai_budget_day ON ai_budget(day_utc)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ai_budget_client ON ai_budget(client_hash, created_at)")
        conn.commit()


def get_client_address(request: Request) -> str:
    """Extract client IP address, honoring proxy headers only if explicitly trusted."""
    if os.getenv("TRUST_PROXY_HEADERS", "false").strip().lower() in {"1", "true", "yes"}:
        forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        if forwarded:
            return forwarded[:128]
    return request.client.host if request.client else "unknown"


def reserve_ai_budget(request: Request, db_path: Path | None = None) -> None:
    """Reserve one live call in a single-instance SQLite budget ledger for FastAPI."""
    target_path = db_path or DB_PATH
    if not target_path.exists():
        initialize_budget_db(target_path)
    now = time.time()
    day = current_day_utc()
    salt = os.getenv("RATE_LIMIT_SALT", "").strip() or _LOCAL_RATE_LIMIT_SALT
    client_hash = sha256(f"{salt}:{get_client_address(request)}".encode()).hexdigest()
    try:
        with sqlite3.connect(target_path, timeout=3, isolation_level="IMMEDIATE") as conn:
            conn.execute("DELETE FROM ai_budget WHERE created_at < ?", (now - 86_400,))
            total = conn.execute("SELECT COUNT(*) FROM ai_budget WHERE day_utc = ?", (day,)).fetchone()[0]
            recent = conn.execute(
                "SELECT COUNT(*) FROM ai_budget WHERE client_hash = ? AND created_at >= ?",
                (client_hash, now - 60),
            ).fetchone()[0]
            limit = max(1, int(os.getenv("AI_DAILY_BUDGET", str(DAILY_AI_BUDGET))))
            burst = max(1, int(os.getenv("AI_BURST_PER_MINUTE", str(BURST_PER_MINUTE))))
            if total >= limit:
                logger.warning("Daily live AI budget reached: %d/%d for day %s", total, limit, day)
                raise HTTPException(
                    status_code=429,
                    detail="StayUp's daily live-AI budget has been reached. Calculated explanations remain available.",
                )
            if recent >= burst:
                logger.warning("Burst rate limit reached for client %s: %d/min", client_hash[:12], recent)
                raise HTTPException(
                    status_code=429,
                    detail="Please wait a minute before requesting another live AI explanation.",
                )
            conn.execute("INSERT INTO ai_budget (day_utc, client_hash, created_at) VALUES (?, ?, ?)", (day, client_hash, now))
            conn.commit()
    except sqlite3.Error as exc:
        logger.error("SQLite budget ledger error: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Live AI budget control is temporarily unavailable; use the calculated explanation.",
        ) from exc


def reserve_streamlit_ai_budget(session_token: str, db_path: Path | None = None) -> None:
    """Fail-closed global daily and per-session call budget for native Streamlit."""
    target_path = db_path or DB_PATH
    initialize_budget_db(target_path)
    now = time.time()
    day = current_day_utc()
    global_limit = max(1, int(os.getenv("AI_DAILY_BUDGET", str(DAILY_AI_BUDGET))))
    burst = max(1, int(os.getenv("AI_BURST_PER_MINUTE", str(BURST_PER_MINUTE))))
    salt = os.getenv("RATE_LIMIT_SALT", "").strip() or "stayup-streamlit-session"
    key = sha256(f"{salt}:{session_token}".encode()).hexdigest()
    try:
        with sqlite3.connect(target_path, timeout=3, isolation_level="IMMEDIATE") as conn:
            conn.execute("DELETE FROM ai_budget WHERE created_at < ?", (now - 86_400,))
            total = conn.execute("SELECT COUNT(*) FROM ai_budget WHERE day_utc = ?", (day,)).fetchone()[0]
            recent = conn.execute(
                "SELECT COUNT(*) FROM ai_budget WHERE client_hash = ? AND created_at >= ?",
                (key, now - 60),
            ).fetchone()[0]
            if total >= global_limit:
                logger.warning("Streamlit: daily AI budget reached: %d/%d on %s", total, global_limit, day)
                raise RuntimeError("The daily live-AI budget has been reached. Calculated explanations remain available.")
            if recent >= burst:
                logger.warning("Streamlit: burst limit hit for session %s: %d/min", key[:12], recent)
                raise RuntimeError("Please wait a minute before requesting another live AI explanation.")
            conn.execute("INSERT INTO ai_budget (day_utc, client_hash, created_at) VALUES (?, ?, ?)", (day, key, now))
            conn.commit()
    except sqlite3.Error as exc:
        logger.error("Streamlit SQLite budget ledger error: %s", exc)
        raise RuntimeError("Live AI budget control is unavailable; the calculated explanation remains available.") from exc
