"""FastAPI entrypoint for StayUp Cricket.

Exposes a strict JSON API for chase state calculation, scenario comparison,
AI-explained turning points, historical replay navigation and model evaluation.
All endpoints delegate to the canonical cricket engine and model modules; no
cricket arithmetic lives in this file. The companion Streamlit interface is the
primary fan-facing deployment target; this API serves as an optional headless mode.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal
import asyncio
import logging
import os
import sqlite3
import time
import uuid

from dotenv import load_dotenv

# Load a local .env for developer runs; real deployments should inject secrets.
# This is intentionally before importing the AI service, which reads settings on demand.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError

from .ai_service import AIProviderError, ai_configuration, select_focus
from .presentation import chase_verdict
from .model import (
    ModelDataError, build_replay_view, load_data, lookup_chase, match_index_public, matchup_grid, next_over_outcomes, pre_match_info, replay_matches, similar_chase_cohort,
)
from .cricket import (
    CricketInputError,
    HERO_REPLAY,
    calculate_manual_state,
    source_reconciliation_summary,
)
from .budget import (
    BURST_PER_MINUTE,
    DAILY_AI_BUDGET,
    DB_PATH,
    _LOCAL_RATE_LIMIT_SALT,
    current_day_utc,
    get_client_address,
    initialize_budget_db as _init_budget_db,
)
from .explanation import (
    _fallback_focus,
    render_explanation,
    scenario_map,
)

logger = logging.getLogger("stayup.api")
BASE_DIR = Path(__file__).resolve().parent
MAX_BODY_BYTES = 16_384


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ChaseInput(StrictModel):
    score: StrictInt = Field(ge=0, le=1000)
    wickets_lost: StrictInt = Field(ge=0, le=10)
    completed_overs: StrictInt = Field(ge=0, le=20)
    balls_in_over: StrictInt = Field(ge=0, le=5)
    target: StrictInt = Field(ge=1, le=1001)
    overs_limit: Literal[20] = 20


class ScenarioRequest(StrictModel):
    state: ChaseInput
    # The scenario panel renders all three options; AI comparison still accepts exactly two.
    scenario_ids: list[Literal["steady", "big_over", "setback"]] = Field(min_length=2, max_length=3)


class AIExplainRequest(StrictModel):
    intent: Literal["current_state_context", "compare_scenarios", "explain_turning_point", "explain_chase_change"]
    question: str = Field(min_length=1, max_length=200)
    state: ChaseInput | None = None
    scenario_ids: list[Literal["steady", "big_over", "setback"]] | None = Field(default=None, min_length=2, max_length=2)
    cursor: StrictInt | None = Field(default=None, ge=0, le=200)
    card_id: str | None = Field(default=None, max_length=64)
    force_refresh: bool = False
    language: Literal["English", "Hinglish", "Hindi", "Telugu", "Tamil", "Kannada", "Bengali"] = "English"


SUPPORTED_QUESTIONS = {
    "current_state_context": "Explain this chase situation.",
    "compare_scenarios": "Compare these hypothetical overs.",
    "explain_turning_point": "Why does this turning point matter?",
    "explain_chase_change": "What changed at this moment?",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_budget_db()
    yield


app = FastAPI(
    title="StayUp Cricket",
    version="1.0.0",
    description="A source-grounded T20 chase companion. Historical examples only; no live feed.",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

@app.middleware("http")
async def security_headers_and_body_bound(request: Request, call_next):
    length = request.headers.get("content-length")
    if length:
        try:
            if int(length) > MAX_BODY_BYTES:
                return JSONResponse(status_code=413, content={"detail": "Request body is too large."})
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header."})
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


def initialize_budget_db() -> None:
    _init_budget_db(DB_PATH)


def reserve_ai_budget(request: Request) -> None:
    """Reserve one live call in a single-instance SQLite budget ledger."""
    if not DB_PATH.exists():
        initialize_budget_db()
    now = time.time()
    day = current_day_utc()
    salt = os.getenv("RATE_LIMIT_SALT", "").strip() or _LOCAL_RATE_LIMIT_SALT
    client_hash = sha256(f"{salt}:{get_client_address(request)}".encode()).hexdigest()
    try:
        with sqlite3.connect(DB_PATH, timeout=3, isolation_level="IMMEDIATE") as conn:
            conn.execute("DELETE FROM ai_budget WHERE created_at < ?", (now - 86_400,))
            total = conn.execute("SELECT COUNT(*) FROM ai_budget WHERE day_utc = ?", (day,)).fetchone()[0]
            recent = conn.execute("SELECT COUNT(*) FROM ai_budget WHERE client_hash = ? AND created_at >= ?", (client_hash, now - 60)).fetchone()[0]
            if total >= DAILY_AI_BUDGET:
                logger.warning("Daily live AI budget reached: %d/%d for day %s", total, DAILY_AI_BUDGET, day)
                raise HTTPException(status_code=429, detail="StayUp's daily live-AI budget has been reached. Calculated explanations remain available.")
            if recent >= BURST_PER_MINUTE:
                logger.warning("Burst rate limit reached for client %s: %d/min", client_hash[:12], recent)
                raise HTTPException(status_code=429, detail="Please wait a minute before requesting another live AI explanation.")
            conn.execute("INSERT INTO ai_budget (day_utc, client_hash, created_at) VALUES (?, ?, ?)", (day, client_hash, now))
            conn.commit()
    except sqlite3.Error as exc:
        logger.error("SQLite budget ledger error: %s", exc)
        raise HTTPException(status_code=503, detail="Live AI budget control is temporarily unavailable; use the calculated explanation.") from exc


def parse_state(data: ChaseInput) -> dict[str, Any]:
    return calculate_manual_state(
        score=data.score,
        wickets_lost=data.wickets_lost,
        completed_overs=data.completed_overs,
        balls_in_over=data.balls_in_over,
        target=data.target,
        overs_limit=data.overs_limit,
    )


def build_ai_context(request_data: AIExplainRequest) -> tuple[dict[str, Any], list[str], list[str], list[str], dict[str, dict[str, set[str]]]]:
    intent = request_data.intent
    if request_data.question != SUPPORTED_QUESTIONS[intent]:
        raise HTTPException(status_code=400, detail="That question is not supported in this context. Use one of the questions shown on the current screen.")
    if intent == "current_state_context":
        if request_data.scenario_ids is not None or request_data.cursor is not None or request_data.card_id is not None:
            raise HTTPException(status_code=400, detail="This question accepts only the manually entered chase state; stale scenario or replay fields are not allowed.")
        if request_data.state is None:
            raise HTTPException(status_code=400, detail="A valid chase state is required for this question.")
        state = parse_state(request_data.state)
        if state["status"] != "active":
            raise HTTPException(status_code=400, detail="This question is available only during an active chase.")
        claims = {
            "runs_required": state["runs_required"],
            "balls_remaining": state["balls_remaining"],
            "wickets_remaining": state["wickets_remaining"],
            "required_run_rate": state["required_run_rate"],
            "current_run_rate": state["current_run_rate"],
        }
        claim_ids = [key for key, value in claims.items() if value is not None]
        focuses = ["runway", "wickets_buffer"]
        requirements: dict[str, dict[str, set[str]]] = {
            "runway": {"current_state_context": {"runs_required", "balls_remaining"}},
            "wickets_buffer": {"current_state_context": {"wickets_remaining", "runs_required"}},
        }
        # Presence in the dict is not enough: at 0.0 completed overs the
        # current rate is undefined (None), so rate-pressure must not be offered.
        if claims["current_run_rate"] is not None and claims["required_run_rate"] is not None:
            focuses.append("rate_pressure")
            requirements["rate_pressure"] = {"current_state_context": {"current_run_rate", "required_run_rate"}}
        context = {"intent": intent, "state": state, "claims": claims, "claim_ids": claim_ids}
        return context, ["current_state_context"], focuses, claim_ids, requirements

    if intent == "compare_scenarios":
        if request_data.cursor is not None or request_data.card_id is not None:
            raise HTTPException(status_code=400, detail="Scenario comparison cannot include replay fields.")
        if request_data.state is None or request_data.scenario_ids is None:
            raise HTTPException(status_code=400, detail="A state and two scenario choices are required.")
        if len(set(request_data.scenario_ids)) != len(request_data.scenario_ids):
            raise HTTPException(status_code=400, detail="Choose different scenarios; duplicate values are not accepted.")
        state = parse_state(request_data.state)
        if state["status"] != "active":
            raise HTTPException(status_code=400, detail="Scenario comparison is available only during an active chase.")
        all_scenarios = scenario_map(state)
        scenarios = [all_scenarios[sid] for sid in request_data.scenario_ids]
        if any(s["after_state"] is None or s["after_state"]["status"] != "active" for s in scenarios):
            raise HTTPException(status_code=400, detail="AI comparison is suppressed when either hypothetical has an ambiguous or terminal outcome.")
        claims: dict[str, Any] = {}
        for scenario in scenarios:
            sid = scenario["id"]
            after = scenario["after_state"]
            claims[f"{sid}_runs_required"] = after["runs_required"]
            claims[f"{sid}_balls_remaining"] = after["balls_remaining"]
            claims[f"{sid}_wickets_remaining"] = after["wickets_remaining"]
            claims[f"{sid}_required_rate"] = after["required_run_rate"]
        claim_ids = list(claims)
        focuses = ["runs_difference", "required_rate_difference"]
        requirements = {
            "runs_difference": {"scenario_difference": {f"{scenarios[0]['id']}_runs_required", f"{scenarios[1]['id']}_runs_required"}},
            "required_rate_difference": {"scenario_difference": {f"{scenarios[0]['id']}_required_rate", f"{scenarios[1]['id']}_required_rate"}},
        }
        if scenarios[0]["after_state"]["wickets_remaining"] != scenarios[1]["after_state"]["wickets_remaining"]:
            focuses.append("wickets_difference")
            requirements["wickets_difference"] = {"scenario_difference": {f"{scenarios[0]['id']}_wickets_remaining", f"{scenarios[1]['id']}_wickets_remaining"}}
        context = {"intent": intent, "base_state": state, "scenarios": scenarios, "claims": claims, "claim_ids": claim_ids}
        return context, ["scenario_difference"], focuses, claim_ids, requirements

    if request_data.state is not None or request_data.scenario_ids is not None:
        raise HTTPException(status_code=400, detail="Replay questions cannot include manual-form or hypothetical-scenario fields.")
    if request_data.cursor is None or request_data.card_id is None:
        raise HTTPException(status_code=400, detail="A replay cursor and evidence-card ID are required.")
    if request_data.cursor > HERO_REPLAY.total_events:
        raise HTTPException(status_code=400, detail="Replay cursor is beyond the source data.")
    card_spec = next((item for item in HERO_REPLAY.turning_points if item["id"] == request_data.card_id), None)
    if card_spec is None or card_spec["end_cursor"] > request_data.cursor:
        raise HTTPException(status_code=400, detail="That turning point has not been reached in the replay yet.")
    card = HERO_REPLAY._card_public(card_spec)
    before, after = card["before_state"], card["after_state"]
    # The closing delivery needs a terminal-state relationship, not a falsely
    # labelled rate change. Keep this path restricted to the verified final card.
    if intent == "explain_chase_change":
        if request_data.card_id != "five-sixes-finish":
            raise HTTPException(status_code=400, detail="The chase-change question is available only for the verified final-over sequence.")
        if not after.get("is_complete") or after.get("status") != "target_reached":
            raise HTTPException(status_code=400, detail="The final-over source does not prove that the target was reached at this cursor.")
        evidence = card.get("evidence", [])
        claims = {
            "before_runs_required": before["runs_required"],
            "before_balls_remaining": before["balls_remaining"],
            "runs_added": sum(int(event["total_runs"]) for event in evidence),
            "recorded_deliveries": len(evidence),
            "after_score": after["score"],
            "target": after["target"],
        }
        if claims["after_score"] < claims["target"] or claims["runs_added"] != claims["after_score"] - before["score"]:
            raise HTTPException(status_code=500, detail="Final-over evidence failed its source reconciliation check.")
        claim_ids = list(claims)
        requirements = {"target_reached": {"target_reached": set(claim_ids)}}
        context = {"intent": intent, "card": card, "claims": claims, "claim_ids": claim_ids}
        return context, ["target_reached"], ["target_reached"], claim_ids, requirements

    if request_data.card_id == "five-sixes-finish":
        raise HTTPException(status_code=400, detail="Use the chase-change question for the final-over sequence.")
    if before.get("required_run_rate") is None or after.get("required_run_rate") is None:
        raise HTTPException(status_code=400, detail="AI explanation is unavailable for a rate-undefined turning point.")
    claim_ids = ["before_required_rate", "after_required_rate", "before_wickets_lost", "after_wickets_lost", "before_runs_required", "after_runs_required", "before_balls_remaining", "after_balls_remaining"]
    claims = {
        "before_required_rate": before["required_run_rate"],
        "after_required_rate": after["required_run_rate"],
        "before_wickets_lost": before["wickets_lost"],
        "after_wickets_lost": after["wickets_lost"],
        "before_runs_required": before["runs_required"],
        "after_runs_required": after["runs_required"],
        "before_balls_remaining": before["balls_remaining"],
        "after_balls_remaining": after["balls_remaining"],
    }
    focuses = ["rate_shift"]
    requirements = {"rate_shift": {"rate_changed": {"before_required_rate", "after_required_rate"}}}
    relationships = ["rate_changed"]
    if before["wickets_lost"] != after["wickets_lost"]:
        focuses.append("wicket_rate_shift")
        relationships.append("wickets_and_rate")
        requirements["wicket_rate_shift"] = {"wickets_and_rate": {"before_wickets_lost", "after_wickets_lost", "before_required_rate", "after_required_rate"}}
    context = {"intent": intent, "card": card, "claims": claims, "claim_ids": claim_ids}
    return context, relationships, focuses, claim_ids, requirements


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
async def home():
    """Keep API mode explicit; the full product UI lives in Streamlit."""
    page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>StayUp Cricket · API mode</title></head><body>
<main><p>STAYUP CRICKET</p><h1>FastAPI API mode</h1>
<p>This process exposes the optional data and explanation API. The full fan experience is the native Streamlit application.</p>
<p>For local development, run <code>streamlit run streamlit_app.py</code> from the project root.</p>
<p><strong>Enter the score yourself. Historical examples only — no live feed.</strong></p>
<p>Available service check: <a href="/api/health">/api/health</a></p>
</main></body></html>"""
    return HTMLResponse(page, headers={"Cache-Control": "no-cache"})


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "StayUp Cricket",
        "version": app.version,
        "ai": ai_configuration(),
        "live_feed": False,
        "history_feature": "enabled when strict cohort reaches the minimum N; otherwise hidden",
        "data": {"status": "ready" if (BASE_DIR / "data" / "model_observations.json.gz").exists() else "missing", "eligible_matches": load_data()["manifest"]["eligible_reconciled_matches"] if (BASE_DIR / "data" / "model_observations.json.gz").exists() else 0},
    }


@app.get("/api/hero/meta")
async def hero_meta():
    return HERO_REPLAY.public_meta()


@app.get("/api/hero/replay")
async def hero_replay(cursor: int = 0):
    try:
        return HERO_REPLAY.public_at(cursor)
    except CricketInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/state")
async def calculate_state(payload: ChaseInput):
    try:
        state = parse_state(payload)
        model_result = None
        next_over = None
        if state["status"] == "active":
            try:
                model_result = lookup_chase(state["score"], state["wickets_lost"], state["legal_balls_used"], state["target"], as_of=current_day_utc())
                if state["balls_remaining"] >= 6:
                    next_over = next_over_outcomes(state["score"], state["wickets_lost"], state["legal_balls_used"], state["target"], as_of=current_day_utc())
            except ModelDataError:
                model_result = {"probability":None,"effective_sample_size":0,"note":"Historical model data is not installed; calculated chase equation remains available."}
        p = model_result.get("probability") if model_result else None
        verdict = chase_verdict(p)
        try:
            cohort = similar_chase_cohort(
                state["score"], state["wickets_lost"], state["legal_balls_used"], state["target"],
                as_of=current_day_utc(), exclude_match_id=None,
            ) if state["status"] == "active" else {"available":False,"k":None,"n":0,"reason":"The current chase state is terminal."}
        except ModelDataError as exc:
            cohort = {"available":False,"k":None,"n":0,"reason":f"Historical cohort is unavailable: {exc}"}
        return {"state":state,"verdict":verdict,"chasing_team_win_chance":p,"defending_team_win_chance":round(1-p,4) if p is not None else None,"win_model":model_result,"cohort":cohort,"next_over_outcomes":next_over}
    except CricketInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/scenarios")
async def calculate_scenarios(payload: ScenarioRequest):
    try:
        if len(set(payload.scenario_ids)) != len(payload.scenario_ids):
            raise CricketInputError("Choose two different hypothetical scenarios; duplicate values are not accepted.")
        state = parse_state(payload.state)
        if state["status"] != "active":
            raise CricketInputError("Scenarios are available only during an active chase.")
        scenarios = scenario_map(state)
        return {"base_state": state, "scenarios": [scenarios[sid] for sid in payload.scenario_ids]}
    except CricketInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/ai/explain")
async def explain_with_ai(payload: AIExplainRequest, request: Request):
    request_id = uuid.uuid4().hex[:12]
    try:
        context, relationships, focuses, claim_ids, requirements = build_ai_context(payload)
    except HTTPException:
        raise
    except (CricketInputError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    fallback_relationship, fallback_focus, fallback_claims = _fallback_focus(payload.intent, context)
    selection = None
    selection_source = "calculated"
    config = ai_configuration()
    provider_error = None
    if config["enabled"] and config["configured"]:
        try:
            # Keep the synchronous provider client and SQLite reservation off the
            # event loop so other users can still navigate/replay during a timeout.
            selection = await asyncio.to_thread(
                select_focus,
                context={"intent": payload.intent, "claims": context["claims"], "source_ids": (context.get("card") or {}).get("event_ids", []), "language": payload.language},
                relationship_ids=relationships,
                focus_ids=focuses,
                claim_ids=claim_ids,
                requirements=requirements,
                force_refresh=payload.force_refresh,
                on_live_call=lambda: reserve_ai_budget(request),
            )
            fallback_relationship = selection.relationship_id
            fallback_focus = selection.focus_id
            fallback_claims = selection.selected_claim_ids
            selection_source = selection.source
        except HTTPException as exc:
            if exc.status_code in {429, 503}:
                provider_error = str(exc.detail)
            else:
                raise
        except AIProviderError as exc:
            if exc.status_code:
                logger.warning("AI explanation provider error HTTP %d: %s", exc.status_code, exc)
                provider_error = f"Live AI was unavailable (provider HTTP {exc.status_code}); calculated explanation shown."
            else:
                logger.warning("AI explanation validation error: %s", exc)
                provider_error = "Live AI was unavailable or its evidence selection failed validation."
        except Exception as exc:
            logger.exception("Unexpected error in explain_with_ai: %s", exc)
            # Keep operational details out of the public response.
            provider_error = "Live AI was unavailable; a calculated explanation is shown instead."

    rendered = render_explanation(payload.intent, fallback_focus, context, payload.language)
    if selection and selection.explanation.strip():
        rendered["answer"] = selection.explanation
    return {
        "request_id": request_id,
        "intent": payload.intent,
        "language": payload.language,
        "relationship_id": fallback_relationship,
        "focus_id": fallback_focus,
        "selected_claim_ids": fallback_claims,
        "answer": rendered["answer"],
        "source": selection_source,
        "provider": selection.provider if selection else None,
        "model": selection.model if selection else None,
        "label": (
            f"Live AI explanation · {selection.model}" if selection and selection.source == "live"
            else f"Cached AI explanation · {selection.model}" if selection and selection.source == "cached"
            else "Calculated explanation"
        ),
        "detail": selection.detail if selection else provider_error or config["note"],
        "ai_status": ai_configuration(),
    }


@app.get("/api/matches")
async def historical_matches():
    try:
        return {"matches": match_index_public(), "count": len(match_index_public()), "data": load_data()["manifest"]}
    except ModelDataError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/matches/{match_id}/replay")
async def historical_replay(match_id: str, cursor: int = 0):
    try:
        view = build_replay_view(match_id, cursor)
        state = view["current_state"]
        if view["first_innings_complete"] and state["innings_index"] == 1 and view["target"] is not None:
            try:
                state_for_cohort = lookup_chase(state["score"], state["wickets_lost"], state["legal_balls_used"], view["target"], as_of=view["match"]["date"], exclude_match_id=match_id)
                view["current_win_model"] = state_for_cohort
                view["next_over_outcomes"] = next_over_outcomes(state["score"], state["wickets_lost"], state["legal_balls_used"], view["target"], as_of=view["match"]["date"], exclude_match_id=match_id) if state["legal_balls_used"] < 120 and state["score"] < view["target"] and state["wickets_lost"] < 10 else {"available":False,"reason":"The chase is terminal."}
                view["cohort"] = similar_chase_cohort(state["score"], state["wickets_lost"], state["legal_balls_used"], view["target"], as_of=view["match"]["date"], exclude_match_id=match_id)
            except ModelDataError:
                view["current_win_model"] = {"probability":None,"effective_sample_size":0,"note":"Historical model data unavailable."}
        return view
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown replay match.") from exc
    except (ValueError, CricketInputError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/matches/{match_id}/pre-match")
async def historical_pre_match(match_id: str):
    result = pre_match_info(match_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Unknown match.")
    return result


@app.get("/api/matches/{match_id}/matchups")
async def historical_matchups(match_id: str, innings_index: int = 0):
    match = next((m for m in replay_matches() if m["match_id"] == str(match_id)), None)
    if match is None: raise HTTPException(status_code=404, detail="Unknown match.")
    if innings_index not in (0, 1): raise HTTPException(status_code=422, detail="innings_index must be 0 or 1.")
    batting = match["first_innings_team"] if innings_index == 0 else match["second_innings_team"]
    bowling = match["second_innings_team"] if innings_index == 0 else match["first_innings_team"]
    return matchup_grid(batting, bowling, as_of=match["date"], exclude_match_id=match_id)


@app.get("/api/model/evaluation")
async def model_evaluation():
    result = load_data().get("backtest", {})
    if not result:
        raise HTTPException(status_code=503, detail="Backtest results are not present. Run scripts/evaluate_model.py.")
    return result


@app.get("/api/model/ground-report")
async def ground_report(venue: str | None = None):
    grounds = load_data().get("grounds", {})
    if venue:
        if venue not in grounds: raise HTTPException(status_code=404, detail="No eligible ground report is available for that venue.")
        return {"venue": venue, **grounds[venue]}
    return {"grounds": grounds}


@app.get("/api/provenance")
async def provenance():
    # Do not expose the final chase score through a side endpoint before the replay
    # cursor reaches it. Internal startup/test reconciliation still checks terminal totals.
    internal = source_reconciliation_summary()
    public_summary = {
        key: internal[key]
        for key in ("match_id", "source_file", "data_version", "chasing_team", "target", "event_records_in_chase")
    }
    return {
        "reconciliation": public_summary,
        "terminal_result_visibility": "Final score and winner are returned only by /api/hero/replay at the final source-event cursor.",
        "dataset": "Cricsheet IPL JSON",
        "data_license": "Open Data Commons Attribution License (ODC-By 1.0)",
        "license_url": "https://cricsheet.org/licensing/",
        "dataset_url": "https://cricsheet.org/downloads/ipl_json.zip",
        "format_documentation": "https://cricsheet.org/format/json/",
        "official_match_url": HERO_REPLAY.OFFICIAL_MATCH_URL,
        "official_report_url": HERO_REPLAY.OFFICIAL_REPORT_URL,
        "scope": "The optional FastAPI interface serves one source-verified hero chase. The native Streamlit entrypoint offers the 15-match replay catalog, historical model lookup, strict K-of-N display gates, matchup grid, pre-match context and model evidence. K-of-N is shown only when its per-query conditions pass.",
        "eligible_reconciled_matches": load_data()["manifest"]["eligible_reconciled_matches"],
        "replay_catalog_count": load_data()["manifest"]["replay_match_count"],
    }


@app.exception_handler(HTTPException)
async def safe_http_error_handler(request: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, str) else "The request could not be completed."
    return JSONResponse(status_code=exc.status_code, content={"detail": detail}, headers=exc.headers or {})
