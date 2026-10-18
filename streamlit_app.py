"""StayUp Cricket — Streamlit deployment entry point.

This is a native Streamlit interface over the same deterministic cricket and
historical-model services used by the FastAPI API. It does not spawn a FastAPI
subprocess; Streamlit Community Cloud can run this file as the app entry point.
"""
from __future__ import annotations
import logging
import os
from datetime import date, datetime, timezone
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

logger = logging.getLogger("stayup.streamlit")

# Streamlit Cloud secrets are exposed through st.secrets, not necessarily the
# process environment. Copy all named runtime values into os.environ.
try:
    if hasattr(st, "secrets"):
        for _name in ("GROQ_API_KEY", "AI_ENABLED", "AI_DAILY_BUDGET", "AI_BURST_PER_MINUTE", "APP_DB_PATH", "RATE_LIMIT_SALT"):
            if _name in st.secrets:
                os.environ[_name] = str(st.secrets[_name]).strip()
            elif _name.lower() in st.secrets:
                os.environ[_name] = str(st.secrets[_name.lower()]).strip()
        for _k, _v in st.secrets.items():
            if isinstance(_v, (str, int, float, bool)):
                os.environ[str(_k).upper()] = str(_v).strip()
            elif isinstance(_v, dict) or hasattr(_v, "items"):
                for _sub_k, _sub_v in _v.items():
                    if isinstance(_sub_v, (str, int, float, bool)):
                        os.environ[str(_sub_k).upper()] = str(_sub_v).strip()
except Exception:
    # A local secrets file is optional. Missing configuration is reported below.
    pass

from app.ai_service import AIProviderError, ai_configuration, select_focus
from app.cricket import CricketInputError, calculate_manual_state
from app.localization import LANGUAGES
from app.model import (
    ModelDataError, build_replay_view, load_data, lookup_chase,
    match_index_public, matchup_grid, next_over_outcomes,
    pre_match_info, replay_matches, similar_chase_cohort,
)
from app.quick_entry import parse_quick_entry
from app.presentation import chase_verdict
from app.theme import DARK_PALETTE, LIGHT_PALETTE
from app.budget import reserve_streamlit_ai_budget
from app.explanation import _fallback_focus, render_explanation

st.set_page_config(page_title="StayUp Cricket", page_icon=None, layout="wide", initial_sidebar_state="expanded")
dark_mode = st.sidebar.toggle("Dark editorial mode", value=False, help="Switch contrast-safe light/dark reading themes.")
palette = DARK_PALETTE if dark_mode else LIGHT_PALETTE
css = """<style>
:root { --stay-ink:__INK__; --stay-paper:__PAPER__; --stay-lime:__LIME__; --stay-line:__LINE__; --stay-muted:__MUTED__; --stay-control:__CONTROL__; --stay-note:__NOTE__; --stay-primary:__PRIMARY__; --stay-primary-text:__PRIMARY_TEXT__; }
.stApp { background:var(--stay-paper); color:var(--stay-ink); }
[data-testid="stHeader"] { background:__HEADER__; }
[data-testid="stSidebar"], [data-testid="stSidebarContent"] { background:var(--stay-paper); }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span { color:var(--stay-ink); }
[data-testid="stMainBlockContainer"] { max-width:1180px; padding-top:1.4rem; }
html, body, [class*="css"] { font-family:Arial,Helvetica,sans-serif; }
h1,h2,h3 { color:var(--stay-ink); letter-spacing:-.035em; }
h1 { font-family:Georgia,'Times New Roman',serif; font-weight:600; font-size:clamp(2.3rem,5vw,4.4rem); line-height:1.0; }
h2 { font-family:Georgia,'Times New Roman',serif; font-size:clamp(1.7rem,3vw,2.5rem); }
h3 { font-size:1.15rem; font-weight:700; }
p, label, li, [data-testid="stMarkdownContainer"] { color:var(--stay-ink); }
[data-testid="stMetric"] { background:transparent; border:0; border-top:1px solid var(--stay-line); padding:12px 0 6px 0; border-radius:0; }
[data-testid="stMetricLabel"] p { font-size:.72rem; letter-spacing:.11em; text-transform:uppercase; color:var(--stay-muted); }
[data-testid="stMetricValue"] { font-family:Consolas,'Courier New',monospace; color:var(--stay-ink); }
[data-testid="stTabs"] button { font-weight:600; }
[data-testid="stTabs"] button[aria-selected="true"] { color:var(--stay-ink); border-bottom-color:var(--stay-lime); }
.stButton button[kind="primary"] { background:var(--stay-primary); color:var(--stay-primary-text); border:1px solid var(--stay-primary); border-radius:2px; font-weight:600; }
.stButton button { border-radius:2px; }
button:focus-visible, input:focus-visible, textarea:focus-visible, [role="slider"]:focus-visible, [role="tab"]:focus-visible { outline:3px solid #82964a !important; outline-offset:2px !important; }
.stTextInput input, .stNumberInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] { border-radius:2px; color:var(--stay-ink); background:var(--stay-control); }
.stAlert { border-radius:2px; }
.stCaption, small, [data-testid="stCaptionContainer"] { color:var(--stay-muted); }
hr { border-color:var(--stay-line); }
.overline { font-family:Consolas,'Courier New',monospace; font-size:.72rem; letter-spacing:.13em; text-transform:uppercase; color:var(--stay-muted); }
.equation { font-family:Consolas,'Courier New',monospace; font-size:clamp(1.5rem,5vw,2.8rem); font-weight:500; letter-spacing:-.05em; padding:.8rem 0 1.1rem 0; border-bottom:1px solid var(--stay-line); overflow-wrap:anywhere; }
.verdict { font-family:Georgia,'Times New Roman',serif; font-size:clamp(2rem,4vw,3.2rem); line-height:1.05; margin:.15rem 0 .5rem 0; }
.data-note { border-left:3px solid #82964a; padding:.6rem .9rem; background:var(--stay-note); color:var(--stay-ink); margin:.4rem 0 1rem 0; }
.source-id { font-family:Consolas,'Courier New',monospace; font-size:.78rem; overflow-wrap:anywhere; }
[data-testid="stDataFrame"], [data-testid="stTable"] { max-width:100%; }
[data-testid="stAppViewContainer"] { overflow-x:hidden; }
@media (max-width: 500px) { [data-testid="stMainBlockContainer"] { padding: .8rem .85rem 2rem .85rem; } h1 { font-size:2.35rem; } [data-testid="stHorizontalBlock"] { gap:.6rem; } }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { scroll-behavior:auto !important; animation:none !important; transition:none !important; } }
</style>"""
for key, value in palette.items():
    css = css.replace(f"__{key}__", value)
st.markdown(css, unsafe_allow_html=True)


def fmt_rate(value: Any) -> str:
    return "not defined" if value is None else f"{float(value):.2f}"


def pct(value: Any) -> str:
    return "Not enough historical support" if value is None else f"{float(value)*100:.1f}%"


def safe_model_lookup(score: int, wickets: int, balls: int, target: int, *, as_of: str | date | None = None, exclude_match_id: str | None = None) -> dict[str, Any]:
    try: return lookup_chase(score,wickets,balls,target,as_of=as_of,exclude_match_id=exclude_match_id)
    except ModelDataError as exc: return {"probability":None,"effective_sample_size":0,"note":str(exc),"source":"unavailable"}


def _get_client_token() -> str:
    """Derive client key from request headers when available, falling back to session token."""
    ip = ""
    try:
        headers = getattr(st.context, "headers", {})
        if headers:
            ip = headers.get("x-forwarded-for", "").split(",")[0].strip() or headers.get("host", "").strip()
    except Exception:
        pass
    if ip:
        return f"ip:{ip}"
    if "ai_session" not in st.session_state:
        st.session_state.ai_session = os.urandom(16).hex()
    return f"session:{st.session_state.ai_session}"


def run_state_ai(state: dict[str, Any], language: str) -> tuple[dict[str, Any], str, str]:
    claims = {
        "runs_required": state["runs_required"],
        "balls_remaining": state["balls_remaining"],
        "wickets_remaining": state["wickets_remaining"],
        "required_run_rate": state["required_run_rate"],
        "current_run_rate": state["current_run_rate"],
    }
    claim_ids = [k for k, v in claims.items() if v is not None]
    focuses = ["runway", "wickets_buffer"]
    requirements = {
        "runway": {"current_state_context": {"runs_required", "balls_remaining"}},
        "wickets_buffer": {"current_state_context": {"wickets_remaining", "runs_required"}},
    }
    if claims["current_run_rate"] is not None and claims["required_run_rate"] is not None:
        focuses.append("rate_pressure")
        requirements["rate_pressure"] = {"current_state_context": {"current_run_rate", "required_run_rate"}}
    context = {"intent": "current_state_context", "state": state, "claims": claims, "claim_ids": claim_ids}
    _, focus, selected = _fallback_focus("current_state_context", context)
    source = "calculated"
    selection = None
    last_error: str | None = None
    config = ai_configuration()
    if config["enabled"] and config["configured"]:
        try:
            token = _get_client_token()
            selection = select_focus(
                context={"intent": "current_state_context", "claims": claims, "source_ids": [], "language": language},
                relationship_ids=["current_state_context"],
                focus_ids=focuses,
                claim_ids=claim_ids,
                requirements=requirements,
                on_live_call=lambda: reserve_streamlit_ai_budget(token),
            )
            focus, selected = selection.focus_id, selection.selected_claim_ids
            source = selection.source
        except (AIProviderError, RuntimeError) as exc:
            logger.warning("Streamlit AI fallback triggered: %s", exc)
            source = "calculated"
            last_error = str(exc)
    rendered = render_explanation("current_state_context", focus, context, language)
    if source in {"live", "cached"} and selection and selection.explanation.strip():
        rendered["answer"] = selection.explanation
    if source == "live":
        label = f"Live AI explanation · {config['model']}"
    elif source == "cached":
        label = f"Cached AI explanation · {config['model']}"
    elif not config["enabled"]:
        label = "Calculated explanation · Live AI is disabled (set AI_ENABLED = 'true' in Secrets)"
    elif not config["configured"]:
        label = "Calculated explanation · GROQ_API_KEY is missing or empty in Secrets"
    elif last_error:
        label = f"Calculated explanation · Provider issue: {last_error}"
    else:
        label = "Calculated explanation · AI unavailable, disabled or not configured"
    rendered.update({"focus_id": focus, "selected_claim_ids": selected, "source": source})
    return rendered, source, label


def run_replay_point_ai(point: dict[str, Any], match_id: str, language: str) -> tuple[dict[str, Any], str, str]:
    """Explain a reached chase over using only its computed before/after state."""
    before = point["before_state"]
    after = point["after_state"]
    card = {"before_state": before, "after_state": after}
    source_ids = list(point.get("source_event_ids", []))

    if after.get("status") == "target_reached":
        intent = "explain_chase_change"
        claims = {
            "before_runs_required": before["runs_required"],
            "before_balls_remaining": before["balls_remaining"],
            "runs_added": int(point["runs_added"]),
            "recorded_deliveries": int(point["recorded_deliveries"]),
            "after_score": int(after["score"]),
            "target": int(after["target"]),
        }
        claim_ids = list(claims)
        relationships = ["target_reached"]
        focuses = ["target_reached"]
        requirements = {"target_reached": {"target_reached": set(claim_ids)}}
        relation, focus = "target_reached", "target_reached"
    else:
        intent = "explain_turning_point"
        claims = {
            "before_required_rate": before.get("required_run_rate"),
            "after_required_rate": after.get("required_run_rate"),
            "before_wickets_lost": int(before["wickets_lost"]),
            "after_wickets_lost": int(after["wickets_lost"]),
            "before_runs_required": int(before["runs_required"]),
            "after_runs_required": int(after["runs_required"]),
            "before_balls_remaining": int(before["balls_remaining"]),
            "after_balls_remaining": int(after["balls_remaining"]),
        }
        claim_ids = [key for key, value in claims.items() if value is not None]
        relationships = ["current_state_context"]
        focuses = ["before_after"]
        requirements = {"before_after": {"current_state_context": {"before_runs_required", "before_balls_remaining", "after_runs_required", "after_balls_remaining"}}}
        relation, focus = "current_state_context", "before_after"
        has_rates = claims["before_required_rate"] is not None and claims["after_required_rate"] is not None
        if has_rates:
            relationships.append("rate_changed")
            focuses.append("rate_shift")
            requirements["rate_shift"] = {"rate_changed": {"before_required_rate", "after_required_rate"}}
            relation, focus = "rate_changed", "rate_shift"
            if claims["before_wickets_lost"] != claims["after_wickets_lost"]:
                relationships.append("wickets_and_rate")
                focuses.append("wicket_rate_shift")
                requirements["wicket_rate_shift"] = {"wickets_and_rate": {"before_wickets_lost", "after_wickets_lost", "before_required_rate", "after_required_rate"}}
                relation, focus = "wickets_and_rate", "wicket_rate_shift"

    context = {"intent": intent, "card": card, "claims": claims, "claim_ids": claim_ids}
    selected_claim_ids = sorted(requirements.get(focus, {}).get(relation, set()))
    source = "calculated"
    selection = None
    last_error: str | None = None
    config = ai_configuration()
    if config["enabled"] and config["configured"]:
        try:
            if "ai_session" not in st.session_state:
                st.session_state.ai_session = os.urandom(16).hex()
            provider_context = {
                "intent": intent,
                "match_id": str(match_id),
                "over": int(point["over"]),
                "source_event_ids": source_ids,
                "claims": claims,
                "language": language,
            }
            selection = select_focus(
                context=provider_context,
                relationship_ids=relationships,
                focus_ids=focuses,
                claim_ids=claim_ids,
                requirements=requirements,
                on_live_call=lambda: reserve_streamlit_ai_budget(_get_client_token()),
            )
            relation, focus = selection.relationship_id, selection.focus_id
            selected_claim_ids = selection.selected_claim_ids
            source = selection.source
        except (AIProviderError, RuntimeError) as exc:
            source = "calculated"
            last_error = str(exc)
    rendered = render_explanation(intent, focus, context, language)
    if source in {"live", "cached"} and selection is not None and selection.explanation.strip():
        rendered["answer"] = selection.explanation
    if source == "live":
        label = f"Live AI explanation · {config['model']}"
    elif source == "cached":
        label = f"Cached AI explanation · {config['model']}"
    elif not config["enabled"]:
        label = "Calculated explanation · Live AI is disabled (set AI_ENABLED = 'true' in Secrets)"
    elif not config["configured"]:
        label = "Calculated explanation · GROQ_API_KEY is missing or empty in Secrets"
    elif last_error:
        label = f"Calculated explanation · Provider issue: {last_error}"
    else:
        label = "Calculated explanation · AI unavailable, disabled or not configured"
    rendered.update({"focus_id": focus, "relationship_id": relation, "selected_claim_ids": selected_claim_ids, "source": source})
    return rendered, source, label


def render_state_section() -> None:
    st.markdown('<div class="overline">01 / Manual chase · no live feed</div>',unsafe_allow_html=True)
    st.title("Is it still alive?")
    st.write("Enter the current scoreboard to see whether the chase is still alive, both teams’ historical win chances, and observed next-over outcomes from comparable prior IPL situations. No live feed or guarantee.")
    if "quick_entry_result" not in st.session_state:
        try: st.session_state.quick_entry_result = parse_quick_entry("112/3 13.2 t189")
        except CricketInputError: st.session_state.quick_entry_result = None
        st.session_state.quick_entry_text = "112/3 13.2 t189"
        st.session_state.field_score = 112
        st.session_state.field_wickets = 3
        st.session_state.field_overs = 13
        st.session_state.field_balls = 2
        st.session_state.field_target = 189

    base = st.session_state.get("quick_entry_result") or {"score": 112, "wickets_lost": 3, "completed_overs": 13, "balls_in_over": 2, "target": 189}

    entry_style = st.radio(
        "Scoreboard entry mode",
        ["Scoreboard fields", "Quick entry notation (e.g. 112/3 13.2 t189)"],
        horizontal=True,
        key="scoreboard_entry_style",
        help="Select whether to adjust individual scoreboard numbers or enter compact match notation.",
    )

    if entry_style == "Scoreboard fields":
        with st.form("scoreboard-fields-form"):
            score_row = st.columns(2)
            score = score_row[0].number_input("Score", min_value=0, max_value=1000, value=int(st.session_state.get("field_score", base.get("score", 112))), key="input_field_score")
            wkts = score_row[1].number_input("Wickets", min_value=0, max_value=10, value=int(st.session_state.get("field_wickets", base.get("wickets_lost", 3))), key="input_field_wickets")
            over_row = st.columns(2)
            overs = over_row[0].number_input("Completed overs", min_value=0, max_value=20, value=int(st.session_state.get("field_overs", base.get("completed_overs", 13))), key="input_field_overs")
            balls = over_row[1].number_input("Balls in over", min_value=0, max_value=5, value=int(st.session_state.get("field_balls", base.get("balls_in_over", 2))), key="input_field_balls")
            target = st.number_input("Target", min_value=1, max_value=1001, value=int(st.session_state.get("field_target", base.get("target", 189))), key="input_field_target")
            submitted = st.form_submit_button("Read this chase", type="primary", use_container_width=False)
        if submitted:
            try:
                state = calculate_manual_state(int(score), int(wkts), int(overs), int(balls), int(target))
                st.session_state.quick_entry_result = {
                    "score": int(score), "wickets_lost": int(wkts), "completed_overs": int(overs),
                    "balls_in_over": int(balls), "target": int(target), "state": state,
                    "overs_notation": f"{overs}.{balls}"
                }
                new_quick = f"{score}/{wkts} {overs}.{balls} t{target}"
                st.session_state.quick_entry_text = new_quick
                st.session_state.input_quick_entry_text = new_quick
                st.session_state.field_score = int(score)
                st.session_state.field_wickets = int(wkts)
                st.session_state.field_overs = int(overs)
                st.session_state.field_balls = int(balls)
                st.session_state.field_target = int(target)
                st.session_state.pop("state_explanation", None)
                st.rerun()
            except CricketInputError as exc:
                st.error(str(exc))
    else:
        with st.form("quick-entry-form"):
            quick = st.text_input(
                "Score / wickets · overs · target",
                value=st.session_state.get("quick_entry_text", "112/3 13.2 t189"),
                help="Format: 112/3 13.2 t189 — overs are cricket notation, not decimals.",
                key="input_quick_entry_text",
            )
            submitted = st.form_submit_button("Read this chase", type="primary", use_container_width=False)
        if submitted:
            try:
                parsed = parse_quick_entry(quick)
                st.session_state.quick_entry_result = parsed
                st.session_state.quick_entry_text = quick
                st.session_state.field_score = parsed["score"]
                st.session_state.field_wickets = parsed["wickets_lost"]
                st.session_state.field_overs = parsed["completed_overs"]
                st.session_state.field_balls = parsed["balls_in_over"]
                st.session_state.field_target = parsed["target"]
                st.session_state.input_field_score = parsed["score"]
                st.session_state.input_field_wickets = parsed["wickets_lost"]
                st.session_state.input_field_overs = parsed["completed_overs"]
                st.session_state.input_field_balls = parsed["balls_in_over"]
                st.session_state.input_field_target = parsed["target"]
                st.session_state.pop("state_explanation", None)
                st.rerun()
            except CricketInputError as exc:
                st.error(f"{exc} Tip: Switch to 'Scoreboard fields' above to enter individual scores directly.")

    parsed = st.session_state.get("quick_entry_result")
    if not parsed:
        return
    state = parsed["state"]
    if state["status"] != "active":
        st.warning(state["headline"])
        return
    model = safe_model_lookup(state["score"], state["wickets_lost"], state["legal_balls_used"], state["target"], as_of=datetime.now(timezone.utc).date())
    p = model.get("probability")
    st.markdown(f'<div class="equation">{state["score"]}/{state["wickets_lost"]} · {state["overs_display"]} overs <span style="color:var(--stay-muted)">→</span> {state["runs_required"]} needed / {state["balls_remaining"]} balls</div>', unsafe_allow_html=True)
    st.markdown('<div class="overline">Chase verdict</div>', unsafe_allow_html=True)
    verdict_text = chase_verdict(p)
    verdict_headline = f"{verdict_text} · {pct(p)}" if p is not None else verdict_text
    st.markdown(f'<div class="verdict">{verdict_headline}</div>', unsafe_allow_html=True)
    st.caption("Based on historical IPL states; never a guarantee. Verdict bands are presentation labels: chasing side favoured at ≥65%, close from 35% to under 65%, defending side favoured below 35%.")
    chance_row = st.columns(2)
    chance_row[0].metric("Chasing side", pct(p))
    chance_row[1].metric("Defending side", pct(1 - p if p is not None else None))
    st.metric("Effective historical sample size", f"≈ {model.get('effective_sample_size', 0):.0f} weighted states" if p is not None else "—")
    if p is not None:
        st.progress(max(0.0, min(1.0, float(p))), text="Historical estimate for the chasing side · not a live probability feed")
    st.caption(f"{model.get('note', 'Historical model output unavailable')} Effective sample size accounts for recency weights; it is not a raw count of independent matches.")
    st.markdown("**The equation**")
    st.write(f"{state['runs_required']} runs from {state['balls_remaining']} legal balls · {state['wickets_remaining']} wickets remaining · current rate {fmt_rate(state['current_run_rate'])} · required rate {fmt_rate(state['required_run_rate'])} runs per over.")
    try:
        live_cohort = similar_chase_cohort(
            state["score"], state["wickets_lost"], state["legal_balls_used"], state["target"],
            as_of=datetime.now(timezone.utc).date(), exclude_match_id=None,
        )
    except ModelDataError as exc:
        live_cohort = {"available": False, "reason": f"Similar-match comparison is unavailable: {exc}"}
    if live_cohort.get("available"):
        st.markdown(f"**K of N similar prior matches:** {live_cohort['k']} of {live_cohort['n']} won the chase.")
        st.caption("Distinct completed IPL matches strictly before today; one closest checkpoint per match, with wickets ±1, legal balls ±6 and required rate ±1.0. This is a historical frequency, not personal odds. Manual entry has no match ID: use this for a current match, and use Replay for old scorecards so the source match can be excluded.")
    else:
        st.caption(live_cohort.get("reason", "The similar-match count is hidden because the minimum sample requirement was not met."))

    try:
        outcomes = next_over_outcomes(state["score"], state["wickets_lost"], state["legal_balls_used"], state["target"], as_of=datetime.now(timezone.utc).date())
    except ModelDataError:
        outcomes = {"available": False, "reason": "Historical outcome data is unavailable."}
    st.subheader("What did similar chases do in the next over?")
    st.caption("These are observed next-over outcomes from the frozen similar-state cohort, not fixed guesses or a prediction of your next over.")
    if outcomes.get("available"):
        qc = st.columns(4)
        qc[0].metric("Similar prior matches", outcomes["n"])
        qc[1].metric("Lower quartile", f"{outcomes['lower_quartile_runs']} runs")
        qc[2].metric("Median", f"{outcomes['median_runs']} runs")
        qc[3].metric("Upper quartile", f"{outcomes['upper_quartile_runs']} runs")
        bucket_df = pd.DataFrame([{"Observed next-over total": k, "Share": v["rate"]} for k, v in outcomes["run_buckets"].items()]).set_index("Observed next-over total")
        st.bar_chart(bucket_df)
        st.write(f"A wicket fell in {outcomes['wicket_over_rate']*100:.1f}% of matched next overs; at least one four or six occurred in {outcomes['boundary_over_rate']*100:.1f}%. Values are descriptive frequencies, not individualized probabilities.")
    else:
        st.info(outcomes.get("reason", "Not enough comparable prior situations."))

    st.subheader("Explain the equation")
    language = st.selectbox("Explanation language", LANGUAGES, index=0, key="live_language")
    cached_exp = st.session_state.get("state_explanation")
    state_sig = (state["score"], state["wickets_lost"], state["legal_balls_used"], state["target"], language)
    if not cached_exp or cached_exp.get("state_sig") != state_sig:
        if state["status"] == "active":
            explanation, source, label = run_state_ai(state, language)
            cached_exp = {"answer": explanation["answer"], "source": source, "label": label, "language": language, "state_sig": state_sig}
            st.session_state.state_explanation = cached_exp
    if st.button("Refresh explanation", key="explain-live"):
        if state["status"] == "active":
            explanation, source, label = run_state_ai(state, language)
            cached_exp = {"answer": explanation["answer"], "source": source, "label": label, "language": language, "state_sig": state_sig}
            st.session_state.state_explanation = cached_exp
    if cached_exp:
        st.markdown(cached_exp["answer"])
        st.caption(cached_exp["label"])


def _match_label(m: dict[str, Any]) -> str:
    return f"{m['season']} · {m['first_innings_team']} vs {m['second_innings_team']} · {m['venue']}"


def _safe_over_dataframe(view: dict[str, Any]) -> pd.DataFrame:
    rows=[]
    for p in view.get("runs_by_over",[]):
        inning="First innings" if p["innings_index"]==0 else "Chase"
        rows.append({"Innings":inning,"Over":p["over"],"Batting side":p["team"],"Runs":p["runs_this_over"],"Wickets marker":("W"*int(p["wickets_this_over"])) or "—","Six marker":("6 × "+str(p["sixes_this_over"])) if p["sixes_this_over"] else "—","Four marker":("4 × "+str(p["fours_this_over"])) if p["fours_this_over"] else "—","Score":f"{p['score']}/{p['wickets_lost']}","Win chance · batting side":pct(p["batting_team_win_probability"]),"Total swing (pp)":p["effect"]["total_swing_pp"],"Runs-only swing (pp)":p["effect"]["runs_only_swing_pp"],"Wicket counterfactual (pp)":p["effect"]["wicket_counterfactual_pp"],"Boundary counterfactual (pp)":p["effect"]["boundary_counterfactual_pp"],"Source records":len(p["source_event_ids"])})
    return pd.DataFrame(rows)


def render_replay_section() -> None:
    st.markdown('<div class="overline">02 / Historical replay · no spoilers</div>',unsafe_allow_html=True)
    st.title("Replay the evidence")
    st.write("Move through real Cricsheet deliveries. Scorecards, future over outcomes, turning points and the result stay hidden until their evidence has been reached.")
    catalog=match_index_public()
    if not catalog:
        st.error("No replay catalog is available. Rebuild the data files from the Cricsheet archive."); return
    by_id={str(m["match_id"]):m for m in catalog}
    ids=[str(m["match_id"]) for m in catalog]
    hero_id = "1359487"
    default_index = ids.index(hero_id) if hero_id in ids else 0
    selected=st.selectbox("Choose a historical replay",ids,index=default_index,format_func=lambda mid:_match_label(by_id[mid]),key="replay_match_choice")
    replay_record=next(m for m in replay_matches() if str(m["match_id"])==selected)
    maximum=len(replay_record["events"])
    cursor=st.slider("Source delivery cursor",min_value=0,max_value=maximum,value=0,step=1,key=f"replay_cursor_{selected}",help="The final result remains hidden until the cursor reaches the final source record.")
    view=build_replay_view(selected,cursor)
    meta=view["match"]
    st.markdown(f"### {meta['first_innings_team']} vs {meta['second_innings_team']}")
    st.caption(f"{meta['season']} · {meta['date']} · {meta['venue']} · Cricsheet match {selected} · source records reached {cursor}/{maximum}")
    if view["target"] is None:
        st.info("The first innings is still in progress. The chase target and all second-innings evidence are withheld until the first innings has ended.")
    else:
        st.metric("Chase target",view["target"])
    active=view["current_state"]
    c=st.columns(2)
    c[0].metric("Current innings",active["team"])
    c[1].metric("Score at cursor",f"{active['score']}/{active['wickets_lost']}")
    c=st.columns(2)
    c[0].metric("Legal overs",active["overs_display"])
    c[1].metric("Delivery sequence",f"{cursor} / {maximum}")
    if cursor and view["events"]:
        ev=view["events"][-1]
        st.markdown(f"**Reached delivery:** {ev['source_delivery_label']} · {ev['batter']} vs {ev['bowler']} · {ev['total_runs']} total runs · {', '.join(w['kind'] for w in ev['wickets']) if ev['wickets'] else 'no wicket'}")
        with st.expander("Inspect exact source evidence for the reached delivery"):
            st.code(ev["source_id"],language=None)
            st.json({k:ev[k] for k in ("source_id","source_delivery_label","batter","bowler","batter_runs","total_runs","extras","wickets","legal_ball","state_after")})
            st.markdown("Source: [Cricsheet IPL JSON archive](https://cricsheet.org/downloads/ipl_json.zip) · [Format documentation](https://cricsheet.org/format/json/)")
    else:
        st.caption("No delivery has been reached yet. Nothing beyond the first source record is shown.")

    points=view.get("win_probability_chart",[])
    st.subheader("Win chance by over")
    if points:
        chart=[]
        for pnt in points:
            phase="First innings" if pnt["innings_index"]==0 else "Chase"
            label=f"{phase} · {pnt['over']}"
            chart.append({"Checkpoint":label,meta["first_innings_team"]:pnt["first_team_probability"]*100 if pnt["first_team_probability"] is not None else None,meta["second_innings_team"]:pnt["second_team_probability"]*100 if pnt["second_team_probability"] is not None else None})
        chart_df = pd.DataFrame(chart).set_index("Checkpoint").reset_index().melt(
            id_vars=["Checkpoint"], var_name="Team", value_name="Win chance (%)"
        )
        probability_chart = (
            alt.Chart(chart_df)
            .mark_line(point=alt.OverlayMarkDef(filled=True, size=48))
            .encode(
                x=alt.X("Checkpoint:N", sort=None, title="Reached checkpoint"),
                y=alt.Y("Win chance (%):Q", scale=alt.Scale(domain=[0, 100], clamp=True),
                        axis=alt.Axis(title="Historical win chance (%)", values=[0, 25, 50, 75, 100])),
                color=alt.Color("Team:N", title="Team"),
                tooltip=["Checkpoint:N", "Team:N", alt.Tooltip("Win chance (%):Q", format=".1f")],
            )
            .properties(height=280)
        )
        st.altair_chart(probability_chart, use_container_width=True)
        st.caption("Estimates use only matches dated strictly before the replayed match and exclude the replay match itself. The line is a historical lookup, not causal attribution.")
    else: st.info("Win-chance points appear only after a full over or terminal state has been reached and prior historical support exists.")

    st.subheader("Runs by over · wicket and boundary markers")
    overs_df=_safe_over_dataframe(view)
    if not overs_df.empty:
        run_rows=overs_df.assign(Checkpoint=overs_df["Innings"]+" · "+overs_df["Over"].astype(str)).set_index("Checkpoint")[["Runs"]]
        st.bar_chart(run_rows)
        st.dataframe(overs_df[["Innings","Over","Batting side","Runs","Wickets marker","Six marker","Four marker","Score"]],use_container_width=True,hide_index=True)
        st.subheader("Descriptive per-over probability change")
        st.dataframe(overs_df[["Innings","Over","Batting side","Total swing (pp)","Runs-only swing (pp)","Wicket counterfactual (pp)","Boundary counterfactual (pp)","Source records"]],use_container_width=True,hide_index=True)
        st.caption("Counterfactual components overlap; they are not independent causal effects. A boundary counterfactual removes recorded boundary runs while keeping other observed state fields fixed.")
    else:
        st.info("Run-by-over data appears as completed overs are reached.")

    st.subheader("Largest reached turning points")
    any_turn = False
    for ii, label in ((0, "First innings"), (1, "Chase")):
        turns = view.get("turning_points", {}).get(ii, [])
        if turns:
            any_turn = True
            st.markdown(f"**{label}**")
            st.dataframe(pd.DataFrame([{"Over": t["over"], "Swing (pp)": t["swing_pp"], "Score": f"{t['score']}/{t['wickets_lost']}", "Runs": t["runs_this_over"], "Wickets": t["wickets_this_over"], "Fours": t["fours_this_over"], "Sixes": t["sixes_this_over"], "Source event IDs": "; ".join(t["source_event_ids"])} for t in turns]), use_container_width=True, hide_index=True)
    if not any_turn:
        st.caption("Turning points are not selected until their supporting over evidence is reached.")

    # The model can discuss only an already-reached chase over. The available
    # cards are derived from this cursor's visible prefix; future cards are absent.
    chase_points = [p for p in view.get("turning_points", {}).get(1, []) if "before_state" in p and "after_state" in p and p["after_state"].get("status") in {"active", "target_reached"}]
    if chase_points:
        st.subheader("Explain a reached chase turning point")
        st.caption("AI selects a grounded relationship from source-linked evidence. The server renders the wording and all numeric facts; unavailable AI uses a labelled calculated explanation.")
        chosen_point = st.selectbox(
            "Reached turning point",
            options=range(len(chase_points)),
            format_func=lambda i: f"Over {chase_points[i]['over']} · {chase_points[i]['runs_this_over']} runs · {chase_points[i]['wickets_this_over']} wickets · {chase_points[i]['swing_pp']:+.1f} percentage points",
            key=f"replay_ai_point_{selected}_{cursor}",
        )
        replay_language = st.selectbox("Explanation language", LANGUAGES, key=f"replay_ai_language_{selected}_{cursor}")
        if st.button("Explain this reached moment", key=f"explain_replay_{selected}_{cursor}", type="primary"):
            explanation, source, label = run_replay_point_ai(chase_points[chosen_point], selected, replay_language)
            st.session_state[f"replay_ai_explanation_{selected}_{cursor}"] = {"answer": explanation["answer"], "label": label, "source": source, "cursor": cursor}
        saved = st.session_state.get(f"replay_ai_explanation_{selected}_{cursor}")
        if saved and saved.get("cursor") == cursor:
            st.markdown(saved["answer"])
            st.caption(saved["label"])

    st.subheader("Partnerships reached so far")
    for ii, label in ((0, "First innings"), (1, "Chase")):
        parts = view["partnerships"].get(ii, [])
        if parts:
            st.markdown(f"**{label}**")
            d = pd.DataFrame([{"Partnership": f"{x['partnership']} · {' & '.join(x['batters'])}", "Runs": x["runs"], "Wickets in segment": x["wickets"]} for x in parts])
            st.bar_chart(d.set_index("Partnership"))
    st.subheader("Batters still to bat")
    b1, b2 = st.columns(2)
    b1.markdown(f"**{meta['first_innings_team']}**")
    b1.write(", ".join(view["batters_still_to_bat"].get(meta["first_innings_team"], [])) or "No unobserved roster names remain in the supplied list.")
    b2.markdown(f"**{meta['second_innings_team']}**")
    b2.write(", ".join(view["batters_still_to_bat"].get(meta["second_innings_team"], [])) or ("Not yet at the chase / no unobserved roster names remain." if cursor < maximum else "No unobserved roster names remain in the supplied list."))

    if view.get("current_win_model"):
        st.subheader("Similar historical states")
        wm = view["current_win_model"]
        st.write(f"Current chase estimate: {pct(wm.get('probability'))}; effective N ≈ {wm.get('effective_sample_size', 0):.0f}. {wm.get('note', '')}")
        cohort = view.get("cohort", {})
        if cohort.get("available"):
            st.markdown(f"**K of N:** {cohort['k']} of {cohort['n']} similar prior matches won the chase under the frozen rule.")
        else:
            st.caption(cohort.get("reason", "Strict K-of-N cohort not available at this cursor."))
        nxt = view.get("next_over_outcomes", {})
        if nxt.get("available"):
            st.write(f"Among {nxt['n']} comparable prior next overs: median {nxt['median_runs']} runs; wicket-in-over rate {nxt['wicket_over_rate']*100:.1f}%. These are historical frequencies, not predictions.")
        else:
            st.caption(nxt.get("reason", "No next-over sample is shown at this cursor."))
    if view["final_result"] is not None:
        res = view["final_result"]
        st.success(f"FINAL RESULT REVEALED · {res['winner']} won. {res['first_innings']['team']} {res['first_innings']['score']}/{res['first_innings']['wickets']} · {res['second_innings']['team']} {res['second_innings']['score']}/{res['second_innings']['wickets']}.")
    else:
        st.caption("Final scores, winner and result margin are withheld until all source records have been reached.")


def render_matchups_section() -> None:
    st.markdown('<div class="overline">03 / Historical matchup evidence</div>', unsafe_allow_html=True)
    st.title("Batter vs bowler")
    st.write("Historical IPL matchup records for the selected team pairing, cut off strictly before the selected match date. These are descriptive samples, not forecasts.")
    catalog = match_index_public()
    choices = [str(m["match_id"]) for m in catalog]
    selected = st.selectbox("Reference match and teams", choices, format_func=lambda mid: _match_label(next(m for m in catalog if str(m["match_id"]) == mid)), key="matchup_match")
    m = next(m for m in replay_matches() if m["match_id"] == selected)
    innings = st.radio("Batting side", [0, 1], format_func=lambda i: m["first_innings_team"] if i == 0 else m["second_innings_team"], horizontal=True, key="matchup_innings")
    batting = m["first_innings_team"] if innings == 0 else m["second_innings_team"]
    bowling = m["second_innings_team"] if innings == 0 else m["first_innings_team"]
    grid = matchup_grid(batting, bowling, as_of=m["date"], exclude_match_id=selected)
    if not grid["batters"] or not grid["bowlers"]:
        st.info("No prior matchup cells are available for this team pairing.")
        return
    matrix = []
    notes = []
    for row in grid["cells"]:
        out = {"Batter": row[0]["batter"] if row else "Unknown"}
        for cell in row:
            out[cell["bowler"]] = f"{cell['runs']} runs / {cell['balls']} balls · {cell['dismissals']} outs"
            notes.append({"Batter": cell["batter"], "Bowler": cell["bowler"], "Balls": cell["balls"], "Matches": cell["matches"], "Runs": cell["runs"], "Dismissals": cell["dismissals"], "Bowler-credited dismissals": cell["credited_dismissals"], "Note": cell["note"]})
        matrix.append(out)
    st.dataframe(pd.DataFrame(matrix).set_index("Batter"), use_container_width=True)
    st.caption(grid["note"])
    st.dataframe(pd.DataFrame(notes), use_container_width=True, hide_index=True)
    st.caption("A batter's dismissal in a matchup is not necessarily credited as a wicket to that bowler; both counts are shown separately.")


def render_prematch_section() -> None:
    st.markdown('<div class="overline">04 / Before the first ball</div>', unsafe_allow_html=True)
    st.title("Team form and ground history")
    st.write("Context is reconstructed only from earlier eligible IPL matches. Ground statistics describe past results, not a curator's report or a current pitch inspection.")
    catalog = match_index_public()
    choices = [str(m["match_id"]) for m in catalog]
    selected = st.selectbox("Reference match", choices, format_func=lambda mid: _match_label(next(m for m in catalog if str(m["match_id"]) == mid)), key="prematch_match")
    info = pre_match_info(selected)
    if not info:
        st.warning("Pre-match context is unavailable for this match.")
        return
    prem = info.get("pre_match") or {}
    team1, team2 = info["teams"]
    aliases = {"Delhi Daredevils": "Delhi Capitals", "Kings XI Punjab": "Punjab Kings", "Royal Challengers Bangalore": "Royal Challengers Bengaluru"}
    a, b = st.columns(2)
    for col, team in ((a, team1), (b, team2)):
        canonical = aliases.get(team, team)
        col.markdown(f"### {team}")
        form = prem.get("team_form", {}).get(canonical, [])
        col.write("Recent form: " + (" · ".join(f"{x['result']} vs {x['opponent']}" for x in form) if form else "No prior matches in this archive."))
        ground = prem.get("ground_record", {}).get(canonical, {})
        col.write(f"Prior record at this ground: {ground.get('wins', 0)} wins from {ground.get('matches', 0)} matches.")
    h = prem.get("head_to_head", {})
    st.subheader("Head-to-head before the match")
    st.write(f"{h.get('team_a', 'Team A')}: {h.get('team_a_wins', 0)} wins · {h.get('team_b', 'Team B')}: {h.get('team_b_wins', 0)} wins · {h.get('matches', 0)} prior meetings.")
    st.subheader("Toss")
    st.write(f"{info.get('toss', {}).get('winner', 'Unknown')} won the toss and chose to {info.get('toss', {}).get('decision', 'unknown')}.")
    st.subheader("Ground history since 2022")
    pitch = (prem.get("pitch_report") or {})
    if pitch:
        cols = st.columns(2)
        cols[0].metric("Prior matches", pitch.get("matches", 0))
        cols[1].metric("1st innings average", pitch.get("first_innings_average", "—"))
        cols = st.columns(2)
        cols[0].metric("Sixes per match", pitch.get("sixes_per_match", "—"))
        cols[1].metric("Wickets per innings", pitch.get("wickets_per_innings", "—"))
        st.write(f"Bat-first win rate: {100*pitch.get('bat_first_win_rate', 0):.1f}%. {pitch.get('note', '')}")
    else:
        st.info("No earlier 2022+ ground sample was available before this match. The app does not substitute later results.")


def render_model_evaluation() -> None:
    st.markdown('<div class="overline">05 / What the model has demonstrated</div>',unsafe_allow_html=True)
    st.title("How accurate is this?")
    st.write("This is a historical state lookup. The backtest trains on eligible IPL matches through 2023 and evaluates unseen 2024–2026 matches. It cannot guarantee a live match outcome.")
    try: ev=load_data().get("backtest",{})
    except ModelDataError: ev={}
    if not ev:
        st.warning("Backtest report is missing. Run `python scripts/evaluate_model.py`."); return
    pooled=ev["pooled"]; mod=pooled["historical_lookup"]; base=pooled["required_rate_only_baseline"]
    cols=st.columns(4)
    cols[0].metric("Test checkpoint states",mod["n"])
    cols[1].metric("Lookup Brier score",mod["brier"])
    cols[2].metric("Required-rate baseline",base["brier"])
    cols[3].metric("Lookup state accuracy",f"{mod['accuracy']*100:.1f}%")
    raw=pooled.get("raw_historical_lookup",{})
    prov=ev.get("provenance",{})
    st.markdown(f'<div class="data-note">On the held-out IPL 2024–2026 split, the calibrated chase lookup classified {mod["accuracy"]*100:.2f}% of {mod["n"]} checkpoint states correctly at a 0.5 threshold; these represent {prov.get("test_matches_in_date_window","—")} distinct test matches and multiple states per match. Brier score improved from {raw.get("brier","—")} raw to {mod["brier"]} after training-only calibration. This does not establish accuracy on other competitions or future seasons. First-innings probability is materially less calibrated on this split; see its separate results below.</div>',unsafe_allow_html=True)
    rows=[]
    for over,values in ev["checkpoints"].items():
        m=values["historical_lookup"]; b=values["required_rate_only_baseline"]
        rows.append({"Completed over":over,"Lookup N":m["n"],"Lookup Brier":m["brier"],"Lookup log loss":m["log_loss"],"Lookup accuracy":m["accuracy"],"Baseline N":b["n"],"Baseline Brier":b["brier"],"Baseline log loss":b["log_loss"],"Baseline accuracy":b["accuracy"],"Unavailable lookup":values["unavailable_model_predictions"]})
    st.subheader("Unseen-match results by checkpoint")
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    st.subheader("Calibration")
    st.write("A calibrated 60% estimate should correspond to an observed win rate near 60% over many comparable predictions. Empty bins remain empty rather than being filled with made-up results.")
    calibration_bins = ev["calibration"]["historical_lookup"]
    st.dataframe(pd.DataFrame(calibration_bins),use_container_width=True,hide_index=True)
    supported_bins = [row for row in calibration_bins if int(row.get("n", 0)) >= 25 and row.get("mean_predicted") is not None and row.get("observed_win_rate") is not None]
    if supported_bins:
        least_calibrated = max(supported_bins, key=lambda row: abs(float(row["observed_win_rate"]) - float(row["mean_predicted"])))
        st.warning(
            f"Calibration is imperfect: in the {least_calibrated['bin']} predicted band, the mean estimate was "
            f"{float(least_calibrated['mean_predicted'])*100:.1f}% while the observed win rate was "
            f"{float(least_calibrated['observed_win_rate'])*100:.1f}% across {int(least_calibrated['n'])} held-out checkpoint states. "
            "These states are not independent matches; treat individual probabilities as approximate historical context, not exact odds."
        )
    st.subheader("First-innings model check")
    first=ev.get("first_innings_pooled",{}).get("historical_lookup",{})
    st.write(f"First-innings lookup: Brier {first.get('brier','—')}, log loss {first.get('log_loss','—')}, accuracy {float(first.get('accuracy') or 0)*100:.1f}% over {first.get('n','—')} held-out state observations.")
    st.caption("First-innings calibration is weaker than the chase model in this split. Treat first-innings win chances as lower-confidence historical context, not a confident forecast.")
    first_rows=[]
    for over, values in ev.get("first_innings_checkpoints",{}).items():
        m=values["historical_lookup"]
        first_rows.append({"Over":over,"N":m["n"],"Brier":m["brier"],"Log loss":m["log_loss"],"Accuracy":m["accuracy"],"Unavailable":values["unavailable_model_predictions"]})
    st.dataframe(pd.DataFrame(first_rows),use_container_width=True,hide_index=True)
    st.caption("See docs/AI_EVALUATION.md for full protocol, per-checkpoint metrics, baseline and limitations. The test period covers IPL 2024–2026 from the supplied archive only.")


try:
    data=load_data()
    manifest=data["manifest"]
except ModelDataError as exc:
    st.title("StayUp Cricket")
    st.error(str(exc))
    st.code("python scripts/build_data.py --input /path/to/ipl_json.zip")
    st.stop()

st.markdown('<div class="overline">STAYUP CRICKET / AN EVIDENCE-LED T20 COMPANION</div>',unsafe_allow_html=True)
st.caption("Enter the score yourself. Historical examples only — no live feed.")
st.caption(f"Cricsheet IPL data · {manifest['eligible_reconciled_matches']:,} eligible reconciled matches · {manifest['replay_match_count']} spoiler-gated replays · ODC-By attribution included")
tabs=st.tabs(["Live chase", "Best replay", "More analysis"])
with tabs[0]:
    render_state_section()
with tabs[1]:
    render_replay_section()
with tabs[2]:
    st.write("The live chase and source-gated replay are the primary experience. Additional research panels live here for reviewers who want to inspect the supporting evidence.")
    analysis_tabs=st.tabs(["Matchups", "Pre-match & pitch", "How accurate is this?"])
    with analysis_tabs[0]: render_matchups_section()
    with analysis_tabs[1]: render_prematch_section()
    with analysis_tabs[2]: render_model_evaluation()

st.divider()
st.caption("Independent prototype. Not affiliated with Dream11, IPL, BCCI or Cricsheet. Historical IPL data is attributed to Cricsheet under ODC-By 1.0. No live feed, broadcast monitoring or prediction guarantee.")
