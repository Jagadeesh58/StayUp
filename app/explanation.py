"""Deterministic explanation rendering, focus selection, and hypothetical scenarios."""
from __future__ import annotations

from typing import Any

from .cricket import build_scenario
from .localization import localized_explanation


def _fmt_rate(value: float | None) -> str:
    """Format run rate with two decimals, or indicate undefined."""
    return "not defined" if value is None else f"{value:.2f} runs per over"


def scenario_map(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build canonical hypothetical aggregate scenarios for an active chase state."""
    return {
        "steady": build_scenario(state, "steady", 6, 0),
        "big_over": build_scenario(state, "big_over", 12, 0),
        "setback": build_scenario(state, "setback", 3, 1),
    }


def _fallback_focus(intent: str, context: dict[str, Any]) -> tuple[str, str, list[str]]:
    """Choose a deterministic, evidence-complete focus when live AI is unavailable."""
    if intent == "current_state_context":
        state = context["state"]
        if state.get("current_run_rate") is not None and state.get("required_run_rate") is not None:
            return "current_state_context", "rate_pressure", ["runs_required", "balls_remaining", "current_run_rate", "required_run_rate"]
        return "current_state_context", "runway", ["runs_required", "balls_remaining"]
    if intent == "compare_scenarios":
        first, second = context["scenarios"]
        focus_claims = [f"{first['id']}_runs_required", f"{second['id']}_runs_required"]
        return "scenario_difference", "runs_difference", focus_claims
    if intent == "explain_chase_change":
        return "target_reached", "target_reached", list(context["claims"])
    card = context["card"]
    before, after = card["before_state"], card["after_state"]
    if before.get("required_run_rate") is not None and after.get("required_run_rate") is not None:
        if before.get("wickets_lost") != after.get("wickets_lost"):
            return "wickets_and_rate", "wicket_rate_shift", ["before_wickets_lost", "after_wickets_lost", "before_required_rate", "after_required_rate"]
        return "rate_changed", "rate_shift", ["before_required_rate", "after_required_rate"]
    return "current_state_context", "runway", ["before_runs_required", "before_balls_remaining"]


def render_explanation(intent: str, focus_id: str, context: dict[str, Any], language: str = "English") -> dict[str, Any]:
    """Render deterministic fallback wording from computed and source-verified facts."""
    if language != "English":
        return localized_explanation(intent, focus_id, context, language)
    if intent == "current_state_context":
        state = context["state"]
        r, b, w = state["runs_required"], state["balls_remaining"], state["wickets_remaining"]
        if focus_id == "rate_pressure" and state.get("current_run_rate") is not None and state.get("required_run_rate") is not None:
            gap = round(state["required_run_rate"] - state["current_run_rate"], 2)
            direction = "above" if gap >= 0 else "below"
            return {"answer": f"The chase needs {state['runs_required']} runs from {state['balls_remaining']} legal balls. The current scoring rate is {_fmt_rate(state['current_run_rate'])}; the required rate is {_fmt_rate(state['required_run_rate'])}. Required pace is {abs(gap):.2f} runs per over {direction} the current rate. Rates describe the equation, not a prediction.", "relationship_id": "current_state_context"}
        if focus_id == "wickets_buffer":
            return {"answer": f"There are {w} wickets remaining while {r} runs are needed from {b} legal balls. Wickets in hand describe the dismissal margin available; they do not guarantee how the chase will finish.", "relationship_id": "current_state_context"}
        return {"answer": f"The chase needs {r} runs from {b} legal balls, with {w} wickets remaining. That is the current equation; this manually entered score does not tell us what happened earlier or what will happen next.", "relationship_id": "current_state_context"}

    if intent == "compare_scenarios":
        first, second = context["scenarios"]
        a_name, b_name = first["name"], second["name"]
        a_state, b_state = first["after_state"], second["after_state"]
        if focus_id == "required_rate_difference":
            gap = round((b_state["required_run_rate"] or 0) - (a_state["required_run_rate"] or 0), 2)
            delta = abs(gap)
            direction = "higher" if gap > 0 else "lower" if gap < 0 else "the same"
            return {"answer": f"After the assumed overs, {a_name.lower()} leaves a required rate of {_fmt_rate(a_state['required_run_rate'])}; {b_name.lower()} leaves {_fmt_rate(b_state['required_run_rate'])}. The second scenario is {direction} by {delta:.2f} runs per over. Both are aggregate hypotheticals, not forecasts.", "relationship_id": "scenario_difference"}
        if focus_id == "wickets_difference":
            return {"answer": f"The {a_name.lower()} assumption leaves {a_state['wickets_remaining']} wickets; the {b_name.lower()} assumption leaves {b_state['wickets_remaining']}. This compares only the supplied six-ball summaries, not a predicted sequence.", "relationship_id": "scenario_difference"}
        delta_runs = a_state["runs_required"] - b_state["runs_required"]
        if delta_runs > 0:
            comparison = f"That is {delta_runs} fewer runs needed under the second assumption."
        elif delta_runs < 0:
            comparison = f"That is {abs(delta_runs)} more runs needed under the second assumption."
        else:
            comparison = "Both assumptions leave the same runs required."
        return {"answer": f"The {a_name.lower()} assumption leaves {a_state['runs_required']} runs needed; the {b_name.lower()} assumption leaves {b_state['runs_required']}. {comparison} Both outcomes are hypothetical six-legal-ball totals, not predictions.", "relationship_id": "scenario_difference"}

    card = context["card"]
    before = card["before_state"]
    after = card["after_state"]
    if intent == "explain_chase_change" and focus_id == "target_reached":
        claims = context["claims"]
        return {"answer": f"The final over began with {claims['before_runs_required']} runs needed from {claims['before_balls_remaining']} legal balls. The {claims['recorded_deliveries']} recorded deliveries added {claims['runs_added']} runs, taking the score to {claims['after_score']} against a target of {claims['target']}. The source sequence reaches the target on the last recorded delivery. This explains the score equation; it does not infer player intent or a win probability.", "relationship_id": "target_reached"}
    if focus_id == "wicket_rate_shift":
        return {"answer": f"Across this verified sequence, wickets lost moved from {before['wickets_lost']} to {after['wickets_lost']}, while the required rate moved from {_fmt_rate(before['required_run_rate'])} to {_fmt_rate(after['required_run_rate'])}. Those two state changes quantify the pressure shift; the data does not establish intent or guarantee an outcome.", "relationship_id": "wickets_and_rate"}
    if focus_id == "rate_shift":
        return {"answer": f"The required rate changed from {_fmt_rate(before['required_run_rate'])} to {_fmt_rate(after['required_run_rate'])} across the selected deliveries. This is a computed before/after comparison from the source-backed replay, not a claim about what would happen next.", "relationship_id": "rate_changed"}
    return {"answer": f"Before this verified moment, {before['runs_required']} runs were needed from {before['balls_remaining']} balls. After it, {after['runs_required']} runs were needed from {after['balls_remaining']} balls. The source shows the change in the equation but cannot establish player intent.", "relationship_id": "current_state_context"}
