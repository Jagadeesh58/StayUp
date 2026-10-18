"""Canonical cricket state engine and source-traced IPL replay for StayUp Cricket.

Facts displayed by the UI are computed here from validated input or from the
prefix of the Cricsheet delivery sequence. A replay state never consumes a
later delivery or the eventual result until the cursor reaches the final event.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

BALLS_PER_OVER = 6
SUPPORTED_OVERS_LIMIT = 20
RETIRED_HURT = {"retired hurt"}
NON_LEGAL_BALL_EXTRAS = {"wides", "noballs"}
DATA_FILE = Path(__file__).parent / "data" / "1359487.json"


class CricketInputError(ValueError):
    """Raised when a submitted cricket state is unsupported or invalid."""


def _plain_int(name: str, value: Any, minimum: int, maximum: int) -> int:
    # bool subclasses int in Python; it is not an acceptable numeric input here.
    if isinstance(value, bool) or not isinstance(value, int):
        raise CricketInputError(f"{name} must be a whole number.")
    if not minimum <= value <= maximum:
        raise CricketInputError(f"{name} must be between {minimum} and {maximum}.")
    return value


def parse_overs_notation(value: str, overs_limit: int = SUPPORTED_OVERS_LIMIT) -> int:
    """Parse cricket notation as overs + legal balls, never as a decimal float."""
    if not isinstance(value, str) or not value.strip():
        raise CricketInputError("Enter overs as notation such as 14.2.")
    parts = value.strip().split(".")
    if len(parts) == 1:
        if not parts[0].isdigit():
            raise CricketInputError("Overs must be a whole number or notation such as 14.2.")
        overs, balls = int(parts[0]), 0
    elif len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        overs, balls = int(parts[0]), int(parts[1])
    else:
        raise CricketInputError("Use overs.balls notation such as 14.2.")
    if not 0 <= balls < BALLS_PER_OVER:
        raise CricketInputError(f"The balls part must be 0–5; {overs}.{balls} is invalid.")
    if overs < 0 or overs > overs_limit or (overs == overs_limit and balls > 0):
        raise CricketInputError(f"The state cannot exceed {overs_limit} completed overs.")
    return overs * BALLS_PER_OVER + balls


def _rate(runs: int, balls: int) -> float | None:
    if balls <= 0:
        return None
    return round((runs * BALLS_PER_OVER) / balls, 2)


def format_overs(legal_balls: int) -> str:
    """Format legal balls using cricket notation; not decimal arithmetic."""
    return f"{legal_balls // BALLS_PER_OVER}.{legal_balls % BALLS_PER_OVER}"


def state_from_totals(
    score: int,
    wickets_lost: int,
    legal_balls_used: int,
    target: int,
    overs_limit: int = SUPPORTED_OVERS_LIMIT,
) -> dict[str, Any]:
    """Return the canonical computed state for already validated match totals."""
    runs_required = max(0, target - score)
    scheduled_balls = overs_limit * BALLS_PER_OVER
    scheduled_balls_remaining = max(0, scheduled_balls - legal_balls_used)
    balls_remaining = scheduled_balls_remaining
    if score >= target:
        status = "target_reached"
        headline = "Target reached. The chase is complete."
    elif wickets_lost >= 10:
        status = "all_out"
        headline = "All out before reaching the target."
    elif legal_balls_used >= scheduled_balls:
        status = "overs_complete"
        headline = f"The 20-over limit is reached, with {runs_required} runs still required."
    else:
        status = "active"
        if scheduled_balls_remaining > 0:
            required_rate = (runs_required * BALLS_PER_OVER) / scheduled_balls_remaining
        else:
            required_rate = None
        if required_rate is not None and required_rate >= 14:
            headline = f"{runs_required} runs from {balls_remaining} balls: the required rate is very high."
        elif required_rate is not None and required_rate >= 10:
            headline = f"{runs_required} runs from {balls_remaining} balls: the chase needs a strong scoring rate."
        else:
            headline = f"{runs_required} runs from {balls_remaining} balls remain in the chase."
    return {
        "score": score,
        "wickets_lost": wickets_lost,
        "wickets_remaining": max(0, 10 - wickets_lost),
        "target": target,
        "runs_required": runs_required,
        "legal_balls_used": legal_balls_used,
        "overs_display": format_overs(legal_balls_used),
        # A completed chase/all-out/overs-complete innings has no playable balls left,
        # even when the scheduled limit had not yet been used.
        "balls_remaining": scheduled_balls_remaining if status == "active" else 0,
        "scheduled_overs": overs_limit,
        "current_run_rate": _rate(score, legal_balls_used),
        "required_run_rate": _rate(runs_required, scheduled_balls_remaining) if status == "active" else None,
        "status": status,
        "headline": headline,
    }


def calculate_manual_state(
    score: int,
    wickets_lost: int,
    completed_overs: int,
    balls_in_over: int,
    target: int,
    overs_limit: int = SUPPORTED_OVERS_LIMIT,
) -> dict[str, Any]:
    score = _plain_int("Score", score, 0, 1000)
    wickets_lost = _plain_int("Wickets lost", wickets_lost, 0, 10)
    completed_overs = _plain_int("Completed overs", completed_overs, 0, overs_limit)
    balls_in_over = _plain_int("Balls in the current over", balls_in_over, 0, 5)
    target = _plain_int("Target", target, 1, 1001)
    if overs_limit != SUPPORTED_OVERS_LIMIT:
        raise CricketInputError("Only a full 20-over T20 chase is supported in this version.")
    if completed_overs == overs_limit and balls_in_over != 0:
        raise CricketInputError("A 20-over innings cannot have extra balls recorded after the limit.")
    legal_balls_used = completed_overs * BALLS_PER_OVER + balls_in_over
    return state_from_totals(score, wickets_lost, legal_balls_used, target, overs_limit)


def build_scenario(state: dict[str, Any], name: str, runs: int, wickets_lost: int) -> dict[str, Any]:
    """Evaluate an aggregate six-legal-ball hypothetical from one immutable base."""
    if state["status"] != "active":
        raise CricketInputError("Scenarios are available only during an active chase.")
    if state["balls_remaining"] < BALLS_PER_OVER:
        raise CricketInputError("A full-over scenario needs at least six legal balls remaining.")
    if name not in {"steady", "big_over", "setback"}:
        raise CricketInputError("Unknown scenario.")
    next_wickets = state["wickets_lost"] + wickets_lost
    after_score = state["score"] + runs
    after_balls = state["legal_balls_used"] + BALLS_PER_OVER
    result: dict[str, Any] = {
        "id": name,
        "name": {"steady": "Steady over", "big_over": "Big over", "setback": "Setback"}[name],
        "assumed_runs": runs,
        "assumed_wickets_lost": wickets_lost,
        "base_state": state.copy(),
        "assumption": "Hypothetical, not a prediction. Six legal balls; no ball-by-ball sequence is simulated.",
    }
    # The plan deliberately treats any 10th-wicket aggregate as indeterminate,
    # even when the total runs appear to cross the target.
    if next_wickets >= 10:
        result.update({
            "status": "undetermined",
            "headline": "Outcome undetermined from an over summary.",
            "after_state": None,
            "comparison_claims": ["terminal_ambiguity"],
        })
        return result
    if after_score >= state["target"]:
        result.update({
            "status": "target_could_be_reached",
            "headline": "The chase could be completed during this hypothetical over; the exact finishing ball and terminal score cannot be inferred from the over total.",
            "after_state": None,
            "comparison_claims": ["target_may_be_reached"],
        })
        return result
    after = state_from_totals(after_score, next_wickets, after_balls, state["target"], state["scheduled_overs"])
    result.update({
        "status": after["status"],
        "headline": (
            f"At the end of the assumed over: {after['score']}/{after['wickets_lost']}, "
            f"{after['runs_required']} needed from {after['balls_remaining']} balls."
            if after["status"] == "active" else after["headline"]
        ),
        "after_state": after,
        "comparison_claims": ["runs_required", "balls_remaining", "wickets_remaining", "required_run_rate"],
    })
    return result


@dataclass(frozen=True)
class ReplayEvent:
    sequence: int
    source_id: str
    source_delivery_label: str
    over_index: int
    delivery_index: int
    batter: str
    bowler: str
    non_striker: str
    batter_runs: int
    total_runs: int
    extras: dict[str, int]
    wickets: tuple[dict[str, Any], ...]
    raw: dict[str, Any]

    @property
    def legal(self) -> bool:
        return not any(key in self.extras for key in NON_LEGAL_BALL_EXTRAS)

    @property
    def team_wickets(self) -> int:
        return sum(1 for wicket in self.wickets if str(wicket.get("kind", "")).lower() not in RETIRED_HURT)

    def public(self) -> dict[str, Any]:
        extra_label = None
        if self.extras:
            extra_label = ", ".join(f"{k.replace('noballs', 'no-balls').replace('wides', 'wides')} +{v}" for k, v in self.extras.items())
        return {
            "sequence": self.sequence,
            "source_id": self.source_id,
            "source_delivery_label": self.source_delivery_label,
            "over_index": self.over_index,
            "batter": self.batter,
            "bowler": self.bowler,
            "non_striker": self.non_striker,
            "batter_runs": self.batter_runs,
            "total_runs": self.total_runs,
            "extras": self.extras,
            "extra_label": extra_label,
            "wickets": list(self.wickets),
            "legal_ball": self.legal,
        }


class HeroReplay:
    """Verified source-traced chase. Only the prefix state is returned to clients."""

    MATCH_ID = "1359487"
    SOURCE_FILE = "1359487.json"
    OFFICIAL_MATCH_URL = "https://www.iplt20.com/matches/a955a189-b359-4fcd-972f-04ceb7af3a11/gujarat-titans-vs-kolkata-knight-riders/match-details"
    OFFICIAL_REPORT_URL = "https://www.iplt20.com/news/article/tata-ipl-2023-match-13-gt-vs-kkr-match-report"

    def __init__(self, data_file: Path = DATA_FILE):
        self.data_file = Path(data_file)
        self.raw_data = json.loads(self.data_file.read_text(encoding="utf-8"))
        self.meta = self.raw_data.get("meta", {})
        self.info = self.raw_data["info"]
        self.chasing_innings_index = 1
        innings = self.raw_data["innings"][self.chasing_innings_index]
        self.chasing_innings = innings
        self.target = int(innings["target"]["runs"])
        self.overs_limit = int(innings["target"].get("overs", self.info.get("overs", 20)))
        self.chase_team = innings["team"]
        self.batting_first_team = self.raw_data["innings"][0]["team"]
        first_innings = self.raw_data["innings"][0]
        self.first_innings_score = sum(int(d.get("runs", {}).get("total", 0)) for o in first_innings.get("overs", []) for d in o.get("deliveries", []))
        self.first_innings_wickets = sum(1 for o in first_innings.get("overs", []) for d in o.get("deliveries", []) for w in d.get("wickets", []) if str(w.get("kind", "")).lower() not in RETIRED_HURT)
        self.events = self._flatten_events(innings)
        self.turning_points = self._build_turning_points()
        self._validate_source_basics()

    def _flatten_events(self, innings: dict[str, Any]) -> list[ReplayEvent]:
        events: list[ReplayEvent] = []
        for over_position, over in enumerate(innings.get("overs", [])):
            over_index = int(over["over"])
            # Do not infer ball labels from floats; keep the source's string value.
            for delivery_position, delivery in enumerate(over.get("deliveries", [])):
                sequence = len(events) + 1
                source_id = f"{self.SOURCE_FILE}#/innings/{self.chasing_innings_index}/overs/{over_position}/deliveries/{delivery_position}"
                events.append(ReplayEvent(
                    sequence=sequence,
                    source_id=source_id,
                    source_delivery_label=str(delivery.get("actual_delivery", f"{over_index}.{delivery_position + 1}")),
                    over_index=over_index,
                    delivery_index=delivery_position,
                    batter=str(delivery.get("batter", "Unknown batter")),
                    bowler=str(delivery.get("bowler", "Unknown bowler")),
                    non_striker=str(delivery.get("non_striker", "Unknown non-striker")),
                    batter_runs=int(delivery.get("runs", {}).get("batter", 0)),
                    total_runs=int(delivery.get("runs", {}).get("total", 0)),
                    extras={str(k): int(v) for k, v in delivery.get("extras", {}).items()},
                    wickets=tuple(dict(w) for w in delivery.get("wickets", [])),
                    raw=delivery,
                ))
        return events

    def _validate_source_basics(self) -> None:
        if len(self.raw_data.get("innings", [])) < 2:
            raise ValueError("Hero replay source does not contain a chase innings.")
        outcome = self.info.get("outcome", {})
        if self.info.get("teams") is None or len(self.info["teams"]) != 2:
            raise ValueError("Hero replay source has unexpected team metadata.")
        if self.target != 205 or self.chase_team != "Kolkata Knight Riders":
            raise ValueError("The supplied hero file no longer matches the reviewed match specification.")
        final_state = self.state_at(len(self.events))
        if final_state["score"] != 207 or final_state["wickets_lost"] != 7 or final_state["legal_balls_used"] != 120:
            raise ValueError("Hero chase totals do not reconcile with hand-checked scorecard values.")
        if outcome.get("winner") != self.chase_team:
            raise ValueError("Hero chase result does not reconcile with source match outcome metadata.")

    def _event_by_over_label(self, over_index: int, label: str) -> ReplayEvent:
        matches = [e for e in self.events if e.over_index == over_index and e.source_delivery_label == label]
        if len(matches) != 1:
            raise ValueError(f"Expected one delivery at source over {over_index}, label {label}; got {len(matches)}.")
        return matches[0]

    def _build_turning_points(self) -> list[dict[str, Any]]:
        # These moments were selected after inspecting the supplied source and
        # checking the official IPL match report. The card content is not sent
        # to the client until its end cursor has been reached.
        hat_trick = [self._event_by_over_label(16, f"16.{n}") for n in (1, 2, 3)]
        if not all(e.team_wickets == 1 for e in hat_trick):
            raise ValueError("The wicket-sequence evidence for the first turning point has changed.")
        pre_last_two = [self._event_by_over_label(18, "18.5"), self._event_by_over_label(18, "18.6")]
        if [e.total_runs for e in pre_last_two] != [6, 4] or not all(e.batter == "RK Singh" for e in pre_last_two):
            raise ValueError("The penultimate-over turning-point evidence no longer reconciles.")
        last_over = [self._event_by_over_label(19, f"19.{n}") for n in range(1, 7)]
        if [e.total_runs for e in last_over] != [1, 6, 6, 6, 6, 6] or last_over[0].batter != "UT Yadav" or not all(e.batter == "RK Singh" for e in last_over[1:]):
            raise ValueError("The final-over turning-point evidence no longer reconciles.")
        return [
            {
                "id": "rashid-hat-trick",
                "title": "Three wickets in three balls",
                "kind": "wickets_and_rate",
                "start_cursor": hat_trick[0].sequence - 1,
                "end_cursor": hat_trick[-1].sequence,
                "event_ids": [e.source_id for e in hat_trick],
                "observation": "Three consecutive deliveries removed Andre Russell, Sunil Narine and Shardul Thakur. The chase moved from 155/4 to 155/7 across that sequence.",
                "selection_reason": "A directly checkable three-wicket burst that sharply changed the number of wickets available.",
            },
            {
                "id": "penultimate-boundaries",
                "title": "Six, then four to reset the equation",
                "kind": "rate_changed",
                "start_cursor": pre_last_two[0].sequence - 1,
                "end_cursor": pre_last_two[-1].sequence,
                "event_ids": [e.source_id for e in pre_last_two],
                "observation": "Rinku Singh scored 10 from the final two balls of the penultimate over: a six followed by a four. The innings state then stood at 176/7, with 29 required from six legal balls.",
                "selection_reason": "The final two balls produce a verifiable before/after equation immediately before the last over.",
            },
            {
                "id": "five-sixes-finish",
                "title": "Five sixes close the chase",
                "kind": "rate_changed",
                "start_cursor": last_over[0].sequence - 1,
                "end_cursor": last_over[-1].sequence,
                "event_ids": [e.source_id for e in last_over],
                "observation": "After Umesh Yadav's single, Rinku Singh hit five consecutive sixes. The replay only reveals the final score and result once these deliveries have all been reached.",
                "selection_reason": "The delivery sequence and final result reconcile with the official IPL report and scorecard.",
            },
        ]

    @property
    def total_events(self) -> int:
        return len(self.events)

    def state_at(self, cursor: int) -> dict[str, Any]:
        if isinstance(cursor, bool) or not isinstance(cursor, int) or not 0 <= cursor <= len(self.events):
            raise CricketInputError(f"Replay cursor must be between 0 and {len(self.events)}.")
        score = 0
        wickets = 0
        legal = 0
        for event in self.events[:cursor]:
            score += event.total_runs
            wickets += event.team_wickets
            legal += 1 if event.legal else 0
        state = state_from_totals(score, min(10, wickets), legal, self.target, self.overs_limit)
        state["cursor"] = cursor
        state["total_events"] = len(self.events)
        state["is_complete"] = cursor == len(self.events)
        return state

    def _event_context(self, cursor: int) -> dict[str, Any] | None:
        if cursor <= 0:
            return None
        event = self.events[cursor - 1]
        pub = event.public()
        pub["state_before"] = self.state_at(cursor - 1)
        pub["state_after"] = self.state_at(cursor)
        return pub

    def _card_public(self, card: dict[str, Any]) -> dict[str, Any]:
        before = self.state_at(card["start_cursor"])
        after = self.state_at(card["end_cursor"])
        return {
            **{k: v for k, v in card.items() if k not in {"selection_reason"}},
            "before_state": before,
            "after_state": after,
            "evidence": [self.events[i].public() for i in range(len(self.events)) if self.events[i].source_id in card["event_ids"]],
            "source": {
                "dataset": "Cricsheet IPL JSON",
                "match_id": self.MATCH_ID,
                "source_file": self.SOURCE_FILE,
                "json_data_version": str(self.meta.get("data_version", "unknown")),
                "official_match_url": self.OFFICIAL_MATCH_URL,
                "official_report_url": self.OFFICIAL_REPORT_URL,
            },
        }

    def public_meta(self) -> dict[str, Any]:
        # Deliberately omit the outcome and final totals until the replay is complete.
        return {
            "match_id": self.MATCH_ID,
            "match_label": "IPL 2023 · Match 13",
            "date": self.info.get("dates", ["Unknown date"])[0],
            "venue": self.info.get("venue", "Venue not supplied"),
            "batting_first_team": self.batting_first_team,
            "first_innings_score": self.first_innings_score,
            "first_innings_wickets": self.first_innings_wickets,
            "chase_team": self.chase_team,
            "target": self.target,
            "overs_limit": self.overs_limit,
            "total_events": self.total_events,
            "source_data_version": str(self.meta.get("data_version", "unknown")),
            "source_file": self.SOURCE_FILE,
            "source_notice": "Historical source: Cricsheet IPL JSON. The bundled match file retains its original JSON data version and source metadata; no independent download timestamp is asserted.",
            "license_url": "https://cricsheet.org/licensing/",
            "official_match_url": self.OFFICIAL_MATCH_URL,
            "official_report_url": self.OFFICIAL_REPORT_URL,
        }

    def public_at(self, cursor: int) -> dict[str, Any]:
        state = self.state_at(cursor)
        current_event = self._event_context(cursor)
        visible_cards = [self._card_public(card) for card in self.turning_points if card["end_cursor"] <= cursor]
        payload: dict[str, Any] = {
            "state": state,
            "current_event": current_event,
            "visible_turning_points": visible_cards,
            "progress_percent": round(100 * cursor / max(1, self.total_events), 1),
        }
        if state["is_complete"]:
            outcome = self.info.get("outcome", {})
            by = outcome.get("by", {})
            margin = f"{by['wickets']} wickets" if "wickets" in by else f"{by['runs']} runs" if "runs" in by else "the recorded result"
            payload["final_result"] = {
                "winner": outcome.get("winner"),
                "margin": margin,
                "final_score": f"{state['score']}/{state['wickets_lost']}",
                "headline": f"{outcome.get('winner')} won by {margin}.",
            }
        return payload


HERO_REPLAY = HeroReplay()


def source_reconciliation_summary() -> dict[str, Any]:
    replay = HERO_REPLAY
    last = replay.state_at(replay.total_events)
    return {
        "match_id": replay.MATCH_ID,
        "source_file": replay.SOURCE_FILE,
        "data_version": str(replay.meta.get("data_version", "unknown")),
        "chasing_team": replay.chase_team,
        "target": replay.target,
        "final_score": f"{last['score']}/{last['wickets_lost']}",
        "legal_balls": last["legal_balls_used"],
        "event_records_in_chase": replay.total_events,
        "reconciles": last["score"] == 207 and last["wickets_lost"] == 7 and last["legal_balls_used"] == 120,
    }
