from datetime import date

import pytest

from app.model import (
    all_matches,
    build_replay_view,
    effective_sample_size,
    apply_logistic_calibration,
    lookup_first_innings,
    load_data,
    lookup_chase,
    partnership_segments,
    similar_chase_cohort,
    swing_percentage_points,
    _matching_cohort_rows,
)
from app.quick_entry import parse_quick_entry


def test_model_dataset_is_reconciled_and_has_multiseason_coverage():
    data = load_data()
    assert data["manifest"]["eligible_reconciled_matches"] >= 1000
    assert data["manifest"]["replay_match_count"] >= 12
    assert len({row["match_id"] for row in all_matches()}) == len(all_matches())
    assert "source_archive_sha256" in data["manifest"]


def test_effective_sample_size_respects_weights():
    assert effective_sample_size([1.0, 1.0]) == pytest.approx(2.0)
    assert effective_sample_size([1.0, 0.5]) == pytest.approx(1.8)
    assert effective_sample_size([]) == 0


def test_chase_lookup_is_bounded_and_strictly_as_of_date():
    result = lookup_chase(112, 3, 80, 189, as_of="2026-10-18")
    assert result["probability"] is None or 0 <= result["probability"] <= 1
    assert result["cutoff_exclusive"] == "2026-10-18"
    # A replay cutoff excludes matches on its date, even if many same-day rows exist.
    prior = lookup_chase(112, 3, 80, 189, as_of="2023-05-01", exclude_match_id="1359487")
    assert prior["cutoff_exclusive"] == "2023-05-01"
    assert all(row["date"] < "2023-05-01" for row in __import__("app.model", fromlist=["_prior_rows"])._prior_rows(2, date(2023, 5, 1), "1359487"))


def test_quick_entry_uses_cricket_over_notation():
    parsed = parse_quick_entry("112/3 13.2 t189")
    assert parsed["score"] == 112
    assert parsed["wickets_lost"] == 3
    assert parsed["state"]["legal_balls_used"] == 80
    assert parsed["state"]["runs_required"] == 77
    with pytest.raises(Exception):
        parse_quick_entry("112/3 13.6 t189")


def test_swing_is_signed_percentage_points_not_percent_change():
    assert swing_percentage_points(0.40, 0.45) == 5.0
    assert swing_percentage_points(0.45, 0.40) == -5.0
    assert swing_percentage_points(None, 0.45) is None


def test_partnership_math_assigns_wicket_delivery_runs_before_close():
    events = [
        {"sequence": 1, "innings_index": 0, "batter": "A", "non_striker": "B", "total_runs": 2, "wickets": [], "legal_ball": True},
        {"sequence": 2, "innings_index": 0, "batter": "A", "non_striker": "B", "total_runs": 1, "wickets": [{"kind": "bowled", "player_out": "B"}], "legal_ball": True},
        {"sequence": 3, "innings_index": 0, "batter": "A", "non_striker": "C", "total_runs": 4, "wickets": [], "legal_ball": True},
    ]
    parts = partnership_segments(events)[0]
    assert len(parts) == 2
    assert parts[0]["runs"] == 3 and parts[0]["wickets"] == 1
    assert parts[1]["runs"] == 4 and parts[1]["wickets"] == 0


def test_replay_hides_target_until_first_innings_prefix_is_complete():
    match = next(m for m in load_data()["replays"] if str(m["match_id"]) == "1359487")
    first_count = sum(1 for ev in match["events"] if ev["innings_index"] == 0)
    before = build_replay_view("1359487", first_count - 1)
    assert before["target"] is None
    assert before["final_result"] is None
    assert all(ev["innings_index"] == 0 for ev in before["events"])
    after_first = build_replay_view("1359487", first_count)
    assert after_first["target"] == match["first_innings"]["score"] + 1
    assert after_first["final_result"] is None


def test_replay_hides_final_result_until_exact_terminal_cursor():
    match = next(m for m in load_data()["replays"] if str(m["match_id"]) == "1359487")
    total = len(match["events"])
    pre_final = build_replay_view("1359487", total - 1)
    assert pre_final["final_result"] is None
    assert pre_final["complete"] is False
    final = build_replay_view("1359487", total)
    assert final["final_result"]["winner"] == match["winner"]
    assert final["complete"] is True
    assert len(final["events"]) == total


def test_k_of_n_is_hidden_or_above_the_minimum_and_excludes_replay_match():
    match = next(m for m in load_data()["replays"] if m["date"] >= "2024-01-01")
    chase = match["second_innings"]["over_summaries"]
    state = next((x for x in chase if x["over"] == 10 and x["score"] < match["first_innings"]["score"] + 1 and x["wickets_lost"] < 10), None)
    if state is None:
        pytest.skip("Selected replay has no active over-10 checkpoint")
    cohort = similar_chase_cohort(state["score"], state["wickets_lost"], state["legal_balls_used"], match["first_innings"]["score"] + 1, as_of=match["date"], exclude_match_id=match["match_id"])
    assert cohort["n"] >= 0
    if cohort["available"]:
        assert cohort["n"] >= cohort["min_n"] == 30
        assert 0 <= cohort["k"] <= cohort["n"]
    else:
        assert cohort["k"] is None
        assert "at least 30" in cohort["reason"] or "terminal" in cohort["reason"]


def test_logistic_calibrator_is_bounded_and_monotonic():
    params={"intercept":0.1,"slope":1.5}
    low=apply_logistic_calibration(0.2,params)
    mid=apply_logistic_calibration(0.5,params)
    high=apply_logistic_calibration(0.8,params)
    assert 0 < low < mid < high < 1
    assert apply_logistic_calibration(0.5,{"intercept":0.0,"slope":1.0}) == pytest.approx(0.5)

def test_runtime_chase_calibration_never_uses_future_calibrator_for_old_replay():
    newer=lookup_chase(112,3,80,189,as_of="2026-10-18")
    if newer.get("probability") is not None:
        assert newer["calibration_status"] == "temporal_platt"
        assert "raw_probability" in newer
        assert 0 <= newer["probability"] <= 1
    older=lookup_chase(112,3,80,189,as_of="2022-10-18")
    if older.get("probability") is not None:
        assert older["calibration_status"] == "raw_for_temporal_integrity"
        assert older["probability"] == older["raw_probability"]

def test_first_innings_lookup_reports_temporal_calibration_status():
    match=next(m for m in load_data()["matches"] if m["date"] >= "2024-01-01")
    state=next(x for x in match["first_innings"]["over_summaries"] if x["over"]==10 and x["legal_balls_used"]==60 and x["wickets_lost"]<10)
    pred=lookup_first_innings(state["score"],state["wickets_lost"],state["legal_balls_used"],as_of=match["date"],exclude_match_id=match["match_id"])
    if pred.get("probability") is not None:
        assert pred["calibration_status"] == "temporal_platt"
        assert pred["raw_probability"] is not None


def test_completed_replay_over_exposes_only_computed_chase_states_for_grounded_ai():
    match = next(m for m in load_data()["replays"] if str(m["match_id"]) == "1359487")
    total = len(match["events"])
    before_final_delivery = build_replay_view("1359487", total - 1)
    assert not any(p["innings_index"] == 1 and p["over"] == 20 for p in before_final_delivery["runs_by_over"])
    assert before_final_delivery["final_result"] is None
    assert all(p["over"] < 20 for p in before_final_delivery["runs_by_over"] if p["innings_index"] == 1)

    complete = build_replay_view("1359487", total)
    final_over = next(p for p in complete["runs_by_over"] if p["innings_index"] == 1 and p["over"] == 20)
    assert final_over["recorded_deliveries"] == 6
    assert final_over["runs_this_over"] == 31
    assert final_over["before_state"]["runs_required"] == 29
    assert final_over["before_state"]["balls_remaining"] == 6
    assert final_over["after_state"]["status"] == "target_reached"
    final_turn = next(p for p in complete["turning_points"][1] if p["over"] == 20)
    assert final_turn["source_event_ids"] == final_over["source_event_ids"]
    assert final_turn["runs_added"] == 31


def test_cohort_rows_exclude_same_day_future_and_replay_match():
    match = next(m for m in load_data()["replays"] if m["date"] >= "2024-01-01")
    chase = match["second_innings"]["over_summaries"]
    state = next(x for x in chase if x["over"] == 10 and x["score"] < match["first_innings"]["score"] + 1 and x["wickets_lost"] < 10)
    _reference, _rate, rows = _matching_cohort_rows(
        state["score"], state["wickets_lost"], state["legal_balls_used"],
        match["first_innings"]["score"] + 1,
        as_of=match["date"], exclude_match_id=match["match_id"],
    )
    assert all(row["date"] < match["date"] for row in rows)
    assert all(str(row["match_id"]) != str(match["match_id"]) for row in rows)
    assert len({str(row["match_id"]) for row in rows}) == len(rows)
