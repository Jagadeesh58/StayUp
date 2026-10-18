from copy import deepcopy
import pytest

from app.cricket import (
    CricketInputError,
    HERO_REPLAY,
    ReplayEvent,
    build_scenario,
    calculate_manual_state,
    parse_overs_notation,
    state_from_totals,
)


def test_overs_notation_is_not_float_math():
    assert parse_overs_notation("14.2") == 86
    assert parse_overs_notation("14") == 84
    with pytest.raises(CricketInputError, match="14.6 is invalid"):
        parse_overs_notation("14.6")
    with pytest.raises(CricketInputError):
        parse_overs_notation("14.10")
    with pytest.raises(CricketInputError, match="13.6 is invalid"):
        parse_overs_notation("13.6")


def test_manual_state_rates_and_remaining_balls():
    state = calculate_manual_state(150, 5, 16, 0, 205)
    assert state["runs_required"] == 55
    assert state["legal_balls_used"] == 96
    assert state["balls_remaining"] == 24
    assert state["current_run_rate"] == 9.38
    assert state["required_run_rate"] == 13.75
    assert state["wickets_remaining"] == 5
    assert state["status"] == "active"


def test_zero_ball_denominator_never_returns_current_rate():
    state = calculate_manual_state(0, 0, 0, 0, 205)
    assert state["current_run_rate"] is None
    assert state["required_run_rate"] == 10.25


def test_rejects_invalid_manual_values_and_unsupported_overs():
    with pytest.raises(CricketInputError):
        calculate_manual_state(150, 5, 14, 6, 205)
    with pytest.raises(CricketInputError, match="Only a full 20-over"):
        calculate_manual_state(50, 2, 5, 0, 170, overs_limit=10)
    with pytest.raises(CricketInputError):
        calculate_manual_state(-1, 2, 5, 0, 170)
    with pytest.raises(CricketInputError):
        calculate_manual_state(50, 2, 21, 0, 170)
    with pytest.raises(CricketInputError):
        calculate_manual_state(50, 2, 20, 1, 170)


def test_terminal_states_are_explicit():
    assert state_from_totals(205, 8, 100, 205)["status"] == "target_reached"
    assert state_from_totals(150, 10, 100, 205)["status"] == "all_out"
    overs_complete = state_from_totals(150, 7, 120, 205)
    assert overs_complete["status"] == "overs_complete"
    assert overs_complete["balls_remaining"] == 0
    assert overs_complete["required_run_rate"] is None


def test_wides_do_not_consume_legal_ball_and_stay_in_source_order():
    nonlegal = [e for e in HERO_REPLAY.events if not e.legal]
    assert len(nonlegal) == 3
    assert all(any(k in e.extras for k in {"wides", "noballs"}) for e in nonlegal)
    assert HERO_REPLAY.state_at(HERO_REPLAY.total_events)["legal_balls_used"] == 120
    assert HERO_REPLAY.total_events == 123


def test_retired_hurt_is_not_a_wicket_but_same_ball_real_dismissal_is():
    base = dict(
        sequence=1, source_id="fixture#/delivery/1", source_delivery_label="23.10",
        over_index=23, delivery_index=9, batter="Batter A", bowler="Bowler A",
        non_striker="Batter B", batter_runs=0, total_runs=0, extras={}, raw={},
    )
    retired_only = ReplayEvent(**base, wickets=({"player_out": "Batter A", "kind": "retired hurt"},))
    mixed = ReplayEvent(**base, wickets=(
        {"player_out": "Batter A", "kind": "retired hurt"},
        {"player_out": "Batter B", "kind": "caught"},
    ))
    assert retired_only.team_wickets == 0
    assert mixed.team_wickets == 1
    assert mixed.source_delivery_label == "23.10"


def test_scenarios_are_immutable_and_computed_from_same_base():
    state = calculate_manual_state(150, 5, 16, 0, 205)
    before = deepcopy(state)
    steady = build_scenario(state, "steady", 6, 0)
    big = build_scenario(state, "big_over", 12, 0)
    assert state == before
    assert steady["after_state"]["runs_required"] == 49
    assert big["after_state"]["runs_required"] == 43
    assert steady["after_state"]["legal_balls_used"] == big["after_state"]["legal_balls_used"] == 102
    assert steady["after_state"]["balls_remaining"] == big["after_state"]["balls_remaining"] == 18
    assert steady["assumption"].startswith("Hypothetical, not a prediction")


def test_scenario_tenth_wicket_is_always_undetermined_even_if_runs_reach_target():
    state = calculate_manual_state(149, 9, 19, 0, 150)
    result = build_scenario(state, "setback", 3, 1)
    assert result["status"] == "undetermined"
    assert result["headline"] == "Outcome undetermined from an over summary."
    assert result["after_state"] is None


def test_scenario_target_crossing_does_not_invent_finishing_ball_or_score():
    state = calculate_manual_state(199, 3, 19, 0, 205)
    result = build_scenario(state, "big_over", 12, 0)
    assert result["status"] == "target_could_be_reached"
    assert result["after_state"] is None
    assert "exact finishing ball" in result["headline"]


def test_scenarios_require_six_balls_and_active_state():
    late_state = calculate_manual_state(180, 5, 19, 5, 205)
    with pytest.raises(CricketInputError, match="at least six legal balls"):
        build_scenario(late_state, "steady", 6, 0)
    terminal = calculate_manual_state(205, 5, 19, 5, 205)
    with pytest.raises(CricketInputError, match="active chase"):
        build_scenario(terminal, "steady", 6, 0)


def test_hero_checkpoints_reconcile_and_turning_points_are_source_traced():
    assert HERO_REPLAY.total_events == 123
    assert HERO_REPLAY.state_at(97)["score"] == 155
    assert HERO_REPLAY.state_at(97)["wickets_lost"] == 4
    assert HERO_REPLAY.state_at(97)["legal_balls_used"] == 96
    assert HERO_REPLAY.state_at(100)["score"] == 155
    assert HERO_REPLAY.state_at(100)["wickets_lost"] == 7
    assert HERO_REPLAY.state_at(100)["required_run_rate"] == 14.29
    assert HERO_REPLAY.state_at(115)["score"] == 166
    assert HERO_REPLAY.state_at(115)["runs_required"] == 39
    assert HERO_REPLAY.state_at(115)["balls_remaining"] == 8
    assert HERO_REPLAY.state_at(117)["score"] == 176
    assert HERO_REPLAY.state_at(117)["runs_required"] == 29
    assert HERO_REPLAY.state_at(117)["balls_remaining"] == 6
    final = HERO_REPLAY.state_at(123)
    assert (final["score"], final["wickets_lost"], final["legal_balls_used"]) == (207, 7, 120)
    assert [c["id"] for c in HERO_REPLAY.turning_points] == ["rashid-hat-trick", "penultimate-boundaries", "five-sixes-finish"]
    assert HERO_REPLAY.turning_points[0]["start_cursor"] == 97
    assert HERO_REPLAY.turning_points[0]["end_cursor"] == 100


def test_replay_payload_never_discloses_final_result_before_final_delivery():
    pre = HERO_REPLAY.public_at(122)
    assert "final_result" not in pre
    assert all(card["end_cursor"] <= 122 for card in pre["visible_turning_points"])
    at_final = HERO_REPLAY.public_at(123)
    assert at_final["final_result"]["winner"] == "Kolkata Knight Riders"
    assert at_final["final_result"]["final_score"] == "207/7"


def test_replay_cursor_validation():
    with pytest.raises(CricketInputError):
        HERO_REPLAY.state_at(-1)
    with pytest.raises(CricketInputError):
        HERO_REPLAY.state_at(124)


def test_terminal_state_does_not_claim_playable_balls_remain():
    completed_early = calculate_manual_state(205, 4, 16, 0, 205)
    all_out_early = calculate_manual_state(180, 10, 16, 0, 205)
    assert completed_early["status"] == "target_reached"
    assert completed_early["balls_remaining"] == 0
    assert completed_early["required_run_rate"] is None
    assert all_out_early["status"] == "all_out"
    assert all_out_early["balls_remaining"] == 0
    assert all_out_early["required_run_rate"] is None


def test_quick_entry_error_reports_the_actual_invalid_over_notation():
    from app.quick_entry import parse_quick_entry

    with pytest.raises(CricketInputError, match=r"13\.6 is invalid"):
        parse_quick_entry("112/3 13.6 t189")
