"""Synthetic tests for EDA competition, phase, aggregation, and state logic."""

from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.exploratory_analysis import (
    _batting_aggregate,
    _bowling_aggregate,
    _build_match_states,
    add_delivery_context,
    competition_group,
)


def test_competition_groups_keep_ipl_t20i_and_other_t20_separate():
    frame = pd.DataFrame(
        {
            "comp_name": ["IPL 2024", "India T20I 2024", "Asia Cup Rising Stars 2025"],
            "match_format": ["T20", "T20I", "T20"],
        }
    )
    assert competition_group(frame).tolist() == ["IPL", "T20I", "Other T20"]


def _phase_tables() -> dict[str, pd.DataFrame]:
    extras_type = [None] * 97
    extras_type[35] = "other"
    extras_one = [0] * 97
    extras_one[35] = 1
    balls = pd.DataFrame(
        {
            "match_id": [1] * 97,
            "innings": [1] * 97,
            "innings_ball_num": list(range(1, 98)),
            "runs": [0] * 97,
            "extras1_runs": extras_one,
            "extras_2_runs": [0] * 97,
            "extras_type": extras_type,
            "is_four": [False] * 97,
            "is_six": [False] * 97,
            "is_wicket": [False] * 97,
            "dismissal_type": [None] * 97,
        }
    )
    matches = pd.DataFrame(
        {
            "match_id": [1], "comp_id": [10], "match_date": ["2024-01-01"],
            "home_team_id": [1], "away_team_id": [2],
            "first_inn_revised_overs": [None], "second_inn_revised_overs": [None],
            "revised_target": [None], "result": ["Example won by 1 wicket"],
        }
    )
    competitions = pd.DataFrame(
        {"comp_id": [10], "comp_name": ["IPL 2024"], "match_format": ["T20"], "season": ["2024"]}
    )
    return {"ball_by_ball": balls, "matches": matches, "competitions": competitions}


def test_phase_assignment_uses_known_legal_delivery_sequence_and_flags_boundary_uncertainty():
    deliveries = add_delivery_context(_phase_tables())
    assert deliveries.loc[34, "innings_phase"] == "Powerplay (overs 1-6)"
    assert deliveries.loc[35, "innings_phase"] == "Powerplay (overs 1-6)"
    assert deliveries.loc[36, "innings_phase"] == "Uncertain phase"
    assert deliveries.loc[37, "innings_phase"] == "Middle (overs 7-15)"
    assert deliveries.loc[35, "legal_delivery_calculated"] is pd.NA or pd.isna(deliveries.loc[35, "legal_delivery_calculated"])


def test_batting_aggregation_exposes_uncertain_faced_balls_and_player_stats():
    frame = pd.DataFrame(
        {
            "competition_group": ["IPL"] * 4,
            "season": ["2024"] * 4,
            "striker_id": [7] * 4,
            "match_id": [1, 1, 2, 2],
            "innings": [1, 1, 1, 1],
            "runs": [1, 4, 0, 0],
            "is_four": [False, True, False, False],
            "is_six": [False] * 4,
            "boundary_event": [False, True, False, False],
            "batter_ball_faced_known": [True, True, True, False],
            "batter_ball_faced_uncertain": [False, False, False, True],
            "batter_dot_known": [False, False, True, False],
            "is_wicket": [False, False, True, False],
            "dismissed_player_id": [None, None, 7, None],
        }
    )
    summary = _batting_aggregate(frame, ["competition_group", "season", "striker_id"]).iloc[0]
    assert summary["runs"] == 5
    assert summary["balls_faced_known"] == 3
    assert summary["balls_faced_uncertain"] == 1
    assert summary["dismissals"] == 1
    assert summary["strike_rate_known_balls"] == pytest.approx(5 / 3 * 100)


def test_bowling_aggregation_uses_known_legal_balls_and_attributable_wickets():
    frame = pd.DataFrame(
        {
            "competition_group": ["IPL"] * 4,
            "season": ["2024"] * 4,
            "bowler_id": [9] * 4,
            "match_id": [1, 1, 2, 2],
            "innings": [1, 1, 1, 1],
            "legal_delivery_calculated": pd.Series([True, True, False, pd.NA], dtype="boolean"),
            "bowler_wicket_calculated": [False, True, False, False],
            "is_wicket": [False, True, False, False],
            "bowler_runs_conceded_known": [1, 4, 1, 0],
            "bowler_runs_attribution_unknown": [False, False, False, True],
            "known_total_run_dot": [False] * 4,
            "boundary_event": [False, True, False, False],
        }
    )
    summary = _bowling_aggregate(frame, ["competition_group", "season", "bowler_id"]).iloc[0]
    assert summary["legal_deliveries_known"] == 2
    assert summary["legal_status_unknown_deliveries"] == 1
    assert summary["bowling_runs_conceded_known"] == 6
    assert summary["bowler_attributable_wickets"] == 1
    assert summary["economy_known"] == 18


def test_match_states_use_pre_delivery_data_and_no_final_result_feature():
    tables = _phase_tables()
    tables["ball_by_ball"] = pd.DataFrame(
        {
            "match_id": [1, 1, 1, 1], "innings": [1, 1, 2, 2],
            "innings_ball_num": [1, 2, 1, 2], "runs": [1, 0, 0, 1],
            "extras1_runs": [0, 0, 0, 0], "extras_2_runs": [0, 0, 0, 0],
            "extras_type": [None] * 4, "is_wicket": [False] * 4,
            "is_four": [False] * 4, "is_six": [False] * 4,
            "dismissal_type": [None] * 4,
        }
    )
    deliveries = add_delivery_context(tables)
    states = _build_match_states(deliveries, tables)
    chase_start = states.loc[states.innings.eq(2)].iloc[0]
    assert chase_start.chase_target == 2
    assert chase_start.runs_before == 0
    assert chase_start.wickets_remaining == 10
    assert chase_start.required_runs == 2
    assert chase_start.target_source == "first-innings total + 1"
    assert "result" not in states.columns
