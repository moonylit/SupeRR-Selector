import numpy as np
import pandas as pd

from src.pressure.match_state import build_delivery_match_states
from src.pressure.pressure_detection import (
    _combine_index,
    PressureConfig,
    _rank_by_context,
    add_high_pressure_flags,
    add_pressure_features,
)
from src.pressure.pressure_validation import _scenario_data


def _tables():
    competitions = pd.DataFrame(
        [{"comp_id": 1, "comp_name": "IPL 2025", "match_format": "T20", "season": 2025}]
    )
    matches = pd.DataFrame(
        [
            {
                "match_id": 10,
                "comp_id": 1,
                "match_date": "2025-04-01",
                "home_team_id": 1,
                "away_team_id": 2,
                "revised_target": np.nan,
                "first_inn_revised_overs": 12,
                "second_inn_revised_overs": 12,
            },
            {
                "match_id": 20,
                "comp_id": 1,
                "match_date": "2025-04-02",
                "home_team_id": 3,
                "away_team_id": 4,
                "revised_target": np.nan,
                "first_inn_revised_overs": 6,
                "second_inn_revised_overs": 6,
            },
        ]
    )
    rows = []

    def add(match, innings, ball, *, runs=0, extra=0, extras_type=None, wicket=False, dismissal=None):
        rows.append(
            {
                "match_id": match,
                "innings": innings,
                "innings_ball_num": ball,
                "batting_team_id": 1 if innings % 2 else 2,
                "bowling_team_id": 2 if innings % 2 else 1,
                "striker_id": 100 + innings,
                "bowler_id": 200 + innings,
                "runs": runs,
                "extras1_runs": extra,
                "extras_2_runs": 0,
                "extras_type": extras_type,
                "is_wicket": wicket,
                "dismissal_type": dismissal,
                "is_four": False,
                "is_six": False,
            }
        )

    # First innings: legal, wide, no-ball (Overstep), bye, unknown legality.
    add(10, 1, 1, runs=2)
    add(10, 1, 2, extra=1, extras_type="wide")
    add(10, 1, 3, runs=1, extra=1, extras_type="Overstep")
    add(10, 1, 4, extra=2, extras_type="byes")
    add(10, 1, 5, extra=1, extras_type="other")
    # A wicket row tests pre-delivery wicket state: this event only affects the next ball.
    add(10, 1, 6, runs=1, wicket=True, dismissal="bowled")
    add(10, 1, 7, runs=1)
    # Chase target is first-innings total + 1.
    add(10, 2, 1, runs=1)
    add(10, 2, 2, wicket=True, dismissal="runout")
    add(10, 2, 3, runs=2)
    # A super-over pair must use six-ball limits and the preceding super-over score.
    add(20, 3, 1, runs=1)
    add(20, 4, 1, runs=1)
    return {
        "ball_by_ball": pd.DataFrame(rows),
        "matches": matches,
        "competitions": competitions,
    }


def test_states_are_pre_delivery_and_wide_no_ball_do_not_consume_balls():
    states = build_delivery_match_states(_tables())
    first = states.loc[(states.match_id == 10) & (states.innings == 1)].reset_index(drop=True)
    assert first.loc[0, "runs_scored_before"] == 0
    assert first.loc[0, "legal_balls_bowled_known"] == 0
    assert first.loc[1, "legal_balls_bowled_known"] == 1  # wide did not consume a ball
    assert first.loc[2, "legal_balls_bowled_known"] == 1  # no-ball did not consume a ball
    assert first.loc[3, "legal_balls_bowled_known"] == 1
    assert first.loc[4, "legal_balls_bowled_known"] == 2  # bye consumed a ball
    assert first.loc[4, "legal_balls_remaining_min"] == 10
    assert first.loc[4, "legal_balls_remaining_max"] == 10
    assert first.loc[5, "legal_balls_remaining_min"] == 9
    assert first.loc[5, "legal_balls_remaining_max"] == 10
    assert first.loc[6, "completed_overs_lower"] == 0
    assert first.loc[6, "completed_overs_upper"] == 0
    assert first.loc[6, "balls_into_over_lower"] == 3
    assert first.loc[6, "balls_into_over_upper"] == 4


def test_wickets_and_chase_requirements_use_only_prior_delivery_events():
    states = build_delivery_match_states(_tables())
    first = states.loc[(states.match_id == 10) & (states.innings == 1)].reset_index(drop=True)
    assert first.loc[5, "wickets_lost_before"] == 0
    assert first.loc[6, "wickets_lost_before"] == 1

    chase = states.loc[(states.match_id == 10) & (states.innings == 2)].reset_index(drop=True)
    assert chase.loc[0, "chase_target"] == 11
    assert chase.loc[0, "runs_required"] == 11
    assert chase.loc[0, "legal_balls_remaining"] == 12
    assert chase.loc[1, "runs_required"] == 10
    assert chase.loc[1, "wickets_remaining"] == 10
    assert chase.loc[2, "wickets_remaining"] == 9


def test_super_over_pairing_and_limits_are_explicit():
    states = build_delivery_match_states(_tables())
    first = states.loc[(states.match_id == 20) & (states.innings == 3)].iloc[0]
    chase = states.loc[(states.match_id == 20) & (states.innings == 4)].iloc[0]
    assert first.innings_ball_limit == 6
    assert first.is_super_over
    assert chase.is_chasing
    assert chase.chase_target == 2
    assert chase.target_source == "previous innings total + 1"


def test_positive_chase_requirement_with_no_balls_remaining_has_infinite_rate():
    tables = _tables()
    tables["matches"].loc[tables["matches"].match_id.eq(10), "second_inn_revised_overs"] = 2
    states = build_delivery_match_states(tables)
    last = states.loc[(states.match_id == 10) & (states.innings == 2)].iloc[-1]
    assert last.runs_required > 0
    assert last.legal_balls_remaining_min == 0
    assert np.isinf(last.required_run_rate_upper)


def test_zero_known_legal_balls_keeps_current_rate_bound_explicit():
    tables = _tables()
    tables["ball_by_ball"].loc[0, ["runs", "extras1_runs", "extras_type"]] = [1, 1, "other"]
    states = build_delivery_match_states(tables)
    next_ball = states.loc[(states.match_id == 10) & (states.innings == 1) & (states.innings_ball_num == 2)].iloc[0]
    assert next_ball.legal_balls_bowled_known == 0
    assert next_ball.legal_balls_bowled_possible == 1
    assert np.isinf(next_ball.current_run_rate_upper)


def test_pressure_outputs_are_bounded_role_specific_and_walk_forward():
    states = build_delivery_match_states(_tables())
    scored = add_high_pressure_flags(add_pressure_features(states))
    for column in (
        "batting_pressure_weighted",
        "batting_pressure_nonlinear",
        "bowling_pressure_weighted",
        "bowling_pressure_nonlinear",
    ):
        assert scored[column].between(0, 100).all()
    # First match on the date has no prior-date comparison set and uses neutral normalization.
    first_date = scored.loc[scored.match_date.eq("2025-04-01")]
    assert first_date["pressure_index_reliability"].notna().all()
    assert scored["historical_boundary_event_rate_prior"].isna().all()
    assert "batting_high_weighted_fixed_80" in scored


def test_formula_monotonicity_and_scenario_order():
    demand = pd.Series([0.2, 0.4, 0.8])
    resource = pd.Series([0.2, 0.4, 0.8])
    weighted, nonlinear = _combine_index(demand, resource)
    assert weighted.is_monotonic_increasing
    assert nonlinear.is_monotonic_increasing
    weighted_config = PressureConfig(demand_weight=0.7, nonlinear_interaction_power=0.8)
    configured_weighted, configured_nonlinear = _combine_index(
        demand, resource, weighted_config.demand_weight, weighted_config.nonlinear_interaction_power
    )
    assert configured_weighted.iloc[-1] > configured_weighted.iloc[0]
    assert configured_nonlinear.iloc[-1] > configured_nonlinear.iloc[0]

    scenarios = _scenario_data().set_index("scenario")
    assert scenarios.loc["high-demand, scarce wickets", "batting_pressure_weighted_synthetic"] > scenarios.loc[
        "low-demand long chase", "batting_pressure_weighted_synthetic"
    ]
    assert scenarios.loc["final-ball, one wicket", "batting_pressure_weighted_synthetic"] > scenarios.loc[
        "high-demand, scarce wickets", "batting_pressure_weighted_synthetic"
    ]
    assert scenarios["is_synthetic_not_match_observation"].all()


def test_outcome_mutation_does_not_change_same_delivery_pre_state():
    tables = _tables()
    original = build_delivery_match_states(tables)
    modified = {key: value.copy() for key, value in tables.items()}
    # Change the first delivery's own outcome; its state before that delivery must be identical.
    modified["ball_by_ball"].loc[0, ["runs", "is_wicket", "dismissal_type"]] = [6, True, "bowled"]
    rerun = build_delivery_match_states(modified)
    original_state = original.iloc[0]
    modified_state = rerun.iloc[0]
    for column in (
        "runs_scored_before", "wickets_lost_before", "legal_balls_bowled_known",
        "legal_balls_remaining_min", "legal_balls_remaining_max",
    ):
        assert original_state[column] == modified_state[column]


def test_current_delivery_outcome_cannot_change_its_pressure_score():
    tables = _tables()
    base_states = build_delivery_match_states(tables)
    base_scores = add_pressure_features(base_states)
    changed = {key: value.copy() for key, value in tables.items()}
    changed["ball_by_ball"].loc[0, ["runs", "is_wicket", "dismissal_type"]] = [6, True, "bowled"]
    changed_scores = add_pressure_features(build_delivery_match_states(changed))
    for column in ("batting_pressure_weighted", "batting_pressure_nonlinear"):
        assert base_scores.loc[0, column] == changed_scores.loc[0, column]


def test_walk_forward_reference_isolated_from_future_months():
    frame = pd.DataFrame(
        {
            "match_date": ["2025-01-01"] * 100 + ["2025-02-01"],
            "competition_group": ["IPL"] * 101,
            "innings_phase": ["Middle"] * 101,
        }
    )
    values = pd.Series([float(i) for i in range(100)] + [1000.0])
    ranks = _rank_by_context(values, frame)
    assert ranks.iloc[:100].eq(0.5).all()
    assert ranks.iloc[100] == 1.0
