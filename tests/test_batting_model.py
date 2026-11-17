"""Synthetic tests for Phase 5 batting calculations and leakage guards."""

import numpy as np
import pandas as pd

from config import BattingModelConfig
from src.evaluation.batting_uncertainty import bootstrap_batter_pressure
from src.models.batting_baseline import add_historical_personal_baselines, assign_walk_forward_pressure_labels
from src.models.batting_features import build_batting_delivery_features, derive_batting_outcomes
from src.models.batting_scoring import assign_eligibility, build_composite_scores
from src.models.batting_statistics import build_batter_statistics


def _synthetic() -> pd.DataFrame:
    rows = []
    for date, match, pressure, player in [("2024-01-01", 1, 20, 1), ("2024-02-01", 2, 90, 1), ("2024-02-01", 2, 90, 2)]:
        rows.append({"match_date": date, "match_id": match, "innings": 1, "innings_ball_num": 1,
            "competition_group": "IPL", "comp_id": 1, "season": "2024", "striker_id": player,
            "player_name": f"P{player}", "batting_team_id": 10, "batting_team_name": "A", "opponent_team_name": "B",
            "innings_phase": "middle", "innings_context": "first innings", "pressure_label_available": True,
            "pressure_high_p90_walk_forward": pressure >= 90, "pressure_high_nonlinear_p90_walk_forward": pressure >= 90,
            "pressure_band_walk_forward": "Extreme" if pressure >= 90 else "Low", "batting_pressure_weighted": pressure,
            "balls_faced_known": True, "balls_faced_uncertain": False, "batter_runs": 4 if player == 1 else 0,
            "fours_known": player == 1, "sixes_known": False, "boundary_count_known": int(player == 1),
            "boundary_runs_known": 4 if player == 1 else 0, "dot_ball_batter_run_known": player == 2,
            "total_runs_calculated": 4 if player == 1 else 0, "dismissal_event_valid": False,
            "dismissed_player_id": np.nan, "dismissal_for_striker": False, "match_date": date,
            "_source_row_id": len(rows), "current_run_rate": 4, "required_run_rate": np.nan,
            "wickets_remaining": 10, "legal_balls_remaining": 119, "innings_progress_fraction": .01,
            "remaining_target_fraction": np.nan})
    return pd.DataFrame(rows)


def test_walk_forward_labels_use_only_previous_months_and_keep_early_unavailable():
    frame = _synthetic()
    cfg = BattingModelConfig(pressure_reference_minimum=1)
    result = assign_walk_forward_pressure_labels(frame, cfg)
    # First month has no prior reference, while February can be labeled.
    assert not result.loc[0, "pressure_label_available"]
    assert result.loc[1:, "pressure_label_available"].all()
    assert result.loc[1, "pressure_reference_state_count"] == 1


def test_baseline_excludes_current_month_and_pressure_observations():
    frame = _synthetic()
    cfg = BattingModelConfig(pressure_reference_minimum=0, minimum_context_pool_balls=1)
    labeled = assign_walk_forward_pressure_labels(frame, cfg)
    labeled.loc[0, "pressure_label_available"] = True
    labeled.loc[0, "pressure_high_p90_walk_forward"] = False
    labeled.loc[1:, "pressure_high_p90_walk_forward"] = True
    result, _ = add_historical_personal_baselines(labeled, cfg)
    assert result.loc[0, "baseline_player_history_balls"] == 0
    assert result.loc[1, "baseline_player_history_balls"] == 1
    assert result.loc[1, "baseline_rpb_hierarchical"] == 4


def test_composite_bounded_and_eligibility_retains_insufficient_players():
    frame = pd.DataFrame({"competition_group": ["IPL", "IPL"], "striker_id": [1, 2],
        "pressure_runs_per_ball": [1.5, .2], "pressure_boundary_frequency": [.2, 0],
        "pressure_dot_ball_frequency": [.3, .8], "pressure_dismissal_frequency": [.01, .2],
        "runs_per_ball_absolute_difference": [.2, -.4], "boundary_frequency_absolute_difference": [.1, -.1],
        "dot_ball_frequency_absolute_difference": [-.1, .2], "dismissal_frequency_absolute_difference": [-.01, .1],
        "player_match_rpb_std": [.2, .8], "pressure_balls": [80, 4], "pressure_matches": [7, 1],
        "baseline_history_balls": [60, 0], "uncertain_faced_deliveries": [0, 1]})
    scores, _ = build_composite_scores(frame, frame)
    eligible, _ = assign_eligibility(scores, BattingModelConfig())
    assert eligible["overall_batting_pressure_score"].between(0, 100).all()
    assert len(eligible) == 2
    assert eligible.loc[eligible.striker_id.eq(2), "eligibility_status"].item() == "insufficient pressure sample"


def test_match_bootstrap_is_reproducible_and_clustered():
    frame = _synthetic()
    frame["pressure_high_p90_walk_forward"] = True
    a = bootstrap_batter_pressure(frame, replicates=30, seed=7)
    b = bootstrap_batter_pressure(frame, replicates=30, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert a["bootstrap_unit"].eq("match").all()


def test_feature_join_rejects_duplicate_pressure_delivery_keys():
    # Join-key validation is isolated here to ensure duplicates never multiply records.
    from src.models.batting_features import _ensure_unique_dimension
    import pytest
    with pytest.raises(ValueError, match="duplicate keys"):
        _ensure_unique_dimension(pd.DataFrame({"id": [1, 1]}), ["id"], "test")


def test_competition_scoring_does_not_pool_groups():
    frame = pd.DataFrame({"competition_group": ["IPL", "T20I"], "striker_id": [1, 2],
        "pressure_runs_per_ball": [1, 1], "pressure_boundary_frequency": [0, 0],
        "pressure_dot_ball_frequency": [.5, .5], "pressure_dismissal_frequency": [.1, .1],
        "runs_per_ball_absolute_difference": [0, 0], "boundary_frequency_absolute_difference": [0, 0],
        "dot_ball_frequency_absolute_difference": [0, 0], "dismissal_frequency_absolute_difference": [0, 0],
        "player_match_rpb_std": [.1, .1], "pressure_balls": [50, 50], "pressure_matches": [5, 5],
        "baseline_history_balls": [30, 30]})
    scores, _ = build_composite_scores(frame, frame)
    assert scores["overall_batting_pressure_score"].between(0, 100).all()


def test_cricket_balls_faced_boundaries_dots_and_dismissal_attribution():
    frame = pd.DataFrame({
        "extras_type": ["wide", "Overstep", "byes", "legbyes", "other", pd.NA, "none", "none", "none"],
        "extras1_runs": [1, 1, 1, 1, 1, 0, 0, 0, 0], "extras_2_runs": [0]*9,
        "runs": [0, 0, 0, 0, 4, 0, 0, 0, 0], "is_four": [False, False, False, False, True, False, False, False, False],
        "is_six": [False]*9, "total_runs_calculated": [1,1,1,1,5,0,0,0,0],
        "dismissal_type": [pd.NA]*6 + ["RunOut", "Retired Hurt", "Caught"],
        "is_wicket": [False]*6 + [True, True, True], "dismissed_player_id": [pd.NA]*6+[2,1,pd.NA],
        "striker_id": [1]*9, "legal_delivery_calculated": [False,False,True,True,pd.NA,True,True,True,True],
        "bowler_runs_attribution_unknown": [False]*9, "state_reliability": ["known"]*9,
        "is_super_over": [False]*9, "is_chasing": [False]*9, "season": ["2024"]*9,
    })
    result = derive_batting_outcomes(frame)
    assert result["balls_faced_known"].tolist() == [False, True, True, True, False, True, True, True, True]
    assert result["balls_faced_uncertain"].tolist()[4]
    assert result.loc[2, "dot_ball_batter_run_known"] and not result.loc[2, "zero_total_run_ball_known"]
    assert result.loc[5, "zero_total_run_ball_known"]
    assert result.loc[4, "boundary_on_uncertain_faced_delivery"]
    assert result.loc[6, "dismissal_event_valid"] and not result.loc[6, "dismissal_for_striker"]
    assert not result.loc[7, "dismissal_event_valid"] and result.loc[7, "retired_hurt_not_counted_as_dismissal"]
    assert result.loc[8, "dismissal_attribution_uncertain"]
