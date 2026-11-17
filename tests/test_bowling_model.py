"""Cricket accounting, leakage, reliability, scoring, and bootstrap tests for Phase 6."""

import numpy as np
import pandas as pd

from config import BowlingModelConfig
from src.data.cricket_calculations import calculate_delivery_metrics
from src.evaluation.bowling_uncertainty import bootstrap_bowler_scores
from src.models.bowling_baseline import (
    add_walk_forward_bowling_baselines,
    assign_walk_forward_pressure_bands,
)
from src.models.bowling_features import aggregate_bowling_counts
from src.models.bowling_scoring import eligibility_scenarios, score_bowlers


def _baseline_rows() -> pd.DataFrame:
    rows = []
    # June's normal-pressure observations are historical only for July. A
    # second match on June 1 must not serve as a baseline for another June 1 match.
    for player, match, date, band, runs in [
        (1, 1, "2024-06-01", "normal", 1),
        (1, 2, "2024-06-01", "high", 6),
        (2, 1, "2024-06-01", "normal", 0),
        (1, 3, "2024-07-01", "high", 2),
    ]:
        for ball in range(6):
            rows.append({
                "competition_group": "IPL", "comp_id": 10,
                "innings_phase": "Powerplay", "innings_context": "first innings",
                "bowler_id": player, "match_id": match, "innings": 1,
                "match_date": date, "bowling_pressure_band_walk_forward": band,
                "legal_ball_known": True, "bowler_runs_conceded_lower": runs,
                "bowler_wicket_calculated": ball == 5 and player == 2,
                "bowler_dot_legal": runs == 0,
                "boundary_delivery": False, "bowler_extra_runs_known": 0,
            })
    return pd.DataFrame(rows)


def test_pressure_bands_use_only_prior_months_and_preserve_early_unavailable():
    rows = []
    for date, score in [("2024-01-01", 10), ("2024-01-02", 20), ("2024-01-03", 30),
                        ("2024-02-01", 5), ("2024-02-02", 95)]:
        rows.append({"match_date": date, "competition_group": "IPL", "innings_phase": "Powerplay",
                     "bowling_pressure_weighted": score})
    config = BowlingModelConfig(pressure_reference_minimum=3)
    result = assign_walk_forward_pressure_bands(pd.DataFrame(rows), config)
    assert result.loc[:2, "bowling_pressure_band_walk_forward"].eq("reference unavailable").all()
    assert result.loc[3, "bowling_pressure_band_walk_forward"] == "low"
    assert result.loc[4, "bowling_pressure_band_walk_forward"] == "high"
    changed = pd.DataFrame(rows + [{"match_date": "2024-02-03", "competition_group": "IPL",
                                    "innings_phase": "Powerplay", "bowling_pressure_weighted": 10000}])
    updated = assign_walk_forward_pressure_bands(changed, config)
    assert updated.loc[3:4, "bowling_pressure_band_walk_forward"].tolist() == result.loc[3:4, "bowling_pressure_band_walk_forward"].tolist()


def test_baselines_exclude_same_day_matches_and_include_only_earlier_dates():
    config = BowlingModelConfig(minimum_peer_baseline_balls=1)
    result, _ = add_walk_forward_bowling_baselines(_baseline_rows(), config)
    same_day = result.loc[(result.bowler_id.eq(1)) & result.match_id.eq(2)]
    july = result.loc[(result.bowler_id.eq(1)) & result.match_id.eq(3)]
    assert same_day.baseline_history_balls.eq(0).all()
    assert same_day.baseline_economy.isna().all()
    assert july.baseline_history_balls.eq(6).all()
    assert july.baseline_economy.notna().all()
    assert july.baseline_pool_source.str.contains("competition_group").all()


def test_cricket_accounting_separates_bowler_charges_and_wickets():
    frame = pd.DataFrame({
        "runs": [4, 0, 2, 0, 0, 0],
        "extras1_runs": [0, 1, 1, 2, 0, 0],
        "extras_2_runs": [0, 0, 0, 0, 0, 0],
        "extras_type": [None, "wide", "overstep", "byes", None, None],
        "is_wicket": [False, False, False, False, True, True],
        "dismissal_type": [None, None, None, None, "runout", "caught"],
    })
    result = calculate_delivery_metrics(frame)
    assert result.legal_delivery_calculated.tolist() == [True, False, False, True, True, True]
    assert result.bowler_runs_conceded_known.tolist() == [4, 1, 3, 0, 0, 0]
    assert result.bowler_wicket_calculated.tolist() == [False, False, False, False, False, True]


def test_aggregate_rates_use_legal_balls_and_handle_zero_denominators():
    frame = pd.DataFrame({
        "match_id": [1, 1], "innings": [1, 1], "competition_group": ["IPL", "IPL"],
        "bowler_id": [7, 7], "legal_ball_known": [True, False], "legal_ball_possible": [True, False],
        "bowler_runs_conceded_lower": [2, 1], "bowler_runs_conceded_upper": [2, 1],
        "bowler_wicket_calculated": [True, False], "bowler_dot_legal": [False, False],
        "team_dot_legal": [False, False], "boundary_delivery": [False, False],
        "four_conceded": [False, False], "six_conceded": [False, False],
        "wide_delivery": [False, True], "no_ball_delivery": [False, False],
        "bowler_extra_runs_known": [0, 1], "bowler_runs_attribution_uncertain_upper": [0, 0],
        "legal_ball_uncertain": [False, False], "bowler_runs_attribution_unknown": [False, False],
        "wicket_attribution_unknown": [False, False],
    })
    result = aggregate_bowling_counts(frame, ["competition_group", "bowler_id"]).iloc[0]
    assert result.legal_balls_known == 1
    assert result.economy == 18
    assert result.wickets_per_legal_over == 6
    empty = frame.assign(legal_ball_known=False, legal_ball_possible=False)
    result = aggregate_bowling_counts(empty, ["competition_group", "bowler_id"]).iloc[0]
    assert np.isnan(result.economy)
    assert np.isnan(result.bowling_strike_rate_balls_per_wicket)


def test_boundary_runs_and_runs_per_legal_ball_are_explicit():
    frame = pd.DataFrame({
        "match_id": [1, 1], "innings": [1, 1], "competition_group": ["IPL", "IPL"],
        "bowler_id": [7, 7], "legal_ball_known": [True, True], "legal_ball_possible": [True, True],
        "bowler_runs_conceded_lower": [4, 6], "bowler_runs_conceded_upper": [4, 6],
        "bowler_wicket_calculated": [False, False], "bowler_dot_legal": [False, False],
        "team_dot_legal": [False, False], "boundary_delivery": [True, True],
        "four_conceded": [True, False], "six_conceded": [False, True],
        "wide_delivery": [False, False], "no_ball_delivery": [False, False],
        "bowler_extra_runs_known": [0, 0], "bowler_runs_attribution_uncertain_upper": [0, 0],
        "legal_ball_uncertain": [False, False], "bowler_runs_attribution_unknown": [False, False],
        "wicket_attribution_unknown": [False, False],
    })
    result = aggregate_bowling_counts(frame, ["competition_group", "bowler_id"]).iloc[0]
    assert result.boundary_runs_conceded == 10
    assert result.runs_per_legal_ball == 5
    assert result.economy == 30


def _score_fixture() -> pd.DataFrame:
    rows = []
    for group, player, economy, wickets in [("IPL", 1, 6.0, 1.5), ("IPL", 2, 8.0, 0.5),
                                             ("T20I", 1, 7.0, 1.0)]:
        rows.append({
            "competition_group": group, "bowler_id": player,
            "economy_high": economy, "wickets_per_legal_over_high": wickets,
            "bowler_dot_rate_high": 0.4, "boundary_concession_rate_high": 0.15,
            "bowler_extras_per_legal_over_high": 0.2,
            "relative_economy_improvement": 0.5 - economy / 10,
            "relative_wickets_per_over_improvement": wickets - 1,
            "relative_dot_rate_improvement": 0.05, "relative_boundary_rate_improvement": 0.02,
            "relative_extras_per_over_improvement": 0.01,
            "pressure_matches": 5, "pressure_innings": 5, "pressure_legal_balls": 60,
            "baseline_history_balls": 40, "baseline_coverage_fraction": 1.0,
            "baseline_peer_balls": 100, "baseline_economy": 7.0,
            "baseline_wickets_per_over": 1.0, "baseline_dot_rate": 0.35,
            "baseline_boundary_rate": 0.2, "baseline_extras_per_over": 0.3,
            "baseline_economy_raw": 7.5, "baseline_wickets_per_over_raw": 1.2,
            "baseline_dot_rate_raw": 0.3, "baseline_boundary_rate_raw": 0.25,
            "baseline_extras_per_over_raw": 0.4,
            "runs_conceded_lower_high": 60, "legal_balls_known_high": 60,
            "bowler_wickets_high": 15, "legal_bowler_dots_high": 24,
            "boundary_deliveries_high": 9, "bowler_extra_runs_known_high": 2,
            "match_performance_p10": 30 + player, "matches_meeting_standard_fraction": 0.6,
            "match_performance_mad": 10.0, "match_performance_std": 12.0,
        })
    rows[1]["pressure_matches"] = 1
    rows[1]["pressure_innings"] = 1
    rows[1]["pressure_legal_balls"] = 12
    rows[1]["baseline_history_balls"] = 0
    rows[1]["baseline_coverage_fraction"] = 0
    return pd.DataFrame(rows)


def test_scores_are_bounded_and_ineligible_players_are_retained():
    scores, weights = score_bowlers(_score_fixture(), BowlingModelConfig())
    assert len(scores) == 3
    assert scores.overall_bowling_pressure_score.between(0, 100).all()
    assert not scores.loc[scores.bowler_id.eq(2), "eligible_analytical_candidate"].item()
    assert not weights.empty
    scenarios = eligibility_scenarios(scores, BowlingModelConfig())
    assert len(scenarios) == 8  # four thresholds for each represented competition group
    assert scenarios.players_total.sum() >= len(scores)


def test_match_bootstrap_is_reproducible_and_uses_shared_match_resampling():
    rows = []
    for match in range(1, 7):
        for player in (1, 2):
            rows.append({
                "competition_group": "IPL", "bowler_id": player, "match_id": match,
                "pressure_band": "high", "legal_balls_known": 12,
                "runs_conceded_lower": 12 + player * match,
                "bowler_wickets": match % 2, "legal_bowler_dots": 4,
                "boundary_deliveries": 2, "bowler_extra_runs_known": 1,
            })
    matches = pd.DataFrame(rows)
    scores = pd.DataFrame({
        "competition_group": ["IPL", "IPL"], "bowler_id": [1, 2],
        "eligible_analytical_candidate": [True, True],
        "baseline_economy": [7.0, 8.0], "baseline_wickets_per_over": [1.0, 1.0],
        "baseline_dot_rate": [0.3, 0.3], "baseline_boundary_rate": [0.2, 0.2],
        "baseline_extras_per_over": [0.2, 0.2], "consistency_score": [60.0, 50.0],
        "statistical_reliability_score": [60.0, 40.0],
    })
    long_a, summary_a = bootstrap_bowler_scores(matches, scores, replicates=25, seed=10)
    long_b, summary_b = bootstrap_bowler_scores(matches, scores, replicates=25, seed=10)
    pd.testing.assert_frame_equal(long_a, long_b)
    pd.testing.assert_frame_equal(summary_a, summary_b)
    assert long_a.groupby("replicate").bowler_id.nunique().eq(2).all()
    assert summary_a.bootstrap_unit.eq("match").all()
    assert summary_a.top10_selection_frequency.between(0, 1).all()
    expected = long_a[[
        "absolute_score", "relative_score", "consistency_score",
        "statistical_reliability_score",
    ]].mean(axis=1)
    assert np.allclose(long_a.bootstrap_score, expected)
