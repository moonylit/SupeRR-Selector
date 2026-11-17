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
from src.models.bowling_statistics import _weighted_baselines
from src.models.bowling_scoring import (
    eligibility_scenarios,
    integrate_bootstrap_reliability,
    score_bowlers,
)


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


def test_baseline_hierarchy_falls_back_and_records_the_selected_level():
    rows = _baseline_rows()
    # Move the only peer into another competition. The exact-comp pool is
    # empty, while the competition-group/phase/context pool has six balls.
    rows.loc[rows.bowler_id.eq(2), "comp_id"] = 20
    result, _ = add_walk_forward_bowling_baselines(
        rows, BowlingModelConfig(minimum_peer_baseline_balls=1)
    )
    july = result.loc[(result.bowler_id.eq(1)) & result.match_id.eq(3)]
    assert july.baseline_pool_source.str.contains(
        "competition_group+innings_phase+innings_context", regex=False
    ).all()
    assert july.baseline_selected_level.eq(1).all()


def test_strict_competition_baseline_does_not_cross_competition_ids():
    rows = _baseline_rows()
    rows.loc[rows.match_id.eq(3), "comp_id"] = 20
    strict, _ = add_walk_forward_bowling_baselines(
        rows, BowlingModelConfig(minimum_peer_baseline_balls=1), strategy="strict_competition"
    )
    july = strict.loc[(strict.bowler_id.eq(1)) & strict.match_id.eq(3)]
    assert july.baseline_pool_source.eq("no prior peer history").all()
    assert july.baseline_history_balls.eq(0).all()


def test_format_hierarchy_uses_personal_history_across_seasons_but_labeled_peer_fallback():
    rows = _baseline_rows()
    rows.loc[rows.match_id.eq(3), "comp_id"] = 20
    result, _ = add_walk_forward_bowling_baselines(
        rows, BowlingModelConfig(minimum_peer_baseline_balls=1), strategy="format_hierarchical"
    )
    july = result.loc[(result.bowler_id.eq(1)) & result.match_id.eq(3)]
    assert july.baseline_history_balls.eq(6).all()
    assert july.baseline_pool_source.str.contains(
        "competition_group+innings_phase+innings_context", regex=False
    ).all()
    assert july.baseline_personal_history_scope.str.contains("same format", regex=False).all()
    assert july.baseline_selected_level.eq(1).all()
    assert july.baseline_economy_approx_se.notna().all()


def test_legacy_selected_context_history_is_preserved_as_a_comparison():
    rows = _baseline_rows()
    extra = rows.loc[(rows.bowler_id.eq(1)) & rows.match_id.eq(1)].copy()
    extra["match_id"] = 5
    extra["match_date"] = "2024-06-02"
    extra["innings_context"] = "chasing"
    rows.loc[rows.match_id.eq(3), "comp_id"] = 20
    rows = pd.concat([rows, extra], ignore_index=True)
    cfg = BowlingModelConfig(minimum_peer_baseline_balls=100)
    legacy, _ = add_walk_forward_bowling_baselines(rows, cfg, strategy="legacy_hierarchical")
    format_level, _ = add_walk_forward_bowling_baselines(rows, cfg, strategy="format_hierarchical")
    old = legacy.loc[(legacy.bowler_id.eq(1)) & legacy.match_id.eq(3)].iloc[0]
    current = format_level.loc[(format_level.bowler_id.eq(1)) & format_level.match_id.eq(3)].iloc[0]
    assert old.baseline_selected_level == 3
    assert old.baseline_history_balls == 12
    assert current.baseline_history_balls == 6
    assert "legacy" in old.baseline_personal_history_scope


def test_broader_t20_pool_is_explicitly_labeled_and_keeps_context():
    rows = _baseline_rows()
    rows.loc[rows.bowler_id.eq(2), "competition_group"] = "T20I"
    rows.loc[rows.match_id.eq(3), "comp_id"] = 20
    result, _ = add_walk_forward_bowling_baselines(
        rows, BowlingModelConfig(minimum_peer_baseline_balls=1), strategy="broader_t20"
    )
    july = result.loc[(result.bowler_id.eq(1)) & result.match_id.eq(3)]
    assert july.baseline_pool_source.eq("innings_phase+innings_context").all()
    assert july.baseline_selected_level.eq(4).all()
    assert july.baseline_strategy.eq("broader_t20").all()
    assert july.baseline_peer_balls.eq(6).all()


def test_baseline_strategy_can_be_recomputed_without_stale_columns():
    rows = _baseline_rows()
    primary, _ = add_walk_forward_bowling_baselines(
        rows, BowlingModelConfig(minimum_peer_baseline_balls=1), strategy="format_hierarchical"
    )
    strict, _ = add_walk_forward_bowling_baselines(
        primary, BowlingModelConfig(minimum_peer_baseline_balls=1), strategy="strict_competition"
    )
    assert strict.baseline_strategy.eq("strict_competition").all()


def test_no_history_remains_unavailable_without_future_or_cross_day_leakage():
    rows = _baseline_rows()
    target = rows.loc[rows.match_id.eq(3)].copy()
    target["bowler_id"] = 99
    target["match_date"] = "2024-07-01"
    result, _ = add_walk_forward_bowling_baselines(
        pd.concat([rows, target], ignore_index=True),
        BowlingModelConfig(minimum_peer_baseline_balls=1000),
        strategy="format_hierarchical",
    )
    no_history = result.loc[result.bowler_id.eq(99)]
    assert no_history.baseline_history_balls.eq(0).all()
    assert no_history.baseline_economy.notna().all()  # peer prior can be used
    assert no_history.baseline_personal_history_scope.str.contains("same format", regex=False).all()


def test_player_only_baseline_has_a_poisson_standard_error():
    rows = _baseline_rows().loc[lambda frame: frame.bowler_id.eq(1)].copy()
    result, _ = add_walk_forward_bowling_baselines(
        rows, BowlingModelConfig(minimum_peer_baseline_balls=100), strategy="strict_competition"
    )
    july = result.loc[result.match_id.eq(3)].iloc[0]
    assert july.baseline_pool_source == "player-only prior history"
    assert np.isclose(july.baseline_economy_approx_se, np.sqrt(6) / 6 * 6)


def test_player_baseline_level_and_source_use_the_same_modal_context():
    rows = _baseline_rows()
    result, _ = add_walk_forward_bowling_baselines(
        rows, BowlingModelConfig(minimum_peer_baseline_balls=1)
    )
    high = result.loc[(result.bowler_id.eq(1)) & result.match_id.eq(3)].copy()
    other = high.iloc[[0]].copy()
    other["baseline_selected_level"] = 1
    other["baseline_pool_source"] = "competition_group+innings_phase+innings_context"
    summary = _weighted_baselines(pd.concat([high, other], ignore_index=True)).iloc[0]
    assert summary.baseline_selected_level == 0
    assert summary.baseline_pool_source == "competition_group+comp_id+innings_phase+innings_context"


def test_future_normal_matches_do_not_change_an_earlier_baseline():
    config = BowlingModelConfig(minimum_peer_baseline_balls=1)
    original, _ = add_walk_forward_bowling_baselines(_baseline_rows(), config)
    future_rows = pd.DataFrame([
        {
            "competition_group": "IPL", "comp_id": 10,
            "innings_phase": "Powerplay", "innings_context": "first innings",
            "bowler_id": 1, "match_id": 4, "innings": 1,
            "match_date": "2024-08-01", "bowling_pressure_band_walk_forward": "normal",
            "legal_ball_known": True, "bowler_runs_conceded_lower": 20,
            "bowler_wicket_calculated": False, "bowler_dot_legal": False,
            "boundary_delivery": True, "bowler_extra_runs_known": 0,
        }
        for _ in range(6)
    ])
    extended, _ = add_walk_forward_bowling_baselines(
        pd.concat([_baseline_rows(), future_rows], ignore_index=True), config
    )
    before = original.loc[(original.bowler_id.eq(1)) & original.match_id.eq(3)].iloc[0]
    after = extended.loc[(extended.bowler_id.eq(1)) & extended.match_id.eq(3)].iloc[0]
    assert after.baseline_history_balls == before.baseline_history_balls
    assert after.baseline_economy == before.baseline_economy


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


def test_legbyes_penalty_and_unknown_extras_are_not_mischarged():
    frame = pd.DataFrame({
        "runs": [0, 0, 0, 0],
        "extras1_runs": [4, 5, 1, 0],
        "extras_2_runs": [0, 0, 0, 0],
        "extras_type": ["legbyes", "Penalty", "other", None],
        "is_wicket": [False, False, False, False],
        "dismissal_type": [None, None, None, None],
    })
    result = calculate_delivery_metrics(frame)
    assert result.bowler_runs_conceded_known.tolist() == [0, 0, 0, 0]
    assert result.legal_delivery_calculated.iloc[0] == True
    assert pd.isna(result.legal_delivery_calculated.iloc[1])
    assert pd.isna(result.legal_delivery_calculated.iloc[2])
    assert result.legal_delivery_calculated.iloc[3] == True
    assert result.bowler_runs_attribution_unknown.tolist() == [False, False, True, False]
    assert result.extras_calculated.tolist() == [4, 5, 1, 0]


def test_only_credited_dismissal_types_count_as_bowler_wickets():
    credited = ["bowled", "caught", "lbw", "stump", "hit wicket"]
    non_credited = ["runout", "retire out", "retired hurt", "obstracting the field", "obstructing the field"]
    dismissal_types = credited + non_credited
    result = calculate_delivery_metrics(pd.DataFrame({
        "runs": [0] * len(dismissal_types),
        "extras1_runs": [0] * len(dismissal_types),
        "extras_2_runs": [0] * len(dismissal_types),
        "extras_type": [None] * len(dismissal_types),
        "is_wicket": [True] * len(dismissal_types),
        "dismissal_type": dismissal_types,
    }))
    assert result.bowler_wicket_calculated.tolist() == [True] * len(credited) + [False] * len(non_credited)
    assert not result.wicket_attribution_unknown.any()


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
    assert np.allclose(
        scores.score_model_d_reliability_adjusted,
        scores.score_model_d_performance_only
        * (0.5 + 0.5 * scores.statistical_reliability_score / 100),
    )
    assert not np.allclose(
        scores.score_model_d_reliability_adjusted,
        scores.score_model_b_correlation_adjusted
        * (0.5 + 0.5 * scores.statistical_reliability_score / 100),
    )
    scenarios = eligibility_scenarios(scores, BowlingModelConfig())
    assert len(scenarios) == 8  # four thresholds for each represented competition group
    assert scenarios.players_total.sum() >= len(scores)


def test_baseline_coverage_gate_is_configurable_and_reported_as_an_exclusion():
    fixture = _score_fixture()
    fixture.loc[fixture.competition_group.eq("IPL") & fixture.bowler_id.eq(1), "baseline_coverage_fraction"] = 0.40
    lenient, _ = score_bowlers(fixture, BowlingModelConfig(baseline_coverage_minimum=0.30))
    strict, _ = score_bowlers(fixture, BowlingModelConfig(baseline_coverage_minimum=0.50))
    target = lenient.competition_group.eq("IPL") & lenient.bowler_id.eq(1)
    assert lenient.loc[target, "eligible_analytical_candidate"].item()
    excluded = strict.loc[strict.competition_group.eq("IPL") & strict.bowler_id.eq(1)].iloc[0]
    assert not excluded.eligible_analytical_candidate
    assert "walk-forward baseline coverage below 50%" in excluded.eligibility_status


def test_bootstrap_reliability_adjusts_model_d_only_once():
    scores, _ = score_bowlers(_score_fixture(), BowlingModelConfig())
    uncertainty = pd.DataFrame({
        "competition_group": scores.competition_group,
        "bowler_id": scores.bowler_id,
        "bootstrap_score_draws": 10,
        "top10_selection_frequency": [0.9, 0.2, 0.6],
        "bootstrap_score_p95": [70.0, 60.0, 65.0],
        "bootstrap_score_p05": [30.0, 20.0, 25.0],
        "bootstrap_rank_std": [1.0, 4.0, 2.0],
    })
    integrated = integrate_bootstrap_reliability(scores, uncertainty)
    expected = integrated.score_model_d_performance_only * (
        0.5 + 0.5 * integrated.statistical_reliability_score / 100
    )
    assert np.allclose(integrated.score_model_d_reliability_adjusted, expected)


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
