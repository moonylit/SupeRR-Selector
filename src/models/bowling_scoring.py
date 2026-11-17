"""Transparent bowling pressure dimensions, eligibility, and composite scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import BowlingModelConfig

PLAYER_KEYS = ["competition_group", "bowler_id"]
ABSOLUTE_METRICS = [
    ("economy_high", False),
    ("wickets_per_legal_over_high", True),
    ("bowler_dot_rate_high", True),
    ("boundary_concession_rate_high", False),
    ("bowler_extras_per_legal_over_high", False),
]
RELATIVE_METRICS = [
    ("relative_economy_improvement", True),
    ("relative_wickets_per_over_improvement", True),
    ("relative_dot_rate_improvement", True),
    ("relative_boundary_rate_improvement", True),
    ("relative_extras_per_over_improvement", True),
]


def _rank_score(frame: pd.DataFrame, column: str, higher_is_better: bool) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    ranked = values.groupby(frame["competition_group"], dropna=False).rank(
        method="average", pct=True
    )
    return (ranked if higher_is_better else 1 - ranked) * 100


def _add_shrunk_pressure_rates(frame: pd.DataFrame, config: BowlingModelConfig) -> pd.DataFrame:
    result = frame.copy()
    rate_fields = {
        "economy": ("runs_conceded_lower_high", "legal_balls_known_high", 6.0, config.economy_prior_balls),
        "wickets_per_legal_over": ("bowler_wickets_high", "legal_balls_known_high", 6.0, config.wicket_prior_balls),
        "dot_rate": ("legal_bowler_dots_high", "legal_balls_known_high", 1.0, config.event_rate_prior_balls),
        "boundary_rate": ("boundary_deliveries_high", "legal_balls_known_high", 1.0, config.event_rate_prior_balls),
        "extras_per_over": ("bowler_extra_runs_known_high", "legal_balls_known_high", 6.0, config.extras_prior_balls),
    }
    for metric, (num_col, den_col, scale, strength) in rate_fields.items():
        numerator = pd.to_numeric(result[num_col], errors="coerce").fillna(0)
        denominator = pd.to_numeric(result[den_col], errors="coerce").fillna(0)
        group_num = numerator.groupby(result["competition_group"], dropna=False).transform("sum")
        group_den = denominator.groupby(result["competition_group"], dropna=False).transform("sum")
        peer_num = (group_num - numerator).clip(lower=0)
        peer_den = (group_den - denominator).clip(lower=0)
        peer_rate = peer_num.div(peer_den.replace(0, np.nan))
        shrunk_per_ball = (numerator + peer_rate * strength).div(denominator + strength)
        raw_per_ball = numerator.div(denominator.replace(0, np.nan))
        result[f"pressure_{metric}_shrunk"] = (shrunk_per_ball.where(peer_rate.notna(), raw_per_ball) * scale)
        result[f"pressure_{metric}_prior"] = peer_rate * scale
        result[f"pressure_{metric}_shrinkage_fraction"] = denominator.div(denominator + strength)
        result[f"pressure_{metric}_prior_balls"] = strength
    return result


def _domain_score(frame: pd.DataFrame, metrics: list[tuple[str, bool]], prefix: str) -> tuple[pd.Series, list[str]]:
    component_columns = []
    for metric, direction in metrics:
        if metric not in frame:
            continue
        column = f"{prefix}_{metric}_score"
        frame[column] = _rank_score(frame, metric, direction)
        component_columns.append(column)
    if not component_columns:
        return pd.Series(np.nan, index=frame.index), component_columns
    return frame[component_columns].mean(axis=1), component_columns


def _correlation_weighted_domain(
    frame: pd.DataFrame, columns: list[str], domain: str
) -> tuple[pd.Series, list[dict[str, object]]]:
    if not columns:
        return pd.Series(np.nan, index=frame.index), []
    if len(columns) == 1:
        return frame[columns[0]], [{"domain": domain, "component": columns[0], "correlation_adjusted_weight": 1.0}]
    correlation = frame[columns].corr(method="spearman").abs().fillna(0)
    raw_weights = {}
    for column in columns:
        mean_corr = float(correlation[column].drop(labels=[column], errors="ignore").mean())
        raw_weights[column] = 1 / (1 + mean_corr)
    total = sum(raw_weights.values()) or 1.0
    weights = {column: value / total for column, value in raw_weights.items()}
    values = frame[columns]
    numerator = sum(values[column].fillna(0) * weight for column, weight in weights.items())
    denominator = sum(values[column].notna().astype(float) * weight for column, weight in weights.items())
    score = numerator.div(denominator.replace(0, np.nan))
    details = [{
        "domain": domain,
        "component": column,
        "mean_absolute_spearman_correlation": float(correlation[column].drop(labels=[column], errors="ignore").mean()),
        "correlation_adjusted_weight": weights[column],
    } for column in columns]
    return score, details


def score_bowlers(
    differentials: pd.DataFrame,
    config: BowlingModelConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create four separable dimensions and four bounded composite models."""
    cfg = config or BowlingModelConfig()
    frame = _add_shrunk_pressure_rates(differentials, cfg)
    absolute_a, absolute_a_cols = _domain_score(frame, ABSOLUTE_METRICS, "absolute")
    relative, relative_cols = _domain_score(frame, RELATIVE_METRICS, "relative")

    shrunk_absolute_metrics = [
        ("pressure_economy_shrunk", False),
        ("pressure_wickets_per_legal_over_shrunk", True),
        ("pressure_dot_rate_shrunk", True),
        ("pressure_boundary_rate_shrunk", False),
        ("pressure_extras_per_over_shrunk", False),
    ]
    absolute_c, absolute_c_cols = _domain_score(frame, shrunk_absolute_metrics, "shrunken")

    consistency_metrics = [
        ("match_performance_p10", True),
        ("matches_meeting_standard_fraction", True),
        ("match_performance_mad", False),
        ("match_performance_std", False),
    ]
    consistency, consistency_cols = _domain_score(frame, consistency_metrics, "consistency")
    reliability_metrics = [
        ("pressure_matches", True), ("pressure_innings", True),
        ("pressure_legal_balls", True), ("baseline_history_balls", True),
        ("baseline_coverage_fraction", True), ("baseline_peer_balls", True),
        ("baseline_selected_level", False),
    ]
    reliability, reliability_cols = _domain_score(frame, reliability_metrics, "reliability")
    frame["absolute_pressure_score"] = absolute_a
    frame["relative_improvement_score"] = relative
    frame["consistency_score"] = consistency
    frame["statistical_reliability_score"] = reliability
    frame["absolute_pressure_score_shrunk"] = absolute_c

    absolute_b, weight_rows = _correlation_weighted_domain(frame, absolute_a_cols, "absolute raw")
    relative_b, rows = _correlation_weighted_domain(frame, relative_cols, "relative improvement")
    weight_rows.extend(rows)
    consistency_b, rows = _correlation_weighted_domain(frame, consistency_cols, "consistency")
    weight_rows.extend(rows)
    reliability_b, rows = _correlation_weighted_domain(frame, reliability_cols, "reliability")
    weight_rows.extend(rows)
    absolute_c_b, rows = _correlation_weighted_domain(frame, absolute_c_cols, "absolute shrunk")
    weight_rows.extend(rows)

    dimensions_a = ["absolute_pressure_score", "relative_improvement_score", "consistency_score", "statistical_reliability_score"]
    frame["score_model_a_equal_weight"] = frame[dimensions_a].mean(axis=1)
    frame["score_model_b_correlation_adjusted"] = (
        absolute_b + relative_b + consistency_b + reliability_b
    ) / 4
    # Model C substitutes peer-shrunk absolute performance and favors evidence
    # that also improves on the bowler's own earlier normal-pressure baseline.
    frame["score_model_c_shrinkage_adjusted"] = (
        absolute_c_b + relative_b + consistency_b + reliability_b
    ) / 4
    # Model D uses reliability as one explicit discount on performance-only
    # evidence. Model B already includes reliability, so discounting B would
    # count the reliability domain twice.
    frame["score_model_d_performance_only"] = (
        absolute_b + relative_b + consistency_b
    ) / 3
    frame["score_model_d_reliability_adjusted"] = frame["score_model_d_performance_only"] * (
        0.50 + 0.50 * frame["statistical_reliability_score"].fillna(0) / 100
    )
    score_columns = [
        "score_model_a_equal_weight", "score_model_b_correlation_adjusted",
        "score_model_c_shrinkage_adjusted", "score_model_d_performance_only",
        "score_model_d_reliability_adjusted",
    ]
    for column in score_columns:
        frame[column] = frame[column].clip(0, 100)
    frame["overall_bowling_pressure_score"] = frame["score_model_c_shrinkage_adjusted"]

    reasons = []
    def under(row: object, field: str, minimum: float) -> bool:
        value = pd.to_numeric(getattr(row, field, np.nan), errors="coerce")
        return bool(pd.isna(value) or value < minimum)

    for row in frame.itertuples(index=False):
        failed = []
        if under(row, "pressure_matches", cfg.eligibility_pressure_matches):
            failed.append("pressure matches below minimum")
        if under(row, "pressure_innings", cfg.eligibility_pressure_innings):
            failed.append("pressure innings below minimum")
        if under(row, "pressure_legal_balls", cfg.eligibility_pressure_balls):
            failed.append("pressure legal balls below minimum")
        if under(row, "baseline_history_balls", cfg.eligibility_baseline_balls):
            failed.append("normal-pressure baseline history below minimum")
        if under(row, "baseline_coverage_fraction", cfg.baseline_coverage_minimum):
            failed.append(f"walk-forward baseline coverage below {cfg.baseline_coverage_minimum:.0%}")
        reasons.append("eligible" if not failed else "; ".join(failed))
    frame["eligibility_status"] = reasons
    frame["eligible_analytical_candidate"] = frame["eligibility_status"].eq("eligible")
    frame["score_interpretation"] = (
        "Descriptive pressure performance score; not a causal pressure effect, talent prediction, or final selection"
    )
    for column in score_columns + ["overall_bowling_pressure_score"]:
        if not frame[column].dropna().between(0, 100).all():
            raise ValueError(f"Composite score {column} escaped 0-100 bounds")

    frame["rank_overall"] = frame.groupby("competition_group", dropna=False)[
        "overall_bowling_pressure_score"
    ].rank(method="min", ascending=False)
    for dimension, column in (
        ("absolute", "absolute_pressure_score"),
        ("relative", "relative_improvement_score"),
        ("consistency", "consistency_score"),
        ("reliability", "statistical_reliability_score"),
    ):
        frame[f"rank_{dimension}"] = frame[column].where(frame["eligible_analytical_candidate"]).groupby(
            frame["competition_group"], dropna=False
        ).rank(method="min", ascending=False)
    frame["rank_eligible"] = frame["overall_bowling_pressure_score"].where(
        frame["eligible_analytical_candidate"]
    ).groupby(frame["competition_group"], dropna=False).rank(method="min", ascending=False)
    for model, source in (
        ("equal_weight", "score_model_a_equal_weight"),
        ("correlation_adjusted", "score_model_b_correlation_adjusted"),
        ("shrinkage_adjusted", "score_model_c_shrinkage_adjusted"),
        ("reliability_adjusted", "score_model_d_reliability_adjusted"),
    ):
        frame[f"rank_{model}"] = frame[source].where(frame["eligible_analytical_candidate"]).groupby(
            frame["competition_group"], dropna=False
        ).rank(method="min", ascending=False)

    weight_table = pd.DataFrame(weight_rows)
    return frame.sort_values(["competition_group", "rank_eligible", "bowler_id"], na_position="last"), weight_table


def eligibility_scenarios(scores: pd.DataFrame, config: BowlingModelConfig | None = None) -> pd.DataFrame:
    """Report every player under multiple exposure and history thresholds."""
    cfg = config or BowlingModelConfig()
    rows = []
    for pressure_balls, pressure_matches, pressure_innings, baseline_balls in cfg.eligibility_scenarios:
        candidates = scores.copy()
        candidates["scenario_eligible"] = (
            candidates["pressure_legal_balls"].ge(pressure_balls)
            & candidates["pressure_matches"].ge(pressure_matches)
            & candidates["pressure_innings"].ge(pressure_innings)
            & candidates["baseline_history_balls"].ge(baseline_balls)
            & candidates["baseline_coverage_fraction"].fillna(0).ge(cfg.baseline_coverage_minimum)
        )
        candidates["scenario_rank"] = candidates["overall_bowling_pressure_score"].where(
            candidates["scenario_eligible"]
        ).groupby(candidates["competition_group"], dropna=False).rank(method="min", ascending=False)
        for group, part in candidates.groupby("competition_group", dropna=False):
            rows.append({
                "scenario_pressure_balls": pressure_balls,
                "scenario_pressure_matches": pressure_matches,
                "scenario_pressure_innings": pressure_innings,
                "scenario_baseline_balls": baseline_balls,
                "competition_group": group,
                "players_total": int(len(part)),
                "players_eligible": int(part["scenario_eligible"].sum()),
                "players_ineligible": int((~part["scenario_eligible"]).sum()),
                "top10_bowler_ids": ",".join(
                    part.loc[part["scenario_eligible"]].nsmallest(10, "scenario_rank")["bowler_id"].astype("Int64").astype(str)
                ),
            })
    return pd.DataFrame(rows)


def integrate_bootstrap_reliability(scores: pd.DataFrame, uncertainty: pd.DataFrame) -> pd.DataFrame:
    """Fold bootstrap rank stability and interval width into reliability."""
    frame = scores.merge(
        uncertainty, on=PLAYER_KEYS, how="left", validate="one_to_one"
    )
    frequency = pd.to_numeric(frame.get("top10_selection_frequency"), errors="coerce")
    width = pd.to_numeric(frame.get("bootstrap_score_p95"), errors="coerce") - pd.to_numeric(
        frame.get("bootstrap_score_p05"), errors="coerce"
    )
    rank_std = pd.to_numeric(frame.get("bootstrap_rank_std"), errors="coerce")
    frame["bootstrap_top10_stability_score"] = _rank_score(
        frame.assign(_top10_frequency=frequency), "_top10_frequency", True
    )
    frame["bootstrap_interval_precision_score"] = _rank_score(
        frame.assign(_interval_width=width), "_interval_width", False
    )
    frame["bootstrap_rank_stability_score"] = _rank_score(
        frame.assign(_rank_std=rank_std), "_rank_std", False
    )
    evidence = frame[[
        "bootstrap_top10_stability_score", "bootstrap_interval_precision_score",
        "bootstrap_rank_stability_score",
    ]].mean(axis=1)
    has_bootstrap = frame["bootstrap_score_draws"].fillna(0).gt(0)
    prior_reliability = frame["statistical_reliability_score"].copy()
    frame.loc[has_bootstrap, "statistical_reliability_score"] = (
        0.5 * frame.loc[has_bootstrap, "statistical_reliability_score"]
        + 0.5 * evidence.loc[has_bootstrap]
    )
    reliability_delta = (frame["statistical_reliability_score"] - prior_reliability) / 4
    for column in (
        "score_model_a_equal_weight", "score_model_b_correlation_adjusted",
        "score_model_c_shrinkage_adjusted",
    ):
        frame[column] = frame[column] + reliability_delta
    frame["score_model_d_reliability_adjusted"] = frame["score_model_d_performance_only"] * (
        0.5 + 0.5 * frame["statistical_reliability_score"].fillna(0) / 100
    )
    for column in (
        "score_model_a_equal_weight", "score_model_b_correlation_adjusted",
        "score_model_c_shrinkage_adjusted", "score_model_d_reliability_adjusted",
    ):
        frame[column] = frame[column].clip(0, 100)
    frame["overall_bowling_pressure_score"] = frame["score_model_c_shrinkage_adjusted"]
    frame["rank_eligible"] = frame["overall_bowling_pressure_score"].where(
        frame["eligible_analytical_candidate"]
    ).groupby(frame["competition_group"], dropna=False).rank(method="min", ascending=False)
    frame["rank_overall"] = frame.groupby("competition_group", dropna=False)[
        "overall_bowling_pressure_score"
    ].rank(method="min", ascending=False)
    return frame
