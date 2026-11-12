"""Pressure-index sensitivity, scenario checks, summaries, and SVG charts."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.visualizations import horizontal_bars, histogram, multi_line_chart
from src.pressure.match_state import build_delivery_match_states
from src.pressure.pressure_detection import PressureConfig, add_high_pressure_flags, add_pressure_features

ROLES = {
    "batting": ("striker_id", "striker_id"),
    "bowling": ("bowler_id", "bowler_id"),
}
FORMULATIONS = ("weighted", "nonlinear")


def _scenario_data() -> pd.DataFrame:
    """Hand-built situations test monotonic behavior, not empirical calibration."""
    scenarios = [
        ("low-demand long chase", 10, 30, 9, 120, False, 0.0),
        ("high-demand, scarce wickets", 30, 12, 2, 120, False, 0.0),
        ("final-ball, one wicket", 18, 6, 1, 120, False, 0.0),
        ("comfortable first innings", np.nan, 100, 10, 120, True, 0.10),
        ("death phase, high chase rate", 30, 12, 5, 120, False, 0.0),
    ]
    rows = []
    for name, runs, balls, wickets, limit, first, pace_demand in scenarios:
        rrr = np.nan if first else runs * 6 / balls
        # For synthetic monotonicity checks only, map rate to a bounded strain
        # with a fixed saturating transform; real rows use empirical ranks.
        demand = pace_demand if first else 1 - np.exp(-rrr / 12)
        resource = 0.5 * (1 - balls / limit) + 0.5 * (1 - wickets / 10)
        weighted = (demand + resource) / 2
        nonlinear = 1 - np.sqrt((1 - demand) * (1 - resource))
        rows.append({
            "scenario": name,
            "runs_required": runs,
            "legal_balls_remaining": balls,
            "wickets_remaining": wickets,
            "required_run_rate": rrr,
            "demand_component_synthetic": demand,
            "resource_component": resource,
            "batting_pressure_weighted_synthetic": 100 * weighted,
            "batting_pressure_nonlinear_synthetic": 100 * nonlinear,
            "is_synthetic_not_match_observation": True,
        })
    return pd.DataFrame(rows)


def _threshold_analysis(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for role, (player_id, _) in ROLES.items():
        for formulation in FORMULATIONS:
            column = f"{role}_pressure_{formulation}"
            score = scores[column]
            options: list[tuple[str, float | None, pd.Series]] = []
            for threshold in (70, 80, 90):
                options.append((f"fixed_{threshold}", float(threshold), score.ge(threshold)))
            by_comp = scores.groupby("competition_group", dropna=False)[column].rank(method="average", pct=True) * 100
            for percentile in (90, 95):
                options.append((f"competition_percentile_{percentile}", float(percentile), by_comp.ge(percentile)))
            by_context = scores.groupby(["competition_group", "innings_phase"], dropna=False)[column].rank(
                method="average", pct=True
            ) * 100
            for percentile in (90, 95):
                options.append((f"context_percentile_{percentile}", float(percentile), by_context.ge(percentile)))
            for method, threshold, mask in options:
                selected = scores.loc[mask.fillna(False)]
                rows.append({
                    "role": role,
                    "formulation": formulation,
                    "classification_method": method,
                    "threshold_value": threshold,
                    "qualifying_deliveries": int(len(selected)),
                    "qualifying_matches": int(selected["match_id"].nunique()),
                    "qualifying_players": int(selected[player_id].nunique()),
                    "qualifying_competitions": int(selected["comp_id"].nunique()),
                    "share_of_all_deliveries_pct": float(len(selected) * 100 / max(len(scores), 1)),
                    "median_score_among_qualifiers": float(selected[column].median()) if len(selected) else None,
                    "threshold_is_candidate_not_final": True,
                })
    return pd.DataFrame(rows)


def _distribution_analysis(scores: pd.DataFrame) -> pd.DataFrame:
    keys = ["competition_group", "innings", "innings_phase"]
    metrics = [
        "batting_pressure_weighted", "batting_pressure_nonlinear",
        "bowling_pressure_weighted", "bowling_pressure_nonlinear",
    ]
    grouped = scores.groupby(keys, dropna=False)
    result = grouped.agg(
        deliveries=("match_id", "size"),
        matches=("match_id", "nunique"),
        legality_uncertain_states=("prior_legality_uncertain_deliveries", lambda s: int(s.gt(0).sum())),
        median_required_run_rate=("required_run_rate", "median"),
        median_current_run_rate=("current_run_rate", "median"),
        median_prior_boundary_event_rate=("historical_boundary_event_rate_prior", "median"),
        median_wickets_remaining=("wickets_remaining", "median"),
        median_balls_remaining=("legal_balls_remaining", "median"),
        median_resource_depletion=("resource_depletion_score", "median"),
    ).reset_index()
    for column in metrics:
        quantiles = grouped[column].quantile([0.1, 0.5, 0.9]).unstack()
        quantiles.columns = [f"{column}_p{int(q*100)}" for q in quantiles.columns]
        result = result.merge(quantiles.reset_index(), on=keys, how="left", validate="one_to_one")
    return result


def _scatter_svg(x: pd.Series, y: pd.Series, title: str, subtitle: str, path: Path, x_label: str) -> None:
    """Write a compact deterministic scatter plot for up to 4,000 sampled rows."""
    sample = pd.DataFrame(
        {"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(sample) > 4000:
        sample = sample.sample(4000, random_state=42)
    width, height = 1200, 720
    left, right, top, bottom = 100, 1120, 140, 610
    xv = sample.x.to_numpy(dtype=float)
    yv = sample.y.to_numpy(dtype=float)
    xmin, xmax = (float(np.nanmin(xv)), float(np.nanmax(xv))) if len(xv) else (0.0, 1.0)
    ymin, ymax = (float(np.nanmin(yv)), float(np.nanmax(yv))) if len(yv) else (0.0, 100.0)
    xmax = max(xmax, xmin + 1)
    ymax = max(ymax, ymin + 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f5f7fb"/>',
        f'<text x="70" y="60" font-family="Arial" font-size="27" font-weight="700" fill="#182230">{escape(title)}</text>',
        f'<text x="70" y="94" font-family="Arial" font-size="14" fill="#667085">{escape(subtitle)}</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#667085"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#667085"/>',
        f'<text x="{(left+right)/2}" y="662" text-anchor="middle" font-family="Arial" font-size="14" fill="#182230">{escape(x_label)}</text>',
        f'<text transform="translate(30 {(top+bottom)/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="14" fill="#182230">Weighted batting pressure (0–100)</text>',
    ]
    for xv_i, yv_i in zip(xv, yv):
        px = left + (xv_i - xmin) * (right - left) / (xmax - xmin)
        py = bottom - (yv_i - ymin) * (bottom - top) / (ymax - ymin)
        parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2.1" fill="#245b8f" opacity="0.25"/>')
    parts.append('<text x="70" y="696" font-family="Arial" font-size="11" fill="#98a2b3">SupeRR Selector · descriptive EDA · no result used for score construction</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _generate_figures(scores: pd.DataFrame, distributions: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    figures: list[Path] = []
    path = out_dir / "pressure_score_distribution.svg"
    histogram(scores["batting_pressure_weighted"], "Batting pressure score distribution", "Weighted index across first innings and chases; delivery-weighted states", path, max_bin=100)
    figures.append(path)

    phase = scores.groupby(["competition_group", "innings_phase"], dropna=False)["batting_pressure_weighted"].median().unstack()
    path = out_dir / "pressure_by_innings_phase.svg"
    horizontal_bars(
        [f"{group} · {phase_name}" for group, row in phase.iterrows() for phase_name in phase.columns],
        [float(phase.loc[group, phase_name]) if pd.notna(phase.loc[group, phase_name]) else 0 for group, row in phase.iterrows() for phase_name in phase.columns],
        "Median batting pressure by innings phase", "Context groups are reported separately; lower/upper state bounds retained in delivery scores", path,
        value_format="{:.1f}",
    )
    figures.append(path)
    comp = scores.groupby("competition_group")["batting_pressure_weighted"].median().sort_values()
    path = out_dir / "pressure_by_competition.svg"
    horizontal_bars(comp.index.tolist(), comp.tolist(), "Median batting pressure by competition", "IPL, T20I, and Other T20 are kept separate", path, value_format="{:.1f}")
    figures.append(path)

    path = out_dir / "pressure_versus_required_run_rate.svg"
    _scatter_svg(scores.loc[scores["is_chasing"], "required_run_rate"], scores.loc[scores["is_chasing"], "batting_pressure_weighted"], "Pressure versus required run rate", "Chase deliveries only; sample capped at 4,000 points for legibility", path, "Required run rate (runs per over)")
    figures.append(path)
    path = out_dir / "pressure_versus_wickets_remaining.svg"
    _scatter_svg(scores["wickets_remaining"], scores["batting_pressure_weighted"], "Pressure versus wickets remaining", "All innings; lower wicket availability generally increases batting resource strain", path, "Wickets remaining")
    figures.append(path)
    path = out_dir / "pressure_versus_balls_remaining.svg"
    _scatter_svg(scores["legal_balls_remaining"], scores["batting_pressure_weighted"], "Pressure versus legal balls remaining", "All innings; uncertain prior legal balls are represented by score bounds", path, "Legal balls remaining (midpoint of explicit bounds)")
    figures.append(path)

    comparison = scores.groupby(pd.cut(scores["batting_pressure_weighted"], np.arange(0, 110, 10), include_lowest=True), observed=False).agg(
        weighted=("batting_pressure_weighted", "median"), nonlinear=("batting_pressure_nonlinear", "median")
    ).dropna()
    labels = [f"{interval.left:.0f}–{interval.right:.0f}" for interval in comparison.index]
    path = out_dir / "alternative_formulation_comparison.svg"
    multi_line_chart(labels, {"weighted": comparison["weighted"].tolist(), "nonlinear": comparison["nonlinear"].tolist()}, "Alternative pressure formulations", "Median scores by weighted-index decile; formulations share normalized demand/resource components", path)
    figures.append(path)
    by_comp = scores.groupby("competition_group").agg(
        fixed=("batting_high_weighted_fixed_80", "sum"),
        competition_p90=("batting_high_weighted_competition_p90", "sum"),
        context_p90=("batting_high_weighted_context_p90", "sum"),
    )
    path = out_dir / "high_pressure_delivery_distribution.svg"
    horizontal_bars(
        [f"{group} · {method}" for group, row in by_comp.iterrows() for method in by_comp.columns],
        [float(by_comp.loc[group, method]) for group, row in by_comp.iterrows() for method in by_comp.columns],
        "Candidate high-pressure delivery counts", "Weighted batting index: fixed 80, competition top 10%, and phase-context top 10%", path,
    )
    figures.append(path)
    return figures


def _close_chase_check(scores: pd.DataFrame, matches: pd.DataFrame) -> dict[str, Any]:
    """Ex-post descriptive check, isolated from score construction."""
    result = matches[["match_id", "result"]].copy()
    text = result["result"].fillna("").astype(str)
    run_margin = pd.to_numeric(text.str.extract(r"(?i)won\s+by\s+(\d+)\s+runs?", expand=False), errors="coerce")
    wicket_margin = pd.to_numeric(text.str.extract(r"(?i)won\s+by\s+(\d+)\s+wickets?", expand=False), errors="coerce")
    result["close_ex_post"] = run_margin.le(5) | wicket_margin.le(2)
    final_states = scores.loc[scores["innings"].eq(2)].groupby("match_id", as_index=False).tail(1)
    checked = final_states[["match_id", "batting_pressure_weighted", "bowling_pressure_weighted"]].merge(
        result[["match_id", "close_ex_post"]], on="match_id", how="left", validate="one_to_one"
    )
    return {
        "matches_with_second_innings_delivery_state": int(len(checked)),
        "close_results_by_margin_ex_post": int(checked["close_ex_post"].fillna(False).sum()),
        "median_last_recorded_batting_pressure_close_ex_post": float(checked.loc[checked.close_ex_post.eq(True), "batting_pressure_weighted"].median()),
        "median_last_recorded_batting_pressure_other_results": float(checked.loc[checked.close_ex_post.eq(False), "batting_pressure_weighted"].median()),
        "outcome_fields_used_for_pressure_scores": False,
        "interpretation": "post-hoc descriptive association only; final results are not score inputs or validation labels for weights",
    }


def write_pressure_analysis(
    states: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    report_dir: str | Path,
    figure_dir: str | Path,
    processed_dir: str | Path,
    config: PressureConfig | None = None,
) -> dict[str, Any]:
    """Create score files, sensitivity reports, methodology, and figures."""
    reports = Path(report_dir)
    reports.mkdir(parents=True, exist_ok=True)
    config = config or PressureConfig()
    scores = add_high_pressure_flags(add_pressure_features(states, config), config)
    dist = _distribution_analysis(scores)
    thresholds = _threshold_analysis(scores)
    scenarios = _scenario_data()
    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    base_columns = [
        "match_id", "comp_id", "comp_name", "match_format", "season", "match_date",
        "innings", "innings_ball_num", "overid", "over_ball_num",
        "batting_team_id", "bowling_team_id",
        "striker_id", "bowler_id", "competition_group", "is_chasing", "is_super_over",
        "innings_phase", "match_situation", "target_source", "limit_source",
        "runs_scored_before", "wickets_lost_before", "wickets_remaining",
        "legal_balls_bowled_known", "legal_balls_bowled_possible",
        "completed_overs_lower", "completed_overs_upper",
        "balls_into_over_lower", "balls_into_over_upper",
        "legal_balls_remaining_min", "legal_balls_remaining_max", "legal_balls_remaining",
        "current_run_rate_lower", "current_run_rate_upper", "current_run_rate",
        "chase_target", "runs_required", "runs_to_defend",
        "runs_required_per_remaining_wicket", "required_run_rate_lower",
        "required_run_rate_upper", "required_run_rate", "prior_legality_uncertain_deliveries",
        "state_reliability",
    ]
    base_columns = [column for column in base_columns if column in scores]
    states.loc[:, base_columns].to_csv(processed / "delivery_match_states.csv", index=False)
    feature_columns = [
        column for column in scores.columns
        if column in base_columns or column.startswith((
            "resource_", "innings_progress_", "wicket_loss_", "chase_rate_",
            "bowling_resource_", "historical_boundary_", "remaining_target_",
            "batting_demand_", "bowling_threat_",
        ))
    ]
    score_columns = [
        column for column in scores.columns
        if column in base_columns or column.startswith((
            "batting_pressure_", "bowling_pressure_", "batting_high_", "bowling_high_",
        ))
    ]
    scores.loc[:, feature_columns].to_csv(processed / "pressure_features.csv", index=False)
    scores.loc[:, score_columns].to_csv(processed / "pressure_scores.csv", index=False)
    dist.to_csv(reports / "pressure_distribution.csv", index=False)
    thresholds.to_csv(reports / "pressure_threshold_analysis.csv", index=False)
    scenarios.to_csv(reports / "pressure_scenario_tests.csv", index=False)
    figures = _generate_figures(scores, dist, Path(figure_dir))
    close_check = _close_chase_check(scores, tables["matches"])

    key_counts = {
        "delivery_state_rows": int(len(scores)),
        "first_innings_rows": int(scores["innings"].eq(1).sum()),
        "chase_rows_with_target": int((scores["is_chasing"] & scores["chase_target"].notna()).sum()),
        "prior_legality_uncertain_state_rows": int(scores["prior_legality_uncertain_deliveries"].gt(0).sum()),
        "current_delivery_legality_uncertain_rows": int(scores["legal_status_unknown_at_delivery"].sum()),
        "phase_uncertain_rows": int(scores["innings_phase"].eq("Uncertain phase").sum()),
        "super_over_rows": int(scores["is_super_over"].sum()),
        "target_unknown_chase_rows": int((scores["is_chasing"] & scores["chase_target"].isna()).sum()),
    }
    scenario_order = scenarios.set_index("scenario")["batting_pressure_weighted_synthetic"]
    monotonicity = {
        "more_demand_and_fewer_wickets_30_from_12_2wkts_exceeds_10_from_30_9wkts": bool(
            scenario_order["high-demand, scarce wickets"] > scenario_order["low-demand long chase"]
        ),
        "final_ball_one_wicket_exceeds_high_demand_12_ball_scenario": bool(
            scenario_order["final-ball, one wicket"] > scenario_order["high-demand, scarce wickets"]
        ),
        "comfortable_first_innings_below_chase_stress": bool(
            scenario_order["comfortable first innings"] < scenario_order["high-demand, scarce wickets"]
        ),
    }
    recommended_candidate = thresholds.loc[
        thresholds.role.eq("batting")
        & thresholds.formulation.eq("weighted")
        & thresholds.classification_method.eq("context_percentile_90")
    ].iloc[0]
    fixed_80_candidate = thresholds.loc[
        thresholds.role.eq("batting")
        & thresholds.formulation.eq("weighted")
        & thresholds.classification_method.eq("fixed_80")
    ].iloc[0]
    summary = {
        "scope": "candidate descriptive pressure indices; no player ratings, rankings, or outcome-trained model",
        "data_counts": key_counts,
        "median_pressure_by_competition": scores.groupby("competition_group")[
            ["batting_pressure_weighted", "batting_pressure_nonlinear", "bowling_pressure_weighted", "bowling_pressure_nonlinear"]
        ].median().reset_index().to_dict("records"),
        "fixed_threshold_80_weighted_batting": thresholds.loc[
            thresholds.role.eq("batting") & thresholds.formulation.eq("weighted")
            & thresholds.classification_method.eq("fixed_80")
        ].iloc[0].to_dict(),
        "provisional_high_pressure_candidate": recommended_candidate.to_dict(),
        "synthetic_monotonicity": monotonicity,
        "close_chase_check_ex_post": close_check,
        "alternative_formulation_median_absolute_difference": float(
            (scores["batting_pressure_weighted"] - scores["batting_pressure_nonlinear"]).abs().median()
        ),
        "thresholds_are_candidates_only": True,
    }
    method_lines = [
        "SUPERR SELECTOR — PRESSURE METHOD AND LIMITATIONS",
        "=" * 78, "",
        "Purpose: reconstruct pre-delivery states and provide two transparent, role-specific descriptive pressure indices. The values are not probabilities, player ratings, or rankings.",
        "",
        "STATE RECONSTRUCTION",
        "- Runs, dismissals, and legal balls are accumulated only from earlier delivery rows in the same innings. The current delivery outcome is never included in its own state.",
        "- Source over/ball ordinals are retained, and completed legal overs plus balls into the over are provided as lower/upper values when prior legality is uncertain.",
        "- Wides and the observed Overstep no-ball label do not consume a legal ball; byes and leg-byes do. Unclassified legality remains unknown.",
        "- For each state, balls remaining are bounded: lower bound assumes all prior unknown-legality deliveries were legal; upper bound assumes none were legal. Point scores use the midpoint and retain lower/upper score bounds.",
        "- A dismissal reduces wickets only when is_wicket is true. Each flagged source row counts one dismissal; multiple flagged rows in an innings sequence each count. The source schema has a boolean wicket flag and one dismissal type, so multiple dismissals encoded inside one row cannot be separately counted.",
        "- Revised target is used only when recorded for the second innings; otherwise the first innings total + 1 defines the chase target. Tie-break innings are paired sequentially; later super-over innings target the preceding super-over total + 1.",
        "- First/second innings revised-over fields are treated as legal-ball limits because observed values are multiples of six (30–114), while the original field names remain ambiguous. Missing limits default to 120; super overs default to 6.",
        "",
        "INDEX DEFINITIONS",
        f"- Empirical ranks are calculated walk-forward within competition group (IPL, T20I, Other T20) and innings phase. Each calendar month uses only dates through the prior month; rank r is bounded to [0,1], and a neutral 0.5 rank is used until at least {config.minimum_reference_states} earlier states exist. This normalizes local scoring environments but is not a common absolute risk probability.",
        "- Chase demand D is the mean of the empirical percentile of (required run rate − current run rate) and the empirical percentile of the fraction of the target still required. For first innings, D is the percentile of under-scoring versus first-innings current-run-rate distribution in the same competition/phase; at the start, missing current rate is neutral (0.5).",
        "- Resource depletion R is the mean of legal-ball progress fraction and wickets lost / 10. Unknown prior legality gives lower/upper bounds on legal-ball progress and scores.",
        f"- Weighted index A = 100 × [wD × D + (1 − wD) × R], with configured wD = {config.demand_weight:.2f}. Equal default weight gives equal weight to demand and depleted-resource domains; component subfeatures are also averaged equally. There is no outcome-trained ground truth, so weights are transparent design parameters, not statistically estimated optimum weights.",
        f"- Nonlinear index B = 100 × [1 − ((1 − D) × (1 − R))^gamma], with configured gamma = {config.nonlinear_interaction_power:.2f}. This symmetric geometric interaction rises more sharply when both demand and resource strain are high. It is an alternative sensitivity formulation, not a selected winner.",
        "- demand_weight, nonlinear_interaction_power, minimum_reference_states, the fixed-flag threshold, and percentile flags are configurable with PressureConfig in src/pressure/pressure_detection.py.",
        "- Bowling indices use role-specific threat: in chases, easier opponent demand is 1 − D and opponent resources are 1 − R; in the first innings, the pace percentile is paired with pace percentile × wickets remaining / 10. These are team-context interpretations, not individual bowler workload.",
        "- Bowling context includes runs to defend, required rate, balls and wickets remaining. A prior-date phase boundary-event rate is included as a contextual feature but is not part of the score; current/future delivery outcomes are not pre-delivery information.",
        "",
        "CLASSIFICATION AND SENSITIVITY",
        "- Candidate definitions compared are fixed score cutoffs 70/80/90, competition-group top 10%/5%, and competition-plus-phase top 10%/5%.",
        "- Provisional recommendation for high-pressure opportunity labels: weighted-index context-adjusted 90th percentile within competition group × innings phase. It directly addresses scoring-environment and phase differences while avoiding a convenient absolute cutoff; it means the most strained 10% of observed contexts in each stratum, not a common probability.",
        f"- In this dataset, that candidate marks {int(recommended_candidate.qualifying_deliveries):,} batting-pressure deliveries across {int(recommended_candidate.qualifying_matches):,} matches and {int(recommended_candidate.qualifying_players):,} batters. Fixed weighted score >=80 marks {int(fixed_80_candidate.qualifying_deliveries):,} deliveries across {int(fixed_80_candidate.qualifying_matches):,} matches and {int(fixed_80_candidate.qualifying_players):,} batters.",
        "- This is a relative descriptive candidate, not an empirically optimal psychological threshold. Fixed cutoffs and competition-only percentiles remain sensitivity comparisons; no player rankings use these labels.",
        "- Examine pressure_threshold_analysis.csv and pressure_distribution.csv before choosing any downstream definition. Classifications can differ substantially by formula, group, and phase.",
        "",
        "LIMITATIONS",
        "- There is no externally labeled, pre-delivery pressure ground truth. Index weights and percentiles describe an explicit construct; they do not prove psychological pressure or predict success.",
        "- Empirical normalization is computed on the supplied dataset and is not a time-held-out calibration. Avoid interpreting it as a real-time prospective scale.",
        "- The dataset has 267 deliveries with uncertain legal-ball status; 266 with uncertain bowler-extra attribution; 16 abandoned matches without deliveries; incomplete innings and missing result/target details can affect context.",
        "- T20 and T20I are kept separate; Other T20 remains its own group. Sparse competition-phase combinations have noisier empirical ranks.",
        "- Close-result checks in pressure_validation.txt are ex-post descriptive associations only. Match outcomes are not used to reconstruct states, normalize indices, set weights, or classify deliveries.",
    ]
    (reports / "pressure_methodology.txt").write_text("\n".join(method_lines) + "\n", encoding="utf-8")

    actual = scores.groupby(["competition_group", "innings_phase"], dropna=False).agg(
        deliveries=("match_id", "size"),
        matches=("match_id", "nunique"),
        median_batting_pressure=("batting_pressure_weighted", "median"),
        median_bowling_pressure=("bowling_pressure_weighted", "median"),
        high_weighted_batting_fixed_80=("batting_high_weighted_fixed_80", "sum"),
        median_required_run_rate=("required_run_rate", "median"),
        median_wickets_remaining=("wickets_remaining", "median"),
        median_legal_balls_remaining=("legal_balls_remaining", "median"),
        median_prior_boundary_event_rate=("historical_boundary_event_rate_prior", "median"),
    ).reset_index()
    actual_text = actual.to_string(index=False, float_format=lambda v: f"{v:.2f}")
    validation_lines = [
        "SUPERR SELECTOR — PRESSURE VALIDATION",
        "=" * 78, "",
        f"Rows scored: {len(scores):,}; weighted and nonlinear indices are both bounded to [0,100].",
        f"Data-state counts: {key_counts}",
        "",
        "OBSERVED DISTRIBUTIONS (descriptive)",
        actual_text,
        "",
        "SYNTHETIC SCENARIOS (not observed matches)",
        scenarios.to_string(index=False, float_format=lambda v: f"{v:.2f}"),
        f"Monotonic checks: {monotonicity}",
        "",
        "ALTERNATIVE FORMULATIONS",
        f"Median absolute batting-score difference between A and B: {summary['alternative_formulation_median_absolute_difference']:.2f} points.",
        "Compare pressure_distribution.csv and pressure_threshold_analysis.csv for score distributions and qualification sensitivity.",
        (
            "Provisional high-pressure opportunity definition: context-adjusted P90 for the weighted batting index "
            f"({int(recommended_candidate.qualifying_deliveries):,} deliveries, {int(recommended_candidate.qualifying_matches):,} matches, "
            f"{int(recommended_candidate.qualifying_players):,} batters). A fixed weighted score >=80 identifies "
            f"{int(fixed_80_candidate.qualifying_deliveries):,} deliveries. The relative P90 better controls group/phase context; "
            "this is a fairness-based design recommendation, not a statistically validated ground-truth cutoff."
        ),
        "",
        "CLOSE CHASES — POST-HOC CHECK ONLY",
        str(close_check),
        "",
        "Caveat: these checks establish software behavior and descriptive face validity only. They do not statistically validate a latent psychological-pressure construct. Outcome fields are excluded from pressure feature construction.",
    ]
    (reports / "pressure_validation.txt").write_text("\n".join(validation_lines) + "\n", encoding="utf-8")
    return {
        "scores": scores,
        "distribution": dist,
        "thresholds": thresholds,
        "scenarios": scenarios,
        "summary": summary,
        "figures": figures,
    }


def run_pressure_engine(
    tables: dict[str, pd.DataFrame],
    report_dir: str | Path,
    figure_dir: str | Path,
    processed_dir: str | Path,
    config: PressureConfig | None = None,
) -> dict[str, Any]:
    """Reconstruct the states, then independently build pressure features."""
    states = build_delivery_match_states(tables)
    return write_pressure_analysis(states, tables, report_dir, figure_dir, processed_dir, config)
