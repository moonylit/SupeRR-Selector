"""Phase 6 bowling-pressure evaluation, uncertainty, outputs, and visualisations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import BowlingModelConfig
from src.evaluation.bowling_uncertainty import bootstrap_bowler_scores
from src.models.bowling_baseline import (
    add_walk_forward_bowling_baselines,
    assign_walk_forward_pressure_bands,
)
from src.models.bowling_features import aggregate_bowling_counts, build_bowling_delivery_features
from src.models.bowling_scoring import (
    eligibility_scenarios,
    integrate_bootstrap_reliability,
    score_bowlers,
)
from src.models.bowling_statistics import build_bowler_pressure_differentials, build_pressure_statistics

LOGGER = logging.getLogger(__name__)
PLAYER_KEYS = ["competition_group", "bowler_id"]


def _compare_rankings(base: pd.DataFrame, alternative: pd.DataFrame, name: str, note: str = "") -> dict[str, Any]:
    left = base.set_index(PLAYER_KEYS)
    right = alternative.set_index(PLAYER_KEYS)
    shared = left[["overall_bowling_pressure_score"]].join(
        right[["overall_bowling_pressure_score"]], how="inner", lsuffix="_base", rsuffix="_scenario"
    ).dropna()
    left_rank = left["rank_eligible"].rename("base")
    right_rank = right["rank_eligible"].rename("scenario")
    ranks = pd.concat([left_rank, right_rank], axis=1, join="inner").dropna()
    top_left = set(
        left.loc[left["eligible_analytical_candidate"]].sort_values("rank_eligible")
        .groupby(level=0, group_keys=False).head(10).index
    )
    top_right = set(
        right.loc[right["eligible_analytical_candidate"]].sort_values("rank_eligible")
        .groupby(level=0, group_keys=False).head(10).index
    )
    return {
        "scenario": name,
        "players_compared": int(len(shared)),
        "top10_overlap_player_group_pairs": int(len(top_left & top_right)),
        "top10_base_pairs": int(len(top_left)),
        "top10_scenario_pairs": int(len(top_right)),
        "eligible_rank_spearman": float(ranks["base"].corr(ranks["scenario"], method="spearman")) if len(ranks) > 1 else np.nan,
        "median_absolute_score_change": float((shared["overall_bowling_pressure_score_base"] - shared["overall_bowling_pressure_score_scenario"]).abs().median()) if len(shared) else np.nan,
        "note": note,
    }


def _score_pressure_scenario(
    deliveries: pd.DataFrame,
    pressure_band: pd.Series,
    config: BowlingModelConfig,
) -> pd.DataFrame:
    scenario = deliveries.copy()
    scenario["bowling_pressure_band_walk_forward"] = pressure_band.to_numpy()
    detail, player, match = build_pressure_statistics(scenario)
    diff = build_bowler_pressure_differentials(scenario, player, match, config)
    scores, _ = score_bowlers(diff, config)
    return scores


def _rescore_baseline_strategy(
    deliveries: pd.DataFrame,
    scores: pd.DataFrame,
    config: BowlingModelConfig,
    strategy: str,
) -> pd.DataFrame:
    """Recompute baseline-relative scores while holding pressure outcomes fixed."""
    based, _ = add_walk_forward_bowling_baselines(deliveries, config, strategy=strategy)
    _detail, player, match = build_pressure_statistics(based)
    differential = build_bowler_pressure_differentials(based, player, match, config)
    alternate, _ = score_bowlers(differential, config)
    names = scores[PLAYER_KEYS + ["player_name"]].drop_duplicates(PLAYER_KEYS)
    alternate = alternate.merge(names, on=PLAYER_KEYS, how="left", validate="one_to_one")
    bootstrap_columns = [
        "bootstrap_score_draws", "top10_selection_frequency", "bootstrap_score_p95",
        "bootstrap_score_p05", "bootstrap_rank_std",
    ]
    bootstrap = scores[PLAYER_KEYS + bootstrap_columns].drop_duplicates(PLAYER_KEYS)
    return integrate_bootstrap_reliability(alternate, bootstrap)


def _baseline_approach_rows(
    deliveries: pd.DataFrame,
    primary: pd.DataFrame,
    config: BowlingModelConfig,
) -> list[dict[str, Any]]:
    """Compare the strict, format-hierarchical, relaxed-coverage and broad pools."""
    approaches: list[tuple[str, pd.DataFrame, str]] = [
        ("B. Format-hierarchical (primary)", primary,
         "Competition-aware peer hierarchy; personal normal history accumulates across the same format, phase and innings context."),
        ("A. Existing selected-context history (legacy)", _rescore_baseline_strategy(
            deliveries, primary, config, "legacy_hierarchical"
        ), "Historical behavior: personal history is measured at the selected peer hierarchy, which can reset at exact competition/season level when that peer pool meets its minimum."),
        ("D. Broader T20 pool sensitivity", _rescore_baseline_strategy(
            deliveries, primary, config, "broader_t20"
        ), "Sensitivity only: cross-format peer rates retain phase/context but may mix incompatible scoring environments."),
    ]
    relaxed = primary.copy()
    relaxed["eligible_analytical_candidate"] = (
        relaxed["pressure_matches"].ge(config.eligibility_pressure_matches)
        & relaxed["pressure_innings"].ge(config.eligibility_pressure_innings)
        & relaxed["pressure_legal_balls"].ge(config.eligibility_pressure_balls)
        & relaxed["baseline_history_balls"].ge(config.eligibility_baseline_balls)
        & relaxed["baseline_coverage_fraction"].fillna(0).ge(0.10)
    )
    relaxed["rank_eligible"] = relaxed["overall_bowling_pressure_score"].where(
        relaxed["eligible_analytical_candidate"]
    ).groupby(relaxed["competition_group"], dropna=False).rank(method="min", ascending=False)
    approaches.append(("C. Format hierarchy, coverage ≥10%", relaxed,
                       "Same baselines and scores as primary; coverage gate reduced from 25% to 10%, with the existing reliability domain retaining coverage/history penalties."))

    rows: list[dict[str, Any]] = []
    for name, scenario, note in approaches:
        row = _compare_rankings(primary, scenario, name, note)
        primary_pool = (
            primary["pressure_matches"].ge(config.eligibility_pressure_matches)
            & primary["pressure_innings"].ge(config.eligibility_pressure_innings)
            & primary["pressure_legal_balls"].ge(config.eligibility_pressure_balls)
            & primary["baseline_coverage_fraction"].fillna(0).ge(config.baseline_coverage_minimum)
        )
        scenario_coverage = 0.10 if name.startswith("C.") else config.baseline_coverage_minimum
        scenario_pool = (
            scenario["pressure_matches"].ge(config.eligibility_pressure_matches)
            & scenario["pressure_innings"].ge(config.eligibility_pressure_innings)
            & scenario["pressure_legal_balls"].ge(config.eligibility_pressure_balls)
            & scenario["baseline_coverage_fraction"].fillna(0).ge(scenario_coverage)
        )
        left = primary.loc[primary_pool].set_index(PLAYER_KEYS)
        right = scenario.loc[scenario_pool].set_index(PLAYER_KEYS)
        left_ranks = left["overall_bowling_pressure_score"].groupby(level=0).rank(
            method="min", ascending=False
        ).rename("primary")
        right_ranks = right["overall_bowling_pressure_score"].groupby(level=0).rank(
            method="min", ascending=False
        ).rename("scenario")
        common_ranks = pd.concat([left_ranks, right_ranks], axis=1, join="inner").dropna()
        left_top = set(left.assign(_rank=left_ranks).sort_values("_rank").groupby(
            level=0, group_keys=False
        ).head(10).index)
        right_top = set(right.assign(_rank=right_ranks).sort_values("_rank").groupby(
            level=0, group_keys=False
        ).head(10).index)
        row["pressure_coverage_top10_overlap"] = len(left_top & right_top)
        row["pressure_coverage_top10_primary"] = len(left_top)
        row["pressure_coverage_top10_scenario"] = len(right_top)
        row["pressure_coverage_rank_spearman"] = (
            common_ranks["primary"].corr(common_ranks["scenario"], method="spearman")
            if len(common_ranks) > 1 else np.nan
        )
        row["note"] += " Bootstrap evidence is held fixed from primary; alternative-baseline uncertainty is not independently bootstrapped."
        for group, label in (("IPL", "ipl"), ("T20I", "t20i"), ("Other T20", "other_t20")):
            row[f"eligible_{label}"] = int(
                scenario.loc[scenario["competition_group"].eq(group), "eligible_analytical_candidate"].sum()
            )
            part = scenario.loc[scenario["competition_group"].eq(group)]
            pressure = (
                part["pressure_matches"].ge(config.eligibility_pressure_matches)
                & part["pressure_innings"].ge(config.eligibility_pressure_innings)
                & part["pressure_legal_balls"].ge(config.eligibility_pressure_balls)
            )
            row[f"pressure_gate_{label}"] = int(pressure.sum())
            coverage = part["baseline_coverage_fraction"].fillna(0).ge(
                0.10 if name.startswith("C.") else config.baseline_coverage_minimum
            )
            row[f"coverage_gate_{label}"] = int((pressure & coverage).sum())
            history = part["baseline_history_balls"].ge(config.eligibility_baseline_balls)
            row[f"history_gate_{label}"] = int((pressure & coverage & history).sum())
            row[f"median_history_balls_{label}"] = float(part.loc[pressure, "baseline_history_balls"].median()) if pressure.any() else np.nan
            row[f"median_peer_balls_{label}"] = float(part.loc[pressure, "baseline_peer_balls"].median()) if pressure.any() else np.nan
            row[f"baseline_levels_{label}"] = "; ".join(
                f"{int(level)}:{int(count)}" for level, count in
                part.loc[pressure, "baseline_selected_level"].value_counts(dropna=False).sort_index().items()
            )
            row[f"baseline_sources_{label}"] = "; ".join(
                f"{source}:{int(count)}" for source, count in
                part.loc[pressure, "baseline_pool_source"].value_counts(dropna=False).items()
            )
        rows.append(row)
    return rows


def _with_alternative_overall(scores: pd.DataFrame, absolute_score: pd.Series) -> pd.DataFrame:
    alternative = scores.copy()
    alternative["overall_bowling_pressure_score"] = (
        absolute_score + alternative["relative_improvement_score"]
        + alternative["consistency_score"] + alternative["statistical_reliability_score"]
    ) / 4
    alternative["rank_eligible"] = alternative["overall_bowling_pressure_score"].where(
        alternative["eligible_analytical_candidate"]
    ).groupby(alternative["competition_group"], dropna=False).rank(method="min", ascending=False)
    return alternative


def _sensitivity_absolute_with_metric(
    scores: pd.DataFrame, metric: str, higher_is_better: bool, component_key: str
) -> pd.DataFrame:
    score = pd.to_numeric(scores[metric], errors="coerce").groupby(
        scores["competition_group"], dropna=False
    ).rank(method="average", pct=True)
    replacement = score if higher_is_better else 1 - score
    components = [
        "absolute_economy_high_score", "absolute_wickets_per_legal_over_high_score",
        "absolute_bowler_dot_rate_high_score", "absolute_boundary_concession_rate_high_score",
        "absolute_bowler_extras_per_legal_over_high_score",
    ]
    adjusted = scores[components].copy()
    component_by_metric = {
        "economy": "absolute_economy_high_score",
        "wickets_per_legal_over": "absolute_wickets_per_legal_over_high_score",
        "bowler_dot_rate": "absolute_bowler_dot_rate_high_score",
        "boundary_concession_rate": "absolute_boundary_concession_rate_high_score",
        "bowler_extras_per_legal_over": "absolute_bowler_extras_per_legal_over_high_score",
    }
    adjusted[component_by_metric[component_key]] = replacement
    return _with_alternative_overall(scores, adjusted.mean(axis=1))


def _absolute_score_from_counts(frame: pd.DataFrame) -> pd.DataFrame:
    high = frame.loc[frame["bowling_pressure_band_walk_forward"].eq("high")]
    if high.empty:
        return pd.DataFrame(columns=PLAYER_KEYS + ["absolute_sensitivity_score"])
    counts = aggregate_bowling_counts(high, PLAYER_KEYS)
    metrics = [
        ("economy", False), ("wickets_per_legal_over", True),
        ("bowler_dot_rate", True), ("boundary_concession_rate", False),
        ("bowler_extras_per_legal_over", False),
    ]
    scores = []
    for column, higher in metrics:
        ranks = counts[column].groupby(counts["competition_group"], dropna=False).rank(method="average", pct=True)
        scores.append(ranks if higher else 1 - ranks)
    counts["absolute_sensitivity_score"] = pd.concat(scores, axis=1).mean(axis=1) * 100
    return counts[PLAYER_KEYS + ["absolute_sensitivity_score"]]


def _sensitivity_analysis(
    deliveries: pd.DataFrame,
    scores: pd.DataFrame,
    config: BowlingModelConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    eligibility = eligibility_scenarios(scores, config)
    for model, column in (
        ("equal-weight composite", "score_model_a_equal_weight"),
        ("correlation-adjusted composite", "score_model_b_correlation_adjusted"),
        ("shrinkage-adjusted composite", "score_model_c_shrinkage_adjusted"),
        ("reliability-adjusted composite", "score_model_d_reliability_adjusted"),
    ):
        alternate = scores.copy()
        alternate["overall_bowling_pressure_score"] = alternate[column]
        alternate["rank_eligible"] = alternate[column].where(alternate["eligible_analytical_candidate"]).groupby(
            alternate["competition_group"], dropna=False
        ).rank(method="min", ascending=False)
        rows.append(_compare_rankings(scores, alternate, model, "Composite model weight sensitivity"))

    strict_context = scores["baseline_pool_source"].eq(
        "competition_group+comp_id+innings_phase+innings_context"
    )
    for name, mask in (
        ("exact competition/phase/innings-context baseline only", strict_context),
        ("hierarchical fallback baseline only", ~strict_context & scores["baseline_coverage_fraction"].fillna(0).gt(0)),
    ):
        alternate = scores.copy()
        alternate["eligible_analytical_candidate"] &= mask
        alternate["rank_eligible"] = alternate["overall_bowling_pressure_score"].where(
            alternate["eligible_analytical_candidate"]
        ).groupby(alternate["competition_group"], dropna=False).rank(method="min", ascending=False)
        rows.append(_compare_rankings(scores, alternate, name, "Baseline context-matching sensitivity; scores use the selected walk-forward estimate"))

    rows.extend(_baseline_approach_rows(deliveries, scores, config))

    # Recompute the normal-history prior contribution at multiple metric-
    # specific exposure strengths; all other dimensions and source counts stay fixed.
    strength_fields = {
        "economy": ("baseline_economy_raw", "baseline_economy_peer", "pressure_economy", "economy_prior_balls", -1),
        "wickets_per_over": ("baseline_wickets_per_over_raw", "baseline_wickets_per_over_peer", "pressure_wickets_per_over", "wicket_prior_balls", 1),
        "dot_rate": ("baseline_dot_rate_raw", "baseline_dot_rate_peer", "pressure_dot_rate", "event_rate_prior_balls", 1),
        "boundary_rate": ("baseline_boundary_rate_raw", "baseline_boundary_rate_peer", "pressure_boundary_rate", "event_rate_prior_balls", -1),
        "extras_per_over": ("baseline_extras_per_over_raw", "baseline_extras_per_over_peer", "pressure_extras_per_over", "extras_prior_balls", -1),
    }
    for multiplier in config.shrinkage_sensitivity_multipliers:
        relative_components = []
        for metric, (raw_col, peer_col, observed_col, prior_attr, sign) in strength_fields.items():
            raw = pd.to_numeric(scores[raw_col], errors="coerce")
            peer = pd.to_numeric(scores[peer_col], errors="coerce")
            history_balls = pd.to_numeric(scores["baseline_history_balls"], errors="coerce").fillna(0)
            prior_balls = float(getattr(config, prior_attr)) * multiplier
            baseline = (raw * history_balls + peer * prior_balls) / (history_balls + prior_balls)
            observed = pd.to_numeric(scores[observed_col], errors="coerce")
            improvement = baseline - observed if sign < 0 else observed - baseline
            relative_components.append(improvement.groupby(scores["competition_group"], dropna=False).rank(method="average", pct=True))
        relative_score = pd.concat(relative_components, axis=1).mean(axis=1) * 100
        alternate = scores.copy()
        alternate["overall_bowling_pressure_score"] = (
            alternate["absolute_pressure_score_shrunk"] + relative_score
            + alternate["consistency_score"] + alternate["statistical_reliability_score"]
        ) / 4
        alternate["rank_eligible"] = alternate["overall_bowling_pressure_score"].where(
            alternate["eligible_analytical_candidate"]
        ).groupby(alternate["competition_group"], dropna=False).rank(method="min", ascending=False)
        rows.append(_compare_rankings(scores, alternate, f"baseline shrinkage prior ×{multiplier:g}", "Metric-specific prior exposure sensitivity"))

    for metric, column, direction, component_key, note in (
        ("economy", "economy_lower_bound_high", False, "economy", "Assume uncertain legal deliveries are legal; lower economy bound"),
        ("economy", "economy_upper_bound_high", False, "economy", "Conservative upper run/confirmed-ball economy bound"),
        ("bowler_extras_per_legal_over", "bowler_extra_runs_per_legal_over_upper_high", False, "bowler_extras_per_legal_over", "Charge uncertain extras to the bowler for upper-bound sensitivity"),
    ):
        if column not in scores:
            continue
        alternate = _sensitivity_absolute_with_metric(scores, column, direction, component_key)
        rows.append(_compare_rankings(scores, alternate, f"uncertain accounting: {column}", note))

    definitions: list[tuple[str, pd.Series, str]] = [
        ("nonlinear walk-forward P90", deliveries["bowling_pressure_band_walk_forward_nonlinear"].eq("high"),
         "Alternate Phase 4 bowling index; walk-forward monthly labels"),
        ("weighted context P90 (static)", deliveries["bowling_high_weighted_context_p90"].fillna(False).astype(bool),
         "Static Phase 4 full-dataset threshold; sensitivity only, not prospective"),
        ("nonlinear context P90 (static)", deliveries["bowling_high_nonlinear_context_p90"].fillna(False).astype(bool),
         "Static Phase 4 full-dataset threshold; sensitivity only, not prospective"),
    ]
    for threshold in (70, 80, 90):
        definitions.append((
            f"weighted fixed threshold {threshold}",
            pd.to_numeric(deliveries["bowler_pressure_score"], errors="coerce").ge(threshold),
            "Fixed absolute score threshold; sensitivity only",
        ))
    for name, flag, note in definitions:
        band = pd.Series("normal", index=deliveries.index, dtype="object")
        band.loc[flag.fillna(False)] = "high"
        band.loc[pd.to_numeric(deliveries["bowler_pressure_score"], errors="coerce").isna()] = "reference unavailable"
        alternate = _score_pressure_scenario(deliveries, band, config)
        rows.append(_compare_rankings(scores, alternate, name, note))

    # Leave-one-season-out checks use absolute pressure outcomes only. Baselines
    # are deliberately not recomputed from the omitted season for this check.
    for season in sorted(deliveries["season"].dropna().astype(str).unique()):
        kept = deliveries.loc[deliveries["season"].astype(str).ne(season)]
        absolute = _absolute_score_from_counts(kept)
        joined = scores[PLAYER_KEYS + ["absolute_pressure_score", "rank_eligible"]].merge(
            absolute, on=PLAYER_KEYS, how="inner", validate="one_to_one"
        )
        current = joined["absolute_sensitivity_score"].groupby(joined["competition_group"], dropna=False).rank(
            method="min", ascending=False
        )
        base = joined["absolute_pressure_score"].groupby(joined["competition_group"], dropna=False).rank(
            method="min", ascending=False
        )
        base_top = joined.assign(_r=base).sort_values("_r").groupby(
            "competition_group", dropna=False, group_keys=False
        ).head(10)
        scenario_top = joined.assign(_r=current).sort_values("_r").groupby(
            "competition_group", dropna=False, group_keys=False
        ).head(10)
        top_base = set(zip(base_top["competition_group"], base_top["bowler_id"]))
        top_scenario = set(zip(scenario_top["competition_group"], scenario_top["bowler_id"]))
        rows.append({
            "scenario": f"leave season {season} out",
            "players_compared": int(len(joined)),
            "top10_overlap_player_group_pairs": int(len(top_base & top_scenario)),
            "top10_base_pairs": int(len(top_base)),
            "top10_scenario_pairs": int(len(top_scenario)),
            "eligible_rank_spearman": float(base.corr(current, method="spearman")) if len(joined) > 1 else np.nan,
            "median_absolute_score_change": np.nan,
            "note": "Absolute-pressure-only ranking stability; historical baselines unchanged.",
        })
    return pd.DataFrame(rows), eligibility


def _write_figures(
    scores: pd.DataFrame,
    detailed: pd.DataFrame,
    uncertainty: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    files: list[Path] = []

    chart_specs = [
        ("top20_absolute_pressure.png", "absolute_pressure_score", "Top 20 absolute pressure performers", "Absolute pressure performance"),
        ("top20_relative_improvers.png", "relative_improvement_score", "Top 20 relative pressure improvers", "Improvement over walk-forward baseline"),
        ("top20_consistent.png", "consistency_score", "Top 20 most consistent pressure bowlers", "Match-level consistency"),
        ("top20_reliable.png", "statistical_reliability_score", "Top 20 most reliable pressure records", "Statistical reliability"),
        ("top20_composite.png", "overall_bowling_pressure_score", "Top 20 composite pressure bowlers", "Shrinkage-adjusted composite score"),
    ]
    for filename, column, title, xlabel in chart_specs:
        candidates = scores.loc[scores["eligible_analytical_candidate"]].sort_values(column, ascending=False).groupby(
            "competition_group", dropna=False, group_keys=False
        ).head(20)
        fig, ax = plt.subplots(figsize=(12, max(5, min(12, 0.28 * max(len(candidates), 1)))))
        if not candidates.empty:
            candidates["chart_label"] = candidates["competition_group"].astype(str) + " · " + candidates["player_name"].fillna(candidates["bowler_id"].astype(str)).astype(str)
            sns.barplot(data=candidates, y="chart_label", x=column, hue="competition_group", dodge=False, ax=ax, legend=False)
        else:
            ax.text(0.5, 0.5, "No bowlers met the configured eligibility thresholds", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Competition group · bowler")
        fig.tight_layout()
        path = figures_dir / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        files.append(path)

    scatter_specs = [
        ("pressure_vs_normal_economy.png", "economy_normal", "economy_high", "Economy per over", "High pressure vs normal pressure economy"),
        ("pressure_vs_normal_wickets.png", "wickets_per_legal_over_normal", "wickets_per_legal_over_high", "Wickets per legal over", "High pressure vs normal pressure wicket rate"),
        ("pressure_vs_normal_dot_rate.png", "bowler_dot_rate_normal", "bowler_dot_rate_high", "Legal-ball dot rate", "High pressure vs normal pressure dot rate"),
        ("pressure_vs_normal_boundaries.png", "boundary_concession_rate_normal", "boundary_concession_rate_high", "Boundary deliveries per legal ball", "Boundary concession: high vs normal pressure"),
    ]
    for filename, xcol, ycol, label, title in scatter_specs:
        fig, ax = plt.subplots(figsize=(8, 6))
        data = scores[[xcol, ycol, "competition_group"]].replace([np.inf, -np.inf], np.nan).dropna()
        if not data.empty:
            sns.scatterplot(data=data, x=xcol, y=ycol, hue="competition_group", alpha=0.7, ax=ax)
        ax.set_xlabel(f"Normal-pressure {label}")
        ax.set_ylabel(f"High-pressure {label}")
        ax.set_title(title)
        fig.tight_layout()
        path = figures_dir / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        files.append(path)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=scores, x="pressure_legal_balls", y="overall_bowling_pressure_score", hue="competition_group", alpha=0.75, ax=ax)
    ax.set(title="Composite pressure score and evidence volume", xlabel="Known legal balls under pressure", ylabel="Composite score (0–100)")
    fig.tight_layout()
    path = figures_dir / "score_vs_pressure_sample_size.png"
    fig.savefig(path, dpi=160); plt.close(fig); files.append(path)

    top_uncertainty = uncertainty.sort_values("bootstrap_score_median", ascending=False).groupby(
        "competition_group", dropna=False, group_keys=False
    ).head(20)
    fig, ax = plt.subplots(figsize=(10, 8))
    if not top_uncertainty.empty:
        top_uncertainty["label"] = top_uncertainty["competition_group"].astype(str) + " · " + top_uncertainty["bowler_id"].astype(str)
        ax.errorbar(
            top_uncertainty["bootstrap_score_median"], np.arange(len(top_uncertainty)),
            xerr=[top_uncertainty["bootstrap_score_median"] - top_uncertainty["bootstrap_score_p05"],
                  top_uncertainty["bootstrap_score_p95"] - top_uncertainty["bootstrap_score_median"]],
            fmt="o", capsize=2, color="#2463a6",
        )
        ax.set_yticks(np.arange(len(top_uncertainty)), top_uncertainty["label"])
    ax.set(title="Match-bootstrap score intervals (90% percentile interval)", xlabel="Bootstrap score", ylabel="Competition group · bowler")
    fig.tight_layout()
    path = figures_dir / "match_bootstrap_uncertainty.png"
    fig.savefig(path, dpi=160); plt.close(fig); files.append(path)

    fig, ax = plt.subplots(figsize=(8, 6))
    if not uncertainty.empty:
        sns.scatterplot(data=uncertainty, x="bootstrap_rank_std", y="top10_selection_frequency", hue="competition_group", alpha=0.7, ax=ax)
    ax.set(title="Ranking stability across match bootstrap draws", xlabel="Bootstrap rank standard deviation", ylabel="Top-10 selection frequency")
    fig.tight_layout()
    path = figures_dir / "ranking_stability.png"
    fig.savefig(path, dpi=160); plt.close(fig); files.append(path)

    comparison = scores.loc[scores["competition_group"].isin(["IPL", "T20I"])].copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    if not comparison.empty:
        sns.boxplot(data=comparison, x="competition_group", y="overall_bowling_pressure_score", ax=ax)
    ax.set(title="IPL and T20I rankings are produced separately", xlabel="Competition group", ylabel="Composite pressure score")
    fig.tight_layout()
    path = figures_dir / "ipl_vs_t20i_scores.png"
    fig.savefig(path, dpi=160); plt.close(fig); files.append(path)

    trend = detailed.loc[detailed["pressure_band"].eq("high")].groupby(
        ["competition_group", "season"], dropna=False
    ).agg(economy=("economy", "mean"), wickets_per_over=("wickets_per_legal_over", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    if not trend.empty:
        sns.lineplot(data=trend, x="season", y="economy", hue="competition_group", marker="o", ax=ax)
    ax.set(title="Season-level high-pressure economy", xlabel="Season", ylabel="Economy per over")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    path = figures_dir / "season_pressure_trends.png"
    fig.savefig(path, dpi=160); plt.close(fig); files.append(path)
    return files


def _write_reports(
    reports_dir: Path,
    details: pd.DataFrame,
    scores: pd.DataFrame,
    uncertainty: pd.DataFrame,
    sensitivity: pd.DataFrame,
    eligibility: pd.DataFrame,
    diagnostics: dict[str, Any],
    config: BowlingModelConfig,
    bootstrap_replicates: int,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    eligible = scores.loc[scores["eligible_analytical_candidate"]]
    top = eligible.sort_values(["competition_group", "rank_eligible"])
    keys = {
        "Absolute pressure performers": "rank_absolute",
        "Relative improvement": "rank_relative",
        "Consistency": "rank_consistency",
        "Statistical reliability": "rank_reliability",
        "Composite": "rank_eligible",
    }
    findings = []
    for label, rank in keys.items():
        findings.append(f"\n{label} — provisional analytical leaders by competition group")
        for group in sorted(scores["competition_group"].dropna().unique()):
            part = top.loc[top["competition_group"].eq(group)]
            if part.empty:
                findings.append(f"{group}: no bowlers met the configured eligibility thresholds.")
                continue
            subset = part.sort_values(rank).head(20)
            names = subset["player_name"].fillna(subset["bowler_id"].astype(str))
            findings.append(f"{group}: " + ", ".join(
                f"{i+1}. {name} (ID {int(pid)})" for i, (name, pid) in enumerate(zip(names, subset["bowler_id"]))
            ))
    (reports_dir / "bowling_key_findings.txt").write_text(
        "SUPERR SELECTOR — PHASE 6 PROVISIONAL BOWLING FINDINGS\n"
        "These are model outputs for analysis and audit, not final competition picks.\n"
        + "\n".join(findings) + "\n\n"
        + "Baseline eligibility comparison (IPL / T20I / Other T20):\n"
        + "\n".join(
            f"{row.scenario}: {int(row.eligible_ipl)} / {int(row.eligible_t20i)} / {int(row.eligible_other_t20)} eligible; "
            f"eligible top-10 overlap {int(row.top10_overlap_player_group_pairs)}/{int(row.top10_base_pairs)}; "
            f"pressure+coverage candidate overlap {int(row.pressure_coverage_top10_overlap)}/{int(row.pressure_coverage_top10_primary)}; "
            f"Spearman rank agreement {row.eligible_rank_spearman:.3f}; median absolute score change {row.median_absolute_score_change:.3f}."
            for row in sensitivity.loc[sensitivity["scenario"].str.startswith(("A.", "B.", "C.", "D."))].itertuples()
        ) + "\n\n"
        + "Competition groups are ranked separately. Only eligible records are shown above; all other bowlers and exclusion reasons remain in bowler_eligibility.csv.\n",
        encoding="utf-8",
    )

    methodology = [
        "SUPERR SELECTOR — PHASE 6 BOWLING METHODOLOGY",
        "=" * 78,
        "Pressure is the distinct Phase 4 bowling_pressure_weighted score: opponent scoring threat/resources while bowling, not the batter's pressure score.",
        "The Phase 4 match state is pre-delivery. The current delivery's runs, extras, and wicket are joined only as outcomes after pressure is assigned.",
        f"Primary high-pressure opportunity: prior-month P90 within competition group × innings phase after {config.pressure_reference_minimum} earlier pressure states. P25/P50/P75/P90 also identify low, normal, elevated, and high strata. Early months remain unavailable.",
        "All matches on the same calendar date are excluded from walk-forward baselines because within-day match chronology is not reliable in the source.",
        "Primary baseline is format-hierarchical. Peer pool levels are exact competition/season + format + phase + innings context; same-format + phase + context; same-format + phase; then same-format pool. Personal normal-pressure history is independently accumulated across seasons within the same format + phase + context, using only prior calendar dates. This prevents a season-specific peer pool from erasing valid cross-season personal history.",
        "Each delivery and bowler summary records baseline_pool_source, numeric baseline_selected_level, baseline_personal_history_scope, own/peer legal-ball exposures, coverage, shrinkage fraction, and metric-specific approximate standard errors. The SEs use an independent Poisson-count approximation and are not calibrated intervals; they omit overdispersion, between-competition heterogeneity, and uncertainty from pressure classification. Fallback level is a separate lower-is-better reliability component. The 300-match pressure bootstrap is conditional on the selected baseline.",
        "The prior selected-context history behavior is retained as comparison A. The primary hierarchy preserves competition-aware peers and same-format personal history (B). Comparison C relaxes baseline coverage from 25% to 10%; evidence reliability still includes history/coverage. Comparison D pools peer rates across all T20 groups only as a sensitivity, retaining phase/context and explicitly marking the broader source. It is not a primary source because IPL, T20I, and other T20 scoring environments may differ.",
        "The source does not identify bowling role consistently; innings phase and innings context are the available tactical comparability strata. No population-level cross-format pool is treated as equivalent to an exact competition baseline.",
        f"Metric-specific peer shrinkage equivalent exposure: economy {config.economy_prior_balls:g} legal balls; wicket rate {config.wicket_prior_balls:g}; dot/boundary rates {config.event_rate_prior_balls:g}; extras {config.extras_prior_balls:g}.",
        "Economy is bowler-charged runs per six confirmed legal balls. Wides and the observed Overstep no-ball label are charged; byes, leg-byes, and penalty extras are not charged to the bowler and remain in team totals. The one source penalty row has five runs and unknown legality. Other/untyped extras remain flagged as uncertain and enter a conservative upper-bound sensitivity.",
        "Bowler-attributable wickets are bowled, caught, LBW, stump/stumped, and hit wicket. Run-outs, retirements, and obstructing-the-field events are not credited.",
        "Primary dot is a confirmed legal ball with zero bowler-charged runs; team-total dot rate is also retained separately. Boundary rates count batter four/six flags per confirmed legal ball.",
        "Relative improvement directions: baseline economy minus pressure economy; pressure wickets/dots minus baseline; baseline boundary/extras rate minus pressure.",
        "Consistency is match-clustered: median/lower-tail match performance, dispersion, and share of pressure matches meeting the competition median standard.",
        "Reliability includes pressure matches/innings/balls, baseline coverage and same-format personal history, peer exposure, bootstrap score intervals, top-10 frequency, and rank stability. baseline_history_balls is the median personal normal-pressure legal-ball history available over a bowler's high-pressure delivery contexts; it is not total career volume.",
        "Composite A averages standardized component percentiles and dimensions equally. B adjusts metric weights inversely to empirical mean absolute Spearman correlation while retaining equal dimension weights. C applies metric-specific peer shrinkage. D averages the correlation-adjusted absolute, relative, and consistency domains, then applies one reliability discount, (0.5 + 0.5 × reliability/100); reliability is not included in the performance-only base.",
        f"Match bootstrap uses {bootstrap_replicates} replicates and seed {config.random_seed}; match IDs are resampled once per competition group and shared multiplicities are applied to every bowler in each selected match, including duplicate-match multiplicities. Each draw recomputes shrunk pressure and baseline-relative rate ranks in four equally weighted domains. Metric components within those domains are equally weighted, unlike the correlation-adjusted primary composite; walk-forward baselines, match consistency, and evidence-volume reliability are fixed. Thus the intervals and top-10 frequencies are conditional stability proxies, not fully propagated intervals for the final composite. A player absent from every sampled match in a draw contributes no score for that draw, so their percentile intervals condition on at least one appearance; bootstrap_score_draws records the count. Percentile intervals are empirical summaries, not calibrated confidence or prediction intervals.",
        "IPL, T20I, and Other T20 are reported separately. No single combined ranking is presented because format and competition environments differ.",
        "Weights are descriptive design choices and correlation corrections, not outcome-trained or optimized to favor any player. No causal pressure effect is claimed.",
    ]
    (reports_dir / "bowling_methodology.txt").write_text("\n".join(methodology) + "\n", encoding="utf-8")

    validation = [
        "SUPERR SELECTOR — PHASE 6 BOWLING VALIDATION",
        "=" * 78,
        f"Source rows: {diagnostics.get('source_delivery_rows', 0):,}; unique delivery keys: {diagnostics.get('unique_delivery_keys', 0):,}.",
        f"Bowler mapping: missing IDs {diagnostics.get('missing_bowler_id_rows', 0):,}; unmatched mapping rows {diagnostics.get('unmatched_bowler_mapping_rows', 0):,}.",
        f"Phase 4 bowling pressure matched rows: {diagnostics.get('bowling_pressure_rows_matched', 0):,}; unmatched rows {diagnostics.get('bowling_pressure_rows_unmatched', 0):,}.",
        f"Known legal deliveries: {diagnostics.get('legal_ball_known_rows', 0):,}; uncertain legal-ball status: {diagnostics.get('legal_ball_uncertain_rows', 0):,}.",
        f"Bowler-attributable wicket rows: {diagnostics.get('bowler_attributable_wicket_rows', 0):,}; uncertain wicket attribution rows {diagnostics.get('unknown_wicket_attribution_rows', 0):,}.",
        f"Uncertain bowler-extra attribution rows: {diagnostics.get('uncertain_bowler_extra_rows', 0):,}; missing bowling-pressure scores: {diagnostics.get('bowling_pressure_missing_rows', 0):,}.",
        f"Player/competition scoring rows: {len(scores):,}; eligible records: {int(scores['eligible_analytical_candidate'].sum()):,}; ineligible records: {int((~scores['eligible_analytical_candidate']).sum()):,}.",
        f"Composite score range: {scores['overall_bowling_pressure_score'].min():.3f}–{scores['overall_bowling_pressure_score'].max():.3f}; all composite scores bounded 0–100: {bool(scores['overall_bowling_pressure_score'].dropna().between(0, 100).all())}.",
        f"Bowler/competition records with baseline coverage: {int(deliveries_baseline_count(scores)):,}; bootstrap rows: {len(uncertainty):,}; match-level replicates: {bootstrap_replicates}; seed: {config.random_seed}.",
        f"Primary baseline strategy: {config.baseline_strategy}; minimum peer exposure: {config.minimum_peer_baseline_balls} legal balls; baseline coverage gate: {config.baseline_coverage_minimum:.0%}; personal history gate: {config.eligibility_baseline_balls} legal balls.",
        "Baseline hierarchy/eligibility comparison rows A–D are in bowling_sensitivity.txt and bowler_sensitivity_analysis.csv; numeric source level and uncertainty estimates are in bowler_baseline_statistics.csv and bowler_eligibility.csv.",
        "Walk-forward pressure thresholds use earlier months only; baselines use strictly earlier calendar dates and exclude all same-date matches.",
        "Raw data and delivery-level derived CSVs are written only to local ignored data/output paths and are not published by this pipeline.",
    ]
    for group, part in scores.groupby("competition_group", dropna=False):
        meets_pressure = (
            part["pressure_matches"].ge(config.eligibility_pressure_matches)
            & part["pressure_innings"].ge(config.eligibility_pressure_innings)
            & part["pressure_legal_balls"].ge(config.eligibility_pressure_balls)
            & part["baseline_coverage_fraction"].fillna(0).ge(config.baseline_coverage_minimum)
        )
        meets_personal_history = part["baseline_history_balls"].ge(config.eligibility_baseline_balls)
        validation.append(
            f"{group}: {int((part['pressure_matches'].ge(config.eligibility_pressure_matches) & part['pressure_innings'].ge(config.eligibility_pressure_innings) & part['pressure_legal_balls'].ge(config.eligibility_pressure_balls)).sum())} meet pressure exposure; {int(meets_pressure.sum())} also meet baseline coverage; "
            f"{int((meets_pressure & meets_personal_history).sum())} meet the configured personal baseline-history threshold."
        )
    (reports_dir / "bowling_validation.txt").write_text("\n".join(validation) + "\n", encoding="utf-8")

    comparison_lines = [
        "SUPERR SELECTOR — PHASE 6 MODEL COMPARISON",
        "A: equal-weight standardized component/domain scores.",
        "B: inverse-correlation component weighting; equal weight across absolute, relative, consistency, reliability domains.",
        "C: empirical peer-shrinkage-adjusted pressure outcomes; displayed provisional composite.",
        "D: the mean correlation-adjusted absolute, relative, and consistency domains is multiplied by (0.5 + 0.5 × reliability/100); reliability is excluded from its base and applied once.",
        "No model is declared predictive without an independent pressure-outcome target.",
        "\nTop-10 overlap and Spearman rank agreement by composite variant:",
        sensitivity.loc[sensitivity["scenario"].str.contains("composite", case=False, na=False)].to_string(index=False),
        "\nCorrelation-adjusted scoring reduces redundant metric influence; see bowling_correlation_weights.csv for component weights.",
    ]
    (reports_dir / "bowling_model_comparison.txt").write_text("\n".join(comparison_lines) + "\n", encoding="utf-8")
    sensitivity_table = "\n".join(
        line.rstrip() for line in sensitivity.to_string(index=False, float_format=lambda value: f"{value:.3f}").splitlines()
    )
    approach_columns = [
        "scenario", "eligible_ipl", "eligible_t20i", "eligible_other_t20",
        "pressure_gate_ipl", "coverage_gate_ipl", "history_gate_ipl",
        "pressure_gate_t20i", "coverage_gate_t20i", "history_gate_t20i",
        "pressure_gate_other_t20", "coverage_gate_other_t20", "history_gate_other_t20",
        "median_history_balls_ipl", "median_peer_balls_ipl", "median_history_balls_t20i",
        "median_peer_balls_t20i", "median_history_balls_other_t20", "median_peer_balls_other_t20",
        "baseline_levels_ipl", "baseline_levels_t20i", "baseline_levels_other_t20",
        "baseline_sources_ipl", "baseline_sources_t20i", "baseline_sources_other_t20",
        "top10_overlap_player_group_pairs", "top10_base_pairs", "eligible_rank_spearman",
        "pressure_coverage_top10_overlap", "pressure_coverage_top10_primary",
        "pressure_coverage_rank_spearman",
        "median_absolute_score_change", "note",
    ]
    approach_table = sensitivity.loc[
        sensitivity["scenario"].str.startswith(("A.", "B.", "C.", "D.")),
        [column for column in approach_columns if column in sensitivity.columns],
    ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    eligibility_table = "\n".join(line.rstrip() for line in eligibility.to_string(index=False).splitlines())
    (reports_dir / "bowling_sensitivity.txt").write_text(
        "SUPERR SELECTOR — PHASE 6 SENSITIVITY\n\n"
        "Baseline approach comparison (A previous selected-context history, B primary format hierarchy, C relaxed coverage, D broad T20 sensitivity):\n"
        + approach_table
        + "\n\nFull historical and current sensitivity scenarios:\n"
        + sensitivity_table
        + "\n\nEligibility threshold counts by competition group:\n"
        + eligibility_table
        + "\n\nLower/upper economy and uncertain extras remain available in processed count tables; static Phase 4 flags are sensitivity-only because they use full-dataset thresholds.\n",
        encoding="utf-8",
    )


def deliveries_baseline_count(scores: pd.DataFrame) -> int:
    return int(pd.to_numeric(scores.get("baseline_coverage_fraction"), errors="coerce").fillna(0).gt(0).sum())


def run_bowling_model(
    tables: dict[str, pd.DataFrame],
    pressure_states: pd.DataFrame,
    pressure_features: pd.DataFrame,
    pressure_scores: pd.DataFrame,
    report_dir: str | Path,
    figure_dir: str | Path,
    processed_dir: str | Path,
    config: BowlingModelConfig | None = None,
    bootstrap_replicates: int | None = None,
) -> dict[str, Any]:
    """Run Phase 6 from existing Phase 4 outputs and write audited artifacts."""
    cfg = config or BowlingModelConfig()
    replicates = int(bootstrap_replicates or cfg.bootstrap_replicates)
    reports, figures, processed = Path(report_dir), Path(figure_dir), Path(processed_dir)
    for directory in (reports, figures, processed):
        directory.mkdir(parents=True, exist_ok=True)

    deliveries, diagnostics = build_bowling_delivery_features(
        tables, pressure_states, pressure_features, pressure_scores
    )
    deliveries = assign_walk_forward_pressure_bands(deliveries, cfg)
    deliveries = assign_walk_forward_pressure_bands(
        deliveries, cfg, cfg.alternate_pressure_column, "_nonlinear", copy_frame=False
    )
    deliveries, baseline_audit = add_walk_forward_bowling_baselines(deliveries, cfg)
    detail, player_stats, match_stats = build_pressure_statistics(deliveries)
    differential = build_bowler_pressure_differentials(deliveries, player_stats, match_stats, cfg)
    scores, correlation_weights = score_bowlers(differential, cfg)
    names = deliveries.loc[deliveries["bowler_id"].notna(), ["competition_group", "bowler_id", "bowler_name"]].drop_duplicates(
        ["competition_group", "bowler_id"]
    )
    scores = scores.merge(names, on=PLAYER_KEYS, how="left", validate="one_to_one")
    scores["player_name"] = scores["bowler_name"].fillna(scores["bowler_id"].astype(str))

    bootstrap_long, uncertainty = bootstrap_bowler_scores(
        match_stats, scores, replicates=replicates, seed=cfg.random_seed, config=cfg
    )
    scores = integrate_bootstrap_reliability(scores, uncertainty)
    sensitivity, eligibility = _sensitivity_analysis(deliveries, scores, cfg)

    rankings = scores.loc[scores["eligible_analytical_candidate"]].sort_values(
        ["competition_group", "rank_eligible", "bowler_id"]
    ).copy()
    rankings.to_csv(processed / "bowler_rankings.csv", index=False)
    deliveries.to_csv(processed / "bowler_pressure_features.csv", index=False)
    detail.to_csv(processed / "bowler_pressure_statistics.csv", index=False)
    baseline_audit.to_csv(processed / "bowler_baseline_statistics.csv", index=False)
    differential.to_csv(processed / "bowler_pressure_differentials.csv", index=False)
    scores.to_csv(processed / "bowler_eligibility.csv", index=False)
    uncertainty.to_csv(processed / "bowler_uncertainty.csv", index=False)
    bootstrap_long.to_csv(processed / "bowler_bootstrap_score_distribution.csv", index=False)
    scores.to_csv(processed / "bowler_composite_scores.csv", index=False)
    sensitivity.to_csv(processed / "bowler_sensitivity_analysis.csv", index=False)
    eligibility.to_csv(processed / "bowler_eligibility_sensitivity.csv", index=False)
    correlation_weights.to_csv(reports / "bowling_correlation_weights.csv", index=False)

    chart_paths = _write_figures(scores, detail, uncertainty, figures)
    _write_reports(
        reports, detail, scores, uncertainty, sensitivity, eligibility,
        diagnostics, cfg, replicates,
    )
    LOGGER.info(
        "Bowling model complete: %s delivery rows, %s bowler/competition scores, %s figures",
        len(deliveries), len(scores), len(chart_paths),
    )
    return {
        "deliveries": deliveries,
        "pressure_statistics": detail,
        "baselines": baseline_audit,
        "differentials": differential,
        "scores": scores,
        "rankings": rankings,
        "uncertainty": uncertainty,
        "sensitivity": sensitivity,
        "eligibility_sensitivity": eligibility,
        "diagnostics": diagnostics,
        "figures": chart_paths,
    }
