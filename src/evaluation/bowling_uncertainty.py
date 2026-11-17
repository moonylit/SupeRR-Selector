"""Reproducible competition-stratified match-cluster bootstrap for bowlers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import BowlingModelConfig


def bootstrap_bowler_scores(
    match_statistics: pd.DataFrame,
    scores: pd.DataFrame,
    replicates: int | None = None,
    seed: int | None = None,
    config: BowlingModelConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resample matches jointly across bowlers and summarize score uncertainty.

    Every draw samples competition-level match IDs once; all bowlers appearing
    in a selected match receive the same multiplicity, including duplicate
    selections. Each score is a conditional equal-metric, four-domain proxy:
    historical baselines, match consistency, and evidence-volume reliability
    are fixed, and metric weights differ from the correlation-adjusted primary
    composite. Players missing from all sampled matches have no draw; reported
    intervals condition on at least one appearance. Percentile intervals are
    empirical summaries, not calibrated or fully propagated intervals.
    """
    cfg = config or BowlingModelConfig()
    n_replicates = int(replicates or cfg.bootstrap_replicates)
    random_seed = int(cfg.random_seed if seed is None else seed)
    rng = np.random.default_rng(random_seed)
    high = match_statistics.loc[match_statistics["pressure_band"].eq("high")].copy()
    long_rows: list[pd.DataFrame] = []

    for competition, match_group in high.groupby("competition_group", dropna=False, sort=False):
        match_ids = match_group["match_id"].drop_duplicates().to_numpy()
        if not len(match_ids):
            continue
        current_scores = scores.loc[scores["competition_group"].eq(competition)].copy()
        if current_scores.empty:
            continue
        for replicate in range(n_replicates):
            draws = rng.choice(match_ids, size=len(match_ids), replace=True)
            multiplicity = pd.Series(draws).value_counts().rename("_match_weight").rename_axis("match_id").reset_index()
            sampled = match_group.merge(multiplicity, on="match_id", how="inner", validate="many_to_one")
            sampled["_weight"] = sampled["_match_weight"].astype(float)
            count_columns = [
                "legal_balls_known", "runs_conceded_lower", "bowler_wickets",
                "legal_bowler_dots", "boundary_deliveries", "bowler_extra_runs_known",
            ]
            weighted_columns = []
            for column in count_columns:
                weighted = f"_weighted_{column}"
                sampled[weighted] = sampled[column] * sampled["_weight"]
                weighted_columns.append(weighted)
            grouped = sampled.groupby("bowler_id", as_index=False, dropna=False)[weighted_columns].sum()
            grouped = grouped.rename(columns={
                f"_weighted_{column}": column for column in count_columns
            })
            grouped["competition_group"] = competition
            grouped = grouped.merge(
                current_scores[[
                    "bowler_id", "eligible_analytical_candidate", "baseline_economy",
                    "baseline_wickets_per_over", "baseline_dot_rate", "baseline_boundary_rate",
                    "baseline_extras_per_over", "consistency_score", "statistical_reliability_score",
                ]], on="bowler_id", how="left", validate="one_to_one",
            )
            denominator = grouped["legal_balls_known"].replace(0, np.nan)
            grouped["economy"] = 6 * grouped["runs_conceded_lower"] / denominator
            grouped["wickets_per_over"] = 6 * grouped["bowler_wickets"] / denominator
            grouped["dot_rate"] = grouped["legal_bowler_dots"] / denominator
            grouped["boundary_rate"] = grouped["boundary_deliveries"] / denominator
            grouped["extras_per_over"] = 6 * grouped["bowler_extra_runs_known"] / denominator

            # Reproduce the four equal-weight domains of the primary composite
            # for each resample. Baselines, match consistency, and the original
            # evidence volume are held fixed, so intervals are conditional on
            # those estimates rather than a full-pipeline bootstrap.
            metric_specs = (
                ("economy", "runs_conceded_lower", False, 6.0, cfg.economy_prior_balls,
                 "baseline_economy"),
                ("wickets_per_over", "bowler_wickets", True, 6.0, cfg.wicket_prior_balls,
                 "baseline_wickets_per_over"),
                ("dot_rate", "legal_bowler_dots", True, 1.0, cfg.event_rate_prior_balls,
                 "baseline_dot_rate"),
                ("boundary_rate", "boundary_deliveries", False, 1.0, cfg.event_rate_prior_balls,
                 "baseline_boundary_rate"),
                ("extras_per_over", "bowler_extra_runs_known", False, 6.0, cfg.extras_prior_balls,
                 "baseline_extras_per_over"),
            )
            absolute_components = []
            relative_components = []
            for metric, numerator, higher, scale, prior_balls, baseline_column in metric_specs:
                group_num = grouped[numerator].sum()
                group_den = grouped["legal_balls_known"].sum()
                peer_num = (group_num - grouped[numerator]).clip(lower=0)
                peer_den = (group_den - grouped["legal_balls_known"]).clip(lower=0)
                peer_rate = peer_num.div(peer_den.replace(0, np.nan))
                shrunk = (
                    grouped[numerator] + peer_rate * prior_balls
                ).div(grouped["legal_balls_known"] + prior_balls) * scale
                raw = grouped[metric]
                shrunk = shrunk.where(peer_rate.notna(), raw)
                absolute_component = shrunk.rank(method="average", pct=True)
                absolute_components.append(absolute_component if higher else 1 - absolute_component)

                baseline = grouped[baseline_column]
                favorable = baseline - raw if not higher else raw - baseline
                relative_components.append(favorable.rank(method="average", pct=True))

            grouped["absolute_score"] = pd.concat(absolute_components, axis=1).mean(axis=1) * 100
            grouped["relative_score"] = pd.concat(relative_components, axis=1).mean(axis=1) * 100
            grouped["bootstrap_score"] = (
                grouped[[
                    "absolute_score", "relative_score", "consistency_score",
                    "statistical_reliability_score",
                ]].mean(axis=1)
            ).clip(0, 100)
            grouped["bootstrap_rank"] = grouped["bootstrap_score"].where(
                grouped["eligible_analytical_candidate"]
            ).rank(method="min", ascending=False)
            eligible_sample = grouped.loc[
                grouped["eligible_analytical_candidate"] & grouped["bootstrap_rank"].notna()
            ].nsmallest(10, "bootstrap_rank")
            top_ids = set(eligible_sample["bowler_id"])
            grouped["bootstrap_top10_selected"] = grouped["bowler_id"].isin(top_ids)
            grouped["replicate"] = replicate + 1
            long_rows.append(grouped[[
                "competition_group", "bowler_id", "replicate", "bootstrap_score",
                "bootstrap_rank", "bootstrap_top10_selected", "eligible_analytical_candidate",
                "absolute_score", "relative_score", "consistency_score",
                "statistical_reliability_score",
            ]])

    long = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame(
        columns=["competition_group", "bowler_id", "replicate", "bootstrap_score", "bootstrap_rank", "bootstrap_top10_selected", "eligible_analytical_candidate", "absolute_score", "relative_score", "consistency_score", "statistical_reliability_score"]
    )
    if long.empty:
        return long, pd.DataFrame(columns=["competition_group", "bowler_id"])
    summary = long.groupby(["competition_group", "bowler_id"], as_index=False, dropna=False).agg(
        bootstrap_score_draws=("bootstrap_score", "count"),
        bootstrap_score_mean=("bootstrap_score", "mean"),
        bootstrap_score_median=("bootstrap_score", "median"),
        bootstrap_score_std=("bootstrap_score", "std"),
        bootstrap_score_p05=("bootstrap_score", lambda x: float(x.dropna().quantile(0.05)) if x.notna().any() else np.nan),
        bootstrap_score_p95=("bootstrap_score", lambda x: float(x.dropna().quantile(0.95)) if x.notna().any() else np.nan),
        bootstrap_score_p025=("bootstrap_score", lambda x: float(x.dropna().quantile(0.025)) if x.notna().any() else np.nan),
        bootstrap_score_p975=("bootstrap_score", lambda x: float(x.dropna().quantile(0.975)) if x.notna().any() else np.nan),
        bootstrap_rank_mean=("bootstrap_rank", "mean"),
        bootstrap_rank_std=("bootstrap_rank", "std"),
        bootstrap_rank_p05=("bootstrap_rank", lambda x: float(x.dropna().quantile(0.05)) if x.notna().any() else np.nan),
        bootstrap_rank_p95=("bootstrap_rank", lambda x: float(x.dropna().quantile(0.95)) if x.notna().any() else np.nan),
        top10_selection_frequency=("bootstrap_top10_selected", "mean"),
    )
    summary["bootstrap_replicates_requested"] = n_replicates
    summary["top10_selection_frequency"] = (
        summary["top10_selection_frequency"]
        * summary["bootstrap_score_draws"] / n_replicates
    )
    summary["bootstrap_unit"] = "match"
    summary["bootstrap_scope"] = "competition-group matches sampled jointly across all bowlers"
    summary["interval_interpretation"] = "empirical percentile bootstrap interval; not a calibrated confidence or prediction interval"
    return long, summary
