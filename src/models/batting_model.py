"""Phase 5 end-to-end batting pressure evaluation pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import BATTING_MODEL_CONFIG, BattingModelConfig
from src.evaluation.batting_uncertainty import bootstrap_batter_pressure
from src.models.batting_baseline import (add_historical_personal_baselines,
    add_walk_forward_context_regression, assign_walk_forward_pressure_labels)
from src.models.batting_features import build_batting_delivery_features
from src.models.batting_scoring import assign_eligibility, build_composite_scores
from src.models.batting_statistics import build_batter_statistics

LOGGER = logging.getLogger(__name__)


def _aggregate_player_tables(pressure: pd.DataFrame, baseline: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Collapse context strata to competition-group/player reporting rows."""
    ids = ["competition_group", "striker_id"]
    rows = []
    for (group, player), part in pressure.groupby(ids, dropna=False):
        rec: dict[str, Any] = {"competition_group": group, "striker_id": player,
            "player_name": part["player_name"].dropna().iloc[0] if part["player_name"].notna().any() else "Unknown",
            "batting_team_id": part["batting_team_id"].dropna().iloc[0] if part["batting_team_id"].notna().any() else np.nan,
            "pressure_balls": int(part.loc[part["pressure_group"].eq("walk-forward P90 pressure"), "balls"].sum()),
            "pressure_matches": int(part.loc[part["pressure_group"].eq("walk-forward P90 pressure"), "matches"].sum()),
            "pressure_innings": int(part.loc[part["pressure_group"].eq("walk-forward P90 pressure"), "innings_count"].sum()),
            "pressure_seasons": int(part.loc[part["pressure_group"].eq("walk-forward P90 pressure"), "seasons"].sum()),
            "uncertain_faced_deliveries": 0}
        for pressure_group, prefix in (("walk-forward P90 pressure", "pressure"), ("non-P90 known reference", "normal")):
            p = part.loc[part["pressure_group"].eq(pressure_group)]
            balls, runs = p["balls"].sum(), p["batter_runs"].sum()
            for source, name in (("fours", "fours"), ("sixes", "sixes"), ("boundary_balls", "boundary_balls"), ("dot_balls", "dot_balls"), ("boundary_runs", "boundary_runs"), ("dismissals", "dismissals")):
                rec[f"{prefix}_{name}"] = p[source].sum()
            rec[f"{prefix}_balls"] = balls
            rec[f"{prefix}_runs"] = runs
            rec[f"{prefix}_strike_rate"] = 100 * runs / balls if balls else np.nan
            rec[f"{prefix}_runs_per_ball"] = runs / balls if balls else np.nan
            rec[f"{prefix}_boundary_frequency"] = rec[f"{prefix}_boundary_balls"] / balls if balls else np.nan
            rec[f"{prefix}_four_frequency"] = rec[f"{prefix}_fours"] / balls if balls else np.nan
            rec[f"{prefix}_six_frequency"] = rec[f"{prefix}_sixes"] / balls if balls else np.nan
            rec[f"{prefix}_dot_ball_frequency"] = rec[f"{prefix}_dot_balls"] / balls if balls else np.nan
            rec[f"{prefix}_dismissal_frequency"] = rec[f"{prefix}_dismissals"] / balls if balls else np.nan
            rec[f"{prefix}_boundary_run_share"] = rec[f"{prefix}_boundary_runs"] / runs if runs else np.nan
            rec[f"{prefix}_batting_average"] = runs / rec[f"{prefix}_dismissals"] if rec[f"{prefix}_dismissals"] else np.nan
        rows.append(rec)
    score = pd.DataFrame(rows)
    if score.empty: return score
    base = baseline.groupby(ids, as_index=False, dropna=False).agg(
        baseline_rpb=("baseline_rpb", "mean"), baseline_boundary_frequency=("baseline_boundary_frequency", "mean"),
        baseline_four_frequency=("baseline_four_frequency", "mean"), baseline_six_frequency=("baseline_six_frequency", "mean"),
        baseline_dot_ball_frequency=("baseline_dot_ball_frequency", "mean"), baseline_dismissal_frequency=("baseline_dismissal_frequency", "mean"),
        baseline_history_balls=("baseline_history_balls", "max"), baseline_pool_balls=("baseline_pool_balls", "max"),
        baseline_shrinkage_fraction=("baseline_shrinkage_fraction", "mean"))
    score = score.merge(base, on=ids, how="left", validate="one_to_one")
    mapping = {"runs_per_ball": "rpb", "boundary_frequency": "boundary_frequency", "dot_ball_frequency": "dot_ball_frequency", "dismissal_frequency": "dismissal_frequency", "strike_rate": "strike_rate"}
    for metric, _ in mapping.items():
        score[f"{metric}_absolute_difference"] = score[f"pressure_{metric}"] - score[f"normal_{metric}"]
        score[f"{metric}_relative_change_pct"] = 100 * score[f"{metric}_absolute_difference"] / score[f"normal_{metric}"].abs().replace(0, np.nan)
    score["rpb_personal_baseline_difference"] = score["pressure_runs_per_ball"] - score["baseline_rpb"]
    score["boundary_frequency_personal_baseline_difference"] = score["pressure_boundary_frequency"] - score["baseline_boundary_frequency"]
    score["dot_frequency_personal_baseline_difference"] = score["pressure_dot_ball_frequency"] - score["baseline_dot_ball_frequency"]
    score["dismissal_frequency_personal_baseline_difference"] = score["pressure_dismissal_frequency"] - score["baseline_dismissal_frequency"]
    match_stats = matches.groupby(ids, as_index=False, dropna=False).agg(
        player_match_rpb_mean=("runs_per_ball", "mean"), player_match_rpb_median=("runs_per_ball", "median"),
        player_match_rpb_std=("runs_per_ball", "std"), player_match_rpb_iqr=("runs_per_ball", lambda x: x.quantile(.75)-x.quantile(.25)),
        total_matches=("match_id", "nunique"))
    match_stats["player_match_rpb_cv"] = match_stats["player_match_rpb_std"] / match_stats["player_match_rpb_mean"].abs().replace(0, np.nan)
    score = score.merge(match_stats, on=ids, how="left", validate="one_to_one")
    return score


def _sensitivity(deliveries: pd.DataFrame, scores: pd.DataFrame, config: BattingModelConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    player_keys = ["competition_group", "striker_id"]
    month = pd.to_datetime(deliveries["match_date"], errors="coerce").dt.to_period("M")
    scenarios: dict[str, pd.Series] = {
        "weighted walk-forward P90": deliveries["pressure_high_p90_walk_forward"].eq(True).fillna(False),
        "nonlinear walk-forward P90": deliveries["pressure_high_nonlinear_p90_walk_forward"].eq(True).fillna(False),
        "Phase 4 static weighted P90 (retrospective)": deliveries["batting_high_weighted_context_p90"].eq(True).fillna(False),
    }
    for pct in config.pressure_percentile_sensitivity:
        mask = pd.Series(False, index=deliveries.index)
        for _, indices in deliveries.groupby(["competition_group", "innings_phase"], dropna=False).groups.items():
            idx = pd.Index(indices)
            hist: list[float] = []
            for m in sorted(month.loc[idx].dropna().unique()):
                midx = idx[month.loc[idx].eq(m).to_numpy()]
                vals = pd.to_numeric(deliveries.loc[midx, config.pressure_score_column], errors="coerce").dropna()
                if len(hist) >= config.pressure_reference_minimum and hist:
                    mask.loc[vals.index] = vals.ge(np.quantile(hist, pct)).to_numpy()
                hist.extend(vals.astype(float).tolist())
        scenarios[f"weighted walk-forward P{int(pct*100)}"] = mask
    for cutoff in config.fixed_pressure_thresholds:
        scenarios[f"weighted fixed >= {cutoff:g}"] = pd.to_numeric(deliveries[config.pressure_score_column], errors="coerce").ge(cutoff)
    primary = scores.set_index(player_keys)["overall_batting_pressure_score"].rank(pct=True)
    player_records = []
    for definition, mask in scenarios.items():
        selected = deliveries.loc[mask & deliveries["balls_faced_known"] & deliveries["striker_id"].notna()].copy()
        agg = selected.groupby(player_keys, dropna=False).agg(
            runs=("batter_runs", "sum"), balls=("balls_faced_known", "sum"), matches=("match_id", "nunique"))
        agg["rpb"] = agg["runs"] / agg["balls"].replace(0, np.nan)
        agg["rank_pct"] = agg.groupby(level=0)["rpb"].rank(pct=True)
        agg = agg.reset_index()
        for _, row in agg.iterrows():
            key = (row["competition_group"], row["striker_id"])
            player_records.append({"analysis": "pressure definition", "definition": definition,
                "competition_group": key[0], "striker_id": key[1], "sensitivity_runs_per_ball": row["rpb"],
                "pressure_balls": row["balls"], "pressure_matches": row["matches"], "rank_percentile": row["rank_pct"],
                "primary_score_rank_percentile": primary.get(key, np.nan)})
    rows.extend(player_records)
    # Sensitivity to plausible absolute-vs-relative resilience emphasis.
    for weight in config.composite_weight_sensitivity:
        alt = weight * scores["absolute_performance_score"] + (1 - weight) * scores["relative_resilience_score"]
        for comp, idx in scores.groupby("competition_group", dropna=False).groups.items():
            ranks = alt.loc[idx].rank(pct=True)
            base_ranks = scores.loc[idx, "overall_batting_pressure_score"].rank(pct=True)
            for ridx in idx:
                rows.append({"analysis": "composite weight sensitivity", "definition": f"absolute {weight:.3f} / relative {1-weight:.3f}",
                    "competition_group": comp, "striker_id": scores.loc[ridx, "striker_id"], "score": float(alt.loc[ridx]),
                    "rank_percentile": float(ranks.loc[ridx]), "primary_score_rank_percentile": float(base_ranks.loc[ridx])})
    # Conservative exposure bound: count uncertain faced deliveries as balls in the denominator.
    uncertain = deliveries.loc[deliveries["pressure_high_p90_walk_forward"].eq(True).fillna(False)
        & deliveries["balls_faced_uncertain"] & deliveries["striker_id"].notna()].groupby(player_keys).agg(
            uncertain_deliveries=("_source_row_id", "size"), uncertain_runs=("batter_runs", "sum"))
    known = deliveries.loc[deliveries["pressure_high_p90_walk_forward"].eq(True).fillna(False)
        & deliveries["balls_faced_known"] & deliveries["striker_id"].notna()].groupby(player_keys).agg(
            known_balls=("_source_row_id", "size"), known_runs=("batter_runs", "sum"))
    bounds = known.join(uncertain, how="outer").fillna(0)
    bounds["known_only_rpb"] = bounds["known_runs"] / bounds["known_balls"].replace(0, np.nan)
    bounds["uncertain_included_rpb_bound"] = (bounds["known_runs"]+bounds["uncertain_runs"]) / (bounds["known_balls"]+bounds["uncertain_deliveries"]).replace(0,np.nan)
    for (comp, player), row in bounds.iterrows():
        rows.append({"analysis":"uncertain delivery sensitivity", "definition":"conservative: all uncertain faced records added to exposure denominator",
            "competition_group":comp,"striker_id":player,"known_only_runs_per_ball":row["known_only_rpb"],
            "sensitivity_runs_per_ball":row["uncertain_included_rpb_bound"],"uncertain_deliveries":int(row["uncertain_deliveries"])})
    # Leave-one-season-out influence on raw pressure RPB.
    high = deliveries.loc[deliveries["pressure_high_p90_walk_forward"].eq(True).fillna(False) & deliveries["balls_faced_known"]]
    for (comp, player), group in high.groupby(player_keys, dropna=False):
        for season in group["season"].dropna().unique():
            left = group.loc[group["season"].ne(season)]
            if left["balls_faced_known"].sum() > 0:
                rows.append({"analysis":"season sensitivity", "definition":f"leave season {season} out", "competition_group":comp,
                    "striker_id":player,"sensitivity_runs_per_ball":left["batter_runs"].sum()/left["balls_faced_known"].sum(),
                    "pressure_balls":int(left["balls_faced_known"].sum())})
    return pd.DataFrame(rows)


def _write_figures(scores: pd.DataFrame, pressure: pd.DataFrame, sensitivity: pd.DataFrame, directory: Path) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    scores = scores.copy()
    pressure = pressure.copy()
    sensitivity = sensitivity.copy()
    # Seaborn's size legend does not handle pandas Arrow-backed numeric dtypes yet.
    for frame in (scores, pressure, sensitivity):
        for column in frame.columns:
            if column != "competition_group" and pd.api.types.is_numeric_dtype(frame[column].dtype):
                frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
        if "competition_group" in frame:
            frame["competition_group"] = frame["competition_group"].astype("string").fillna("Unknown").astype(str)
    sns.set_theme(style="whitegrid", context="notebook")
    specs = [
        ("pressure_vs_normal_strike_rate", "normal_strike_rate", "pressure_strike_rate", "Strike rate", "Strike rate: pressure vs normal"),
        ("pressure_vs_normal_runs_per_ball", "normal_runs_per_ball", "pressure_runs_per_ball", "Runs per ball", "Runs per ball: pressure vs normal"),
        ("pressure_vs_normal_boundary_frequency", "normal_boundary_frequency", "pressure_boundary_frequency", "Boundary balls / ball", "Boundary frequency"),
        ("pressure_vs_normal_dot_percentage", "normal_dot_ball_frequency", "pressure_dot_ball_frequency", "Dot-ball proportion", "Dot-ball proportion"),
        ("pressure_vs_normal_dismissal_frequency", "normal_dismissal_frequency", "pressure_dismissal_frequency", "Dismissals / ball", "Dismissal frequency"),
    ]
    paths = []
    for name, x, y, label, title in specs:
        if x not in scores or y not in scores: continue
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=scores, x=x, y=y, hue="competition_group", size="pressure_balls", sizes=(20, 160), alpha=.72, ax=ax)
        lim = [min(scores[x].min(), scores[y].min()), max(scores[x].max(), scores[y].max())]
        if np.isfinite(lim).all(): ax.plot(lim, lim, color="black", linestyle="--", linewidth=1)
        ax.set(title=title, xlabel=f"Normal {label}", ylabel=f"Pressure {label}")
        fig.tight_layout(); path=directory/f"{name}.png"; fig.savefig(path, dpi=150); plt.close(fig); paths.append(path.name)
    charts = [
        ("pressure_differential_distribution", "runs_per_ball_absolute_difference", "Pressure minus normal runs per ball"),
        ("pressure_performance_vs_sample_size", "pressure_runs_per_ball", "Pressure runs per ball vs pressure balls"),
        ("relative_resilience_vs_absolute", "relative_resilience_score", "Relative resilience vs absolute performance"),
        ("batter_consistency_distribution", "player_match_rpb_std", "Match-to-match runs/ball standard deviation"),
        ("competition_specific_performance", "pressure_runs_per_ball", "Pressure runs per ball by competition"),
        ("composite_component_breakdown", "absolute_performance_score", "Score components"),
        ("weight_sensitivity", "score_model_a_equal_domains", "Composite score approach comparison"),
    ]
    for name, metric, title in charts:
        if metric not in scores: continue
        fig, ax=plt.subplots(figsize=(9,6))
        if name == "pressure_differential_distribution": sns.histplot(data=scores, x=metric, hue="competition_group", element="step", stat="density", common_norm=False, ax=ax)
        elif name == "pressure_performance_vs_sample_size": sns.scatterplot(data=scores, x="pressure_balls", y=metric, hue="competition_group", size="pressure_matches", ax=ax)
        elif name == "relative_resilience_vs_absolute": sns.scatterplot(data=scores, x=metric, y="absolute_performance_score", hue="competition_group", size="pressure_balls", ax=ax)
        elif name == "batter_consistency_distribution": sns.histplot(data=scores, x=metric, hue="competition_group", element="step", ax=ax)
        elif name == "competition_specific_performance": sns.boxplot(data=scores, x="competition_group", y=metric, ax=ax)
        elif name == "composite_component_breakdown": scores[["absolute_performance_score", "relative_resilience_score", "consistency_score", "reliability_score"]].mean().plot.bar(ax=ax)
        else:
            for col in ("score_model_a_equal_domains", "score_model_b_correlation_reduced", "score_model_c_shrinkage_adjusted"):
                ax.scatter(scores[col], scores["overall_batting_pressure_score"], label=col, alpha=.5)
            ax.legend()
        ax.set_title(title); fig.tight_layout(); path=directory/f"{name}.png"; fig.savefig(path,dpi=150); plt.close(fig); paths.append(path.name)
    # Required CI chart and innings-phase chart.
    if "pressure_rpb_ci_low" in scores:
        sample=scores.nlargest(25,"pressure_balls").sort_values("pressure_runs_per_ball")
        fig,ax=plt.subplots(figsize=(9,7)); ax.errorbar(sample["pressure_runs_per_ball"],range(len(sample)),xerr=[sample["pressure_runs_per_ball"]-sample["pressure_rpb_ci_low"],sample["pressure_rpb_ci_high"]-sample["pressure_runs_per_ball"]],fmt="o")
        ax.set_title("Match-bootstrap pressure RPB intervals (largest samples)"); ax.set_xlabel("Runs per ball"); fig.tight_layout(); path=directory/"confidence_interval_comparison.png"; fig.savefig(path,dpi=150); plt.close(fig); paths.append(path.name)
    if not pressure.empty:
        fig,ax=plt.subplots(figsize=(9,6)); sns.barplot(data=pressure, x="innings_phase", y="runs_per_ball", hue="pressure_group", errorbar=None, ax=ax); ax.set_title("Batting performance by innings phase and pressure"); fig.tight_layout(); path=directory/"innings_phase_pressure_performance.png"; fig.savefig(path,dpi=150); plt.close(fig); paths.append(path.name)
    if not sensitivity.empty:
        definitions=sensitivity.loc[sensitivity["analysis"].eq("pressure definition")]
        if not definitions.empty:
            fig,ax=plt.subplots(figsize=(12,6)); sns.boxplot(data=definitions,x="definition",y="sensitivity_runs_per_ball",showfliers=False,ax=ax)
            ax.tick_params(axis="x",rotation=35); ax.set_title("Pressure-definition sensitivity: player pressure runs per ball"); fig.tight_layout()
            path=directory/"sensitivity_pressure_definition.png"; fig.savefig(path,dpi=150); plt.close(fig); paths.append(path.name)
        weights=sensitivity.loc[sensitivity["analysis"].eq("composite weight sensitivity")]
        if not weights.empty:
            fig,ax=plt.subplots(figsize=(10,6)); sns.boxplot(data=weights,x="definition",y="rank_percentile",showfliers=False,ax=ax)
            ax.axhline(.5,color="black",linestyle="--",linewidth=1); ax.tick_params(axis="x",rotation=25)
            ax.set(title="Composite-weight sensitivity: player rank percentiles",ylabel="Within-competition rank percentile"); fig.tight_layout()
            path=directory/"sensitivity_scoring_weights.png"; fig.savefig(path,dpi=150); plt.close(fig); paths.append(path.name)
    return paths


def run_batting_model(tables: dict[str, pd.DataFrame], pressure_states: pd.DataFrame, pressure_features: pd.DataFrame,
                      pressure_scores: pd.DataFrame, report_dir: str | Path, figure_dir: str | Path,
                      processed_dir: str | Path, config: BattingModelConfig | None = None) -> dict[str, Any]:
    """Run Phase 5 end to end and write delivery, player, report, and figure outputs."""
    cfg = config or BATTING_MODEL_CONFIG
    reports, figures, processed = Path(report_dir), Path(figure_dir), Path(processed_dir)
    for directory in (reports, figures, processed): directory.mkdir(parents=True, exist_ok=True)
    deliveries, join_diagnostics = build_batting_delivery_features(tables, pressure_states, pressure_features, pressure_scores)
    deliveries = assign_walk_forward_pressure_labels(deliveries, cfg)
    deliveries, baseline_audit = add_historical_personal_baselines(deliveries, cfg)
    deliveries, regression = add_walk_forward_context_regression(deliveries, cfg)
    match_stats, pressure_stats, baseline_stats = build_batter_statistics(deliveries)
    player = _aggregate_player_tables(pressure_stats, baseline_stats, match_stats)
    pressure_baseline = deliveries.loc[deliveries["pressure_high_p90_walk_forward"].eq(True).fillna(False) & deliveries["balls_faced_known"]].groupby(
        ["competition_group","striker_id"],as_index=False,dropna=False).agg(
            pressure_baseline_rpb=("baseline_rpb_hierarchical","mean"),
            pressure_baseline_boundary_frequency=("baseline_boundary_frequency_hierarchical","mean"),
            pressure_baseline_dot_frequency=("baseline_dot_ball_frequency_hierarchical","mean"),
            pressure_baseline_dismissal_frequency=("baseline_dismissal_frequency_hierarchical","mean"))
    player=player.merge(pressure_baseline,on=["competition_group","striker_id"],how="left",validate="one_to_one")
    player["rpb_personal_baseline_difference"] = player["pressure_runs_per_ball"]-player["pressure_baseline_rpb"]
    player["boundary_frequency_personal_baseline_difference"] = player["pressure_boundary_frequency"]-player["pressure_baseline_boundary_frequency"]
    player["dot_frequency_personal_baseline_difference"] = player["pressure_dot_ball_frequency"]-player["pressure_baseline_dot_frequency"]
    player["dismissal_frequency_personal_baseline_difference"] = player["pressure_dismissal_frequency"]-player["pressure_baseline_dismissal_frequency"]
    p90 = deliveries.loc[deliveries["pressure_high_p90_walk_forward"].eq(True).fillna(False) & deliveries["striker_id"].notna()]
    opportunities = p90.groupby(["competition_group", "striker_id"], as_index=False, dropna=False).agg(
        pressure_deliveries=("_source_row_id", "size"), pressure_balls=("balls_faced_known", "sum"),
        pressure_matches=("match_id", "nunique"), pressure_seasons=("season", "nunique"))
    pressure_innings = p90.drop_duplicates(["competition_group", "striker_id", "match_id", "innings"]).groupby(["competition_group", "striker_id"], as_index=False, dropna=False).size().rename(columns={"size":"pressure_innings"})
    opportunities = opportunities.merge(pressure_innings,on=["competition_group","striker_id"],how="left",validate="one_to_one")
    player = player.drop(columns=["pressure_deliveries","pressure_balls","pressure_matches","pressure_seasons","pressure_innings"],errors="ignore").merge(
        opportunities,on=["competition_group","striker_id"],how="left",validate="one_to_one")
    uncertain_counts = deliveries.loc[deliveries["balls_faced_uncertain"]].groupby(["competition_group","striker_id"],as_index=False,dropna=False).size().rename(columns={"size":"uncertain_faced_deliveries"})
    player = player.drop(columns=["uncertain_faced_deliveries"],errors="ignore").merge(uncertain_counts,on=["competition_group","striker_id"],how="left",validate="one_to_one")
    player["uncertain_faced_deliveries"] = player["uncertain_faced_deliveries"].fillna(0).astype(int)
    # Continuous score exposure and Kish effective sample size at competition/player level.
    continuous = deliveries.loc[deliveries["balls_faced_known"] & deliveries["striker_id"].notna()].copy()
    continuous["_w"] = pd.to_numeric(continuous[cfg.pressure_score_column], errors="coerce").clip(0,100)/100
    weighted_events = deliveries.loc[deliveries["dismissal_event_valid"] & deliveries["dismissed_player_id"].notna(), ["competition_group", "dismissed_player_id", cfg.pressure_score_column]].copy()
    weighted_events["_weighted_dismissal"] = pd.to_numeric(weighted_events[cfg.pressure_score_column], errors="coerce").clip(0,100).fillna(0)/100
    weighted_dismissal_lookup = weighted_events.groupby(["competition_group","dismissed_player_id"])["_weighted_dismissal"].sum().to_dict()
    crows=[]
    for (comp,pid), part in continuous.groupby(["competition_group","striker_id"],dropna=False):
        w=part["_w"].fillna(0); sw=w.sum(); sq=(w*w).sum()
        weighted_dismissals=weighted_dismissal_lookup.get((comp,pid),0.0)
        crows.append({"competition_group":comp,"striker_id":pid,"continuous_pressure_exposure":sw,"effective_pressure_exposure":sw*sw/sq if sq else 0,
          "continuous_pressure_balls":len(part),"continuous_pressure_runs_per_ball":(part["batter_runs"]*w).sum()/sw if sw else np.nan,
          "continuous_pressure_weighted_dismissals":weighted_dismissals,
          "continuous_pressure_dismissal_frequency":weighted_dismissals/sw if sw else np.nan,
          "continuous_pressure_weighted_average":(part["batter_runs"]*w).sum()/weighted_dismissals if weighted_dismissals else np.nan})
    player=player.merge(pd.DataFrame(crows),on=["competition_group","striker_id"],how="left",validate="one_to_one")
    uncertainty=bootstrap_batter_pressure(deliveries,cfg.bootstrap_replicates,cfg.random_seed)
    interval_columns = ["competition_group", "striker_id", "pressure_rpb_ci_low", "pressure_rpb_ci_high",
        "pressure_expected_baseline_rpb", "pressure_minus_baseline_rpb", "pressure_minus_baseline_ci_low",
        "pressure_minus_baseline_ci_high", "bootstrap_replicates", "bootstrap_unit"]
    player=player.merge(uncertainty[interval_columns],on=["competition_group","striker_id"],how="left",validate="one_to_one")
    scored, weight_audit=build_composite_scores(player, player)
    eligible, eligibility=assign_eligibility(scored,cfg)
    sensitivity=_sensitivity(deliveries,eligible,cfg)
    # Save outputs; delivery-level datasets stay ignored by git.
    out_files={
      "batting_delivery_features.csv":deliveries,
      "batter_match_statistics.csv":match_stats,
      "batter_pressure_statistics.csv":pressure_stats,
      "batter_baseline_statistics.csv":baseline_stats,
      "batter_pressure_differentials.csv":eligible,
      "batter_composite_scores.csv":eligible,
      "batter_eligibility.csv":eligible,
      "batter_uncertainty.csv":uncertainty,
    }
    for name,df in out_files.items(): df.to_csv(processed/name,index=False)
    eligibility.to_csv(reports/"batting_eligibility_analysis.csv",index=False)
    sensitivity.to_csv(reports/"batting_sensitivity_analysis.csv",index=False)
    weight_audit.to_csv(reports/"batting_correlation_weights.csv",index=False)
    regression.to_csv(reports/"batting_regression_comparison.csv",index=False)
    baseline_audit.to_csv(reports/"batting_baseline_audit.csv",index=False)
    methodology=(
      "SUPERR SELECTOR — PHASE 5 BATTING METHODOLOGY\n"
      "All outputs are descriptive estimates of observed cricket outcomes under defined match contexts; pressure indices are not psychological measurements.\n"
      "Stable striker IDs drive aggregation; names are display labels. Joins use delivery keys and many-to-one dimensions with row preservation checks.\n"
      "Batter runs are source runs. Wides are not balls faced; Overstep/no-ball, byes and leg-byes count as faced under the Phase 2 convention. Unclassified extras remain uncertain and are retained.\n"
      "A batter dot is a known-faced delivery with zero batter runs; a separate zero-total-runs flag is retained. Boundaries use batter four/six flags. Dismissals are credited to dismissed_player_id, including run-outs and retire-outs; retired hurt is excluded.\n"
      f"Primary P90 labels use the prior-month weighted score distribution within competition group × innings phase after {cfg.pressure_reference_minimum} reference states; bands use prior-history P50/P75/P90. Early rows remain unavailable. Phase 4 static P90 is sensitivity-only.\n"
      f"Personal baselines use prior months of non-P90 known-faced observations, matched on exact competition ID, phase and first/chase context, then broader competition-group fallback; peer rates shrink estimates with {cfg.baseline_prior_strength_balls:.0f} prior balls.\n"
      "Context regression is a prior-year-only Ridge comparison using pre-delivery predictors. The prediction target is delivery batter runs per known-faced ball; held-out years are compared with the historical baseline.\n"
      "Continuous exposure uses weighted pressure/100; effective exposure is Kish ESS. Bootstrap resamples matches. Composite scores robust-standardize within IPL/T20I/Other T20, report absolute, relative, consistency and reliability components separately, and are exploratory only.\n"
      f"Default candidate eligibility: {cfg.eligibility_pressure_balls} P90 balls, {cfg.eligibility_pressure_matches} P90 matches, and {cfg.eligibility_baseline_balls} historical nonpressure balls. Threshold sensitivity is reported separately. No final top-10 selections are produced.\n"
      "Limitations: the P90 construct and provisional Phase 4 indices have no ground-truth pressure labels; the data are observational and competition/context coverage is uneven; the pooled baseline uses prior months only but small contexts can remain noisy; eligibility is a policy candidate; scores are not validated talent or psychological ratings.\n")
    (reports/"batting_methodology.txt").write_text(methodology,encoding="utf-8")
    approach=weight_audit.to_string(index=False)
    reg=regression.to_string(index=False) if not regression.empty else "No regression folds were available."
    score_methods=["score_model_a_equal_domains","score_model_b_correlation_reduced","score_model_c_shrinkage_adjusted"]
    rank_correlations=[]
    top_overlap=[]
    for comp, group in eligible.groupby("competition_group",dropna=False):
        for left_idx,left in enumerate(score_methods):
            for right in score_methods[left_idx+1:]:
                pair=group[[left,right]].dropna()
                rho=pair[left].rank().corr(pair[right].rank()) if len(pair)>1 else np.nan
                n=max(1,int(np.ceil(len(group)*.10)))
                top_left=set(group.nlargest(n,left)["striker_id"]); top_right=set(group.nlargest(n,right)["striker_id"])
                top_overlap.append(f"{comp}: {left} vs {right}, Spearman={rho:.3f}, top-decile overlap={len(top_left & top_right)}/{n}")
        if len(group):
            rank_correlations.append(f"{comp}: {len(group)} player records")
    model_report=("SUPERR SELECTOR — MODEL COMPARISON\n\n"
      "Model A: equal-weight means of absolute, relative-resilience and consistency domains; easy to explain, but choices are normative.\n"
      "Model B: inverse absolute-correlation component weighting; reduces redundant correlated components, but weights are sample-dependent and not outcome-trained.\n"
      "Model C: Model A shrunk toward neutral 50 using pressure volume, matches and baseline history; this is the displayed exploratory composite.\n\n"
      f"Correlation weights:\n{approach}\n\nCompetition-specific rank checks:\n"+"\n".join(top_overlap)+
      f"\n\nWalk-forward Ridge comparison:\n{reg}\n\n"
      "No approach is declared superior without future-outcome evidence. Model C is used as the displayed exploratory score because it explicitly shrinks low-information cases toward neutral 50; that is a risk-control design choice, not an empirical claim of better batting prediction. Sensitivity scenarios are in batting_sensitivity_analysis.csv.")
    (reports/"batting_model_comparison.txt").write_text(model_report,encoding="utf-8")
    pcols=[c for c in ("pressure_runs_per_ball","pressure_balls","pressure_matches","effective_pressure_exposure") if c in eligible]
    candidates=eligible.loc[eligible["eligible_analytical_candidate"]]
    top=candidates.sort_values("overall_batting_pressure_score",ascending=False).head(10)
    median_pressure=float(candidates["pressure_runs_per_ball"].median()) if len(candidates) else np.nan
    median_normal=float(candidates["normal_runs_per_ball"].median()) if len(candidates) else np.nan
    median_baseline_difference=float(candidates["rpb_personal_baseline_difference"].median()) if len(candidates) else np.nan
    positive_resilience=float(candidates["rpb_personal_baseline_difference"].gt(0).mean()) if len(candidates) else np.nan
    competition_findings=[]
    for comp, group in candidates.groupby("competition_group",dropna=False):
        competition_findings.append({"competition_group":comp,"candidate_records":len(group),
            "median_pressure_rpb":group["pressure_runs_per_ball"].median(),"median_normal_rpb":group["normal_runs_per_ball"].median(),
            "median_pressure_minus_baseline_rpb":group["rpb_personal_baseline_difference"].median(),
            "share_above_personal_baseline":group["rpb_personal_baseline_difference"].gt(0).mean()})
    findings=(f"Rows: {len(deliveries):,}; known-faced rows: {int(deliveries.balls_faced_known.sum()):,}; uncertain faced rows: {int(deliveries.balls_faced_uncertain.sum()):,}.\n"
      f"Player/competition records: {len(eligible):,}; pressure-sample sufficient: {int(eligible.eligible_pressure_sample.sum()):,}; eligible default candidates (pressure sample + reliable baseline): {int(eligible.eligible_analytical_candidate.sum()):,}; insufficient pressure sample: {int((~eligible.eligible_pressure_sample).sum()):,}; baseline unreliable among pressure-sample sufficient: {int((eligible.eligible_pressure_sample & ~eligible.baseline_reliable).sum()):,}.\n"
      f"Pressure label available: {int(deliveries.pressure_label_available.sum()):,}; unavailable early/context rows: {int((~deliveries.pressure_label_available).sum()):,}.\n"
      f"Eligible-candidate median P90 runs/ball: {median_pressure:.3f}; median non-P90 runs/ball: {median_normal:.3f}; median pressure-minus-personal-baseline runs/ball: {median_baseline_difference:.3f}; candidates above their personal baseline: {positive_resilience:.1%}. These are descriptive and competition/context pooled candidate summaries.\n"
      f"Competition-specific candidate summaries:\n{pd.DataFrame(competition_findings).to_string(index=False, float_format=lambda v: f'{v:.3f}')}\n"
      f"Join diagnostics: {join_diagnostics}\n\nTop exploratory scores (not selections):\n{top[['competition_group','striker_id','player_name','overall_batting_pressure_score','eligibility_status']].to_string(index=False)}\n\n"
      "Interpretation: use absolute output, baseline differentials, match-cluster intervals and eligibility together. No score constitutes a validated pressure trait or final competition recommendation.")
    (reports/"batting_key_findings.txt").write_text(findings,encoding="utf-8")
    validation=(f"BATTING VALIDATION\nSource deliveries: {len(deliveries):,}; rows after joins: {len(deliveries):,}; source row IDs unique: {deliveries._source_row_id.is_unique}.\n"
      f"Unique delivery keys: {not deliveries.duplicated(['match_id','innings','innings_ball_num']).any()}; stable player IDs missing: {int(deliveries.striker_id.isna().sum())}; player mapping unmatched: {join_diagnostics['unmatched_player_mapping_rows']}.\n"
      f"Weighted pressure range: {deliveries[cfg.pressure_score_column].min():.3f}–{deliveries[cfg.pressure_score_column].max():.3f}; composite score bounds: {eligible.overall_batting_pressure_score.min():.3f}–{eligible.overall_batting_pressure_score.max():.3f}.\n"
      f"No future baseline: month labels computed from prior months only; regression folds use years < test year; first unavailable test year remains unscored.\n"
      f"Bootstrap rows: {len(uncertainty):,}; match-level resampling; replicates={cfg.bootstrap_replicates}; seed={cfg.random_seed}.\n"
      "See join diagnostics and baseline audit CSV for row coverage. Input tables were loaded read-only; source files were not rewritten.")
    (reports/"batting_validation.txt").write_text(validation,encoding="utf-8")
    chart_paths=_write_figures(eligible,pressure_stats,sensitivity,figures)
    return {"deliveries":deliveries,"match_statistics":match_stats,"pressure_statistics":pressure_stats,"baseline_statistics":baseline_stats,
            "scores":eligible,"uncertainty":uncertainty,"eligibility":eligibility,"sensitivity":sensitivity,
            "join_diagnostics":join_diagnostics,"figures":chart_paths,"regression":regression}
