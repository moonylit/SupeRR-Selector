"""Pressure/normal bowling summaries and match-level consistency features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.bowling_features import aggregate_bowling_counts

PLAYER_KEYS = ["competition_group", "bowler_id"]


def build_pressure_statistics(deliveries: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return contextual delivery summaries, competition-group summaries and match rows."""
    labeled = deliveries.loc[
        deliveries["bowler_id"].notna()
        & deliveries["bowling_pressure_band_walk_forward"].isin(["high", "normal", "low", "elevated"])
    ].copy()
    detailed_keys = [
        "competition_group", "comp_id", "season", "bowler_id", "innings_phase",
        "innings_context", "bowling_pressure_band_walk_forward",
    ]
    detailed = aggregate_bowling_counts(labeled, detailed_keys)
    detailed = detailed.rename(columns={"bowling_pressure_band_walk_forward": "pressure_band"})

    player_keys = ["competition_group", "bowler_id", "bowling_pressure_band_walk_forward"]
    player = aggregate_bowling_counts(labeled, player_keys).rename(
        columns={"bowling_pressure_band_walk_forward": "pressure_band"}
    )
    match_keys = ["competition_group", "bowler_id", "match_id", "bowling_pressure_band_walk_forward"]
    match = aggregate_bowling_counts(labeled, match_keys).rename(
        columns={"bowling_pressure_band_walk_forward": "pressure_band"}
    )
    return detailed, player, match


def _weighted_baselines(high: pd.DataFrame) -> pd.DataFrame:
    metrics = ["economy", "wickets_per_over", "dot_rate", "boundary_rate", "extras_per_over"]
    rows: list[dict[str, object]] = []
    for keys, group in high.groupby(PLAYER_KEYS, dropna=False, sort=False):
        weight = pd.to_numeric(group["legal_ball_known"], errors="coerce").fillna(0)
        row: dict[str, object] = dict(zip(PLAYER_KEYS, keys if isinstance(keys, tuple) else (keys,)))
        row["baseline_coverage_fraction"] = float(group["baseline_available"].fillna(False).mean())
        row["baseline_history_balls"] = float(group["baseline_history_balls"].median())
        row["baseline_peer_balls"] = float(group["baseline_peer_balls"].median())
        row["baseline_pool_source"] = (
            group["baseline_pool_source"].mode(dropna=True).iloc[0]
            if group["baseline_pool_source"].notna().any() else "no prior normal-pressure baseline"
        )
        for metric in metrics:
            column = f"baseline_{metric}"
            values = pd.to_numeric(group[column], errors="coerce")
            valid = values.notna() & weight.gt(0)
            row[column] = float(np.average(values.loc[valid], weights=weight.loc[valid])) if valid.any() else np.nan
            raw = pd.to_numeric(group[f"baseline_{metric}_raw"], errors="coerce")
            row[f"{column}_raw"] = float(np.average(raw.loc[raw.notna() & weight.gt(0)], weights=weight.loc[raw.notna() & weight.gt(0)])) if (raw.notna() & weight.gt(0)).any() else np.nan
            peer = pd.to_numeric(group[f"{column}_peer"], errors="coerce")
            row[f"{column}_peer"] = float(np.average(peer.loc[peer.notna() & weight.gt(0)], weights=weight.loc[peer.notna() & weight.gt(0)])) if (peer.notna() & weight.gt(0)).any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _match_consistency(match_stats: pd.DataFrame) -> pd.DataFrame:
    high = match_stats.loc[match_stats["pressure_band"].eq("high")].copy()
    high = high.loc[high["legal_balls_known"].gt(0) & high["economy"].notna()]
    if high.empty:
        return pd.DataFrame(columns=PLAYER_KEYS)
    # A match-level score is a simple mean of four within-competition
    # percentiles, with favorable directions made explicit.
    for column, higher in (
        ("economy", False), ("wickets_per_legal_over", True),
        ("bowler_dot_rate", True), ("boundary_concession_rate", False),
    ):
        high[f"match_{column}_pct"] = high.groupby("competition_group", dropna=False)[column].rank(
            method="average", pct=True
        )
        if not higher:
            high[f"match_{column}_pct"] = 1 - high[f"match_{column}_pct"]
    component_columns = [
        "match_economy_pct", "match_wickets_per_legal_over_pct",
        "match_bowler_dot_rate_pct", "match_boundary_concession_rate_pct",
    ]
    high["match_performance_index"] = high[component_columns].mean(axis=1) * 100
    high["match_meets_competition_standard"] = high["match_performance_index"].ge(
        high.groupby("competition_group", dropna=False)["match_performance_index"].transform("median")
    )
    return high.groupby(PLAYER_KEYS, as_index=False, dropna=False).agg(
        consistency_matches=("match_id", "nunique"),
        match_performance_median=("match_performance_index", "median"),
        match_performance_mean=("match_performance_index", "mean"),
        match_performance_std=("match_performance_index", "std"),
        match_performance_mad=("match_performance_index", lambda s: float((s - s.median()).abs().median())),
        match_performance_p10=("match_performance_index", lambda s: float(s.quantile(0.10))),
        matches_meeting_standard_fraction=("match_meets_competition_standard", "mean"),
        pressure_match_economy_std=("economy", "std"),
        pressure_match_economy_median=("economy", "median"),
        pressure_match_wicket_rate_std=("wickets_per_legal_over", "std"),
    )


def build_bowler_pressure_differentials(
    deliveries: pd.DataFrame,
    player_pressure_stats: pd.DataFrame,
    match_pressure_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Compare high-pressure outcomes with normal performance and prior baselines."""
    wide = player_pressure_stats.pivot(
        index=PLAYER_KEYS, columns="pressure_band",
        values=[
            "delivery_rows", "matches", "innings", "legal_balls_known", "legal_balls_possible",
            "overs_known", "overs_possible", "runs_conceded_lower", "runs_conceded_upper",
            "runs_per_legal_ball", "boundary_runs_conceded", "bowler_wickets",
            "legal_bowler_dots", "legal_team_dots", "boundary_deliveries",
            "fours_conceded", "sixes_conceded", "wide_deliveries", "no_ball_deliveries",
            "bowler_extra_runs_known", "economy", "economy_lower_bound", "economy_upper_bound",
            "wickets_per_legal_over", "wickets_per_100_legal_balls",
            "bowling_strike_rate_balls_per_wicket", "bowler_dot_rate", "team_dot_rate",
            "boundary_concession_rate", "four_rate", "six_rate", "wides_per_legal_over",
            "no_balls_per_legal_over", "bowler_extras_per_legal_over",
            "bowler_extra_runs_per_legal_over_upper",
        ],
    )
    wide.columns = [f"{metric}_{band}" for metric, band in wide.columns]
    wide = wide.reset_index()
    rate_columns = {
        "economy", "economy_lower_bound", "economy_upper_bound", "runs_per_legal_ball",
        "wickets_per_legal_over", "wickets_per_100_legal_balls",
        "bowling_strike_rate_balls_per_wicket", "bowler_dot_rate", "team_dot_rate",
        "boundary_concession_rate", "four_rate", "six_rate", "wides_per_legal_over",
        "no_balls_per_legal_over", "bowler_extras_per_legal_over",
        "bowler_extra_runs_per_legal_over_upper",
    }
    missing_columns: dict[str, float] = {}
    for band in ("high", "normal", "low", "elevated"):
        for metric in [
            "delivery_rows", "matches", "innings", "legal_balls_known", "legal_balls_possible",
            "overs_known", "overs_possible", "runs_conceded_lower", "runs_conceded_upper",
            "runs_per_legal_ball", "boundary_runs_conceded", "bowler_wickets",
            "legal_bowler_dots", "legal_team_dots", "boundary_deliveries",
            "fours_conceded", "sixes_conceded", "wide_deliveries", "no_ball_deliveries",
            "bowler_extra_runs_known", "economy", "economy_lower_bound", "economy_upper_bound",
            "wickets_per_legal_over", "wickets_per_100_legal_balls",
            "bowling_strike_rate_balls_per_wicket", "bowler_dot_rate", "team_dot_rate",
            "boundary_concession_rate", "four_rate", "six_rate", "wides_per_legal_over",
            "no_balls_per_legal_over", "bowler_extras_per_legal_over",
            "bowler_extra_runs_per_legal_over_upper",
        ]:
            column = f"{metric}_{band}"
            if column not in wide:
                missing_columns[column] = np.nan if metric in rate_columns else 0.0
    if missing_columns:
        wide = pd.concat([wide, pd.DataFrame(missing_columns, index=wide.index)], axis=1)
    wide = wide.copy()
    count_metrics = {
        "delivery_rows", "matches", "innings", "legal_balls_known", "legal_balls_possible",
        "overs_known", "overs_possible", "runs_conceded_lower", "runs_conceded_upper",
        "boundary_runs_conceded",
        "bowler_wickets", "legal_bowler_dots", "legal_team_dots", "boundary_deliveries",
        "fours_conceded", "sixes_conceded", "wide_deliveries", "no_ball_deliveries",
        "bowler_extra_runs_known",
    }
    for band in ("high", "normal", "low", "elevated"):
        for metric in count_metrics:
            column = f"{metric}_{band}"
            if column in wide:
                wide[column] = wide[column].fillna(0)
    wide["pressure_matches"] = wide.get("matches_high", 0)
    wide["pressure_innings"] = wide.get("innings_high", 0)
    wide["pressure_legal_balls"] = wide.get("legal_balls_known_high", 0)
    wide["pressure_overs"] = wide["pressure_legal_balls"] / 6

    high = deliveries.loc[
        deliveries["bowler_id"].notna()
        & deliveries["bowling_pressure_band_walk_forward"].eq("high")
    ]
    directions = {
        "economy": "lower", "wickets_per_over": "higher", "dot_rate": "higher",
        "boundary_rate": "lower", "extras_per_over": "lower",
    }
    baseline = _weighted_baselines(high)
    if baseline.empty:
        baseline = pd.DataFrame(columns=PLAYER_KEYS + [
            "baseline_coverage_fraction", "baseline_history_balls", "baseline_peer_balls",
            "baseline_pool_source", *[f"baseline_{metric}" for metric in directions],
            *[f"baseline_{metric}_raw" for metric in directions],
            *[f"baseline_{metric}_peer" for metric in directions],
        ])
    consistency = _match_consistency(match_pressure_stats)
    result = wide.merge(baseline, on=PLAYER_KEYS, how="left", validate="one_to_one").merge(
        consistency, on=PLAYER_KEYS, how="left", validate="one_to_one"
    )
    observed_map = {
        "economy": "economy_high", "wickets_per_over": "wickets_per_legal_over_high",
        "dot_rate": "bowler_dot_rate_high", "boundary_rate": "boundary_concession_rate_high",
        "extras_per_over": "bowler_extras_per_legal_over_high",
    }
    additions: dict[str, pd.Series] = {}
    for metric, direction in directions.items():
        observed = pd.to_numeric(result.get(observed_map[metric]), errors="coerce")
        expected = pd.to_numeric(result.get(f"baseline_{metric}"), errors="coerce")
        additions[f"pressure_{metric}"] = observed
        additions[f"relative_{metric}_improvement"] = expected - observed if direction == "lower" else observed - expected
        additions[f"relative_{metric}_raw_improvement"] = (
            pd.to_numeric(result.get(f"baseline_{metric}_raw"), errors="coerce") - observed
            if direction == "lower" else observed - pd.to_numeric(result.get(f"baseline_{metric}_raw"), errors="coerce")
        )
    result = pd.concat([result, pd.DataFrame(additions, index=result.index)], axis=1).copy()
    result["baseline_reliable"] = (
        result["baseline_history_balls"].fillna(0).ge(24)
        & result["baseline_coverage_fraction"].fillna(0).ge(0.25)
    )
    return result
