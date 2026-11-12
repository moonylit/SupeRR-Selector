"""Player-match and pressure-stratified batting statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num.div(den.where(den.ne(0)))


def _aggregate(frame: pd.DataFrame, keys: list[str], prefix: str = "") -> pd.DataFrame:
    known = frame.loc[frame["balls_faced_known"] & frame["striker_id"].notna()].copy()
    if known.empty:
        return pd.DataFrame(columns=keys)
    known["_fours"] = known["fours_known"].astype(int)
    known["_sixes"] = known["sixes_known"].astype(int)
    known["_boundaries"] = known["boundary_count_known"].gt(0).astype(int)
    known["_dots"] = known["dot_ball_batter_run_known"].astype(int)
    known["_boundary_runs"] = known["boundary_runs_known"].astype(int)
    known["_innings_key"] = known["match_id"].astype("string") + ":" + known["innings"].astype("string")
    agg = known.groupby(keys, as_index=False, dropna=False).agg(
        balls=("balls_faced_known", "sum"), batter_runs=("batter_runs", "sum"), fours=("_fours", "sum"),
        sixes=("_sixes", "sum"), boundary_balls=("_boundaries", "sum"), dot_balls=("_dots", "sum"),
        boundary_runs=("_boundary_runs", "sum"), pressure_deliveries=("_source_row_id", "size"),
        innings_count=("_innings_key", "nunique"), matches=("match_id", "nunique"), seasons=("season", "nunique"),
    )
    event_rows = frame.loc[frame["dismissal_event_valid"] & frame["dismissed_player_id"].notna()].copy()
    if event_rows.empty:
        agg["dismissals"] = 0
    else:
        event_rows["striker_id"] = event_rows["dismissed_player_id"]
        event_agg = event_rows.groupby(keys, as_index=False, dropna=False).size().rename(columns={"size": "dismissals"})
        agg = agg.merge(event_agg, on=keys, how="outer", validate="one_to_one")
    agg["dismissals"] = agg["dismissals"].fillna(0)
    agg["strike_rate"] = 100 * _safe_div(agg["batter_runs"], agg["balls"])
    agg["runs_per_ball"] = _safe_div(agg["batter_runs"], agg["balls"])
    agg["boundary_frequency"] = _safe_div(agg["boundary_balls"], agg["balls"])
    agg["four_frequency"] = _safe_div(agg["fours"], agg["balls"])
    agg["six_frequency"] = _safe_div(agg["sixes"], agg["balls"])
    agg["boundary_run_share"] = _safe_div(agg["boundary_runs"], agg["batter_runs"])
    agg["dot_ball_frequency"] = _safe_div(agg["dot_balls"], agg["balls"])
    agg["dismissal_frequency"] = _safe_div(agg["dismissals"], agg["balls"])
    agg["batting_average"] = _safe_div(agg["batter_runs"], agg["dismissals"])
    if prefix:
        rename = {c: f"{prefix}{c}" for c in agg.columns if c not in keys}
        agg = agg.rename(columns=rename)
    return agg


def build_batter_statistics(deliveries: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build match, pressure band, continuous exposure, and baseline tables."""
    # Aggregate on stable IDs; source display-name variation must not split a batter's record.
    identity = ["competition_group", "comp_id", "season", "striker_id", "batting_team_id"]
    player_match_keys = identity + ["match_id", "innings"]
    match = _aggregate(deliveries, player_match_keys)
    display = deliveries.groupby(["comp_id", "striker_id", "batting_team_id"], as_index=False, dropna=False).agg(
        player_name=("player_name", "first"), batting_team_name=("batting_team_name", "first"),
        opponent_team_name=("opponent_team_name", "first"))
    if not match.empty:
        match = match.merge(display, on=["comp_id", "striker_id", "batting_team_id"], how="left", validate="many_to_one")
    if not match.empty:
        by_match = match.groupby(["competition_group", "striker_id"], dropna=False)["runs_per_ball"]
        match["player_match_rpb_mean"] = by_match.transform("mean")
        match["player_match_rpb_median"] = by_match.transform("median")
        match["player_match_rpb_std"] = by_match.transform("std")
        match["player_match_rpb_iqr"] = by_match.transform(lambda x: x.quantile(.75) - x.quantile(.25))
        match["player_match_rpb_cv"] = match["player_match_rpb_std"] / match["player_match_rpb_mean"].abs().replace(0, np.nan)

    frame = deliveries.copy()
    frame["pressure_group"] = np.select(
        [frame["pressure_high_p90_walk_forward"].eq(True).fillna(False), frame["pressure_high_p90_walk_forward"].eq(False).fillna(False)],
        ["walk-forward P90 pressure", "non-P90 known reference"], default="pressure label unavailable")
    player_keys = identity + ["innings_phase", "innings_context"]
    stat_rows = []
    for label, subset in frame.groupby("pressure_group", dropna=False):
        piece = _aggregate(subset, player_keys)
        piece["pressure_group"] = label
        stat_rows.append(piece)
    for band in ("Low", "Moderate", "High", "Extreme"):
        subset = frame.loc[frame["pressure_band_walk_forward"].eq(band)]
        if not subset.empty:
            piece = _aggregate(subset, player_keys)
            piece["pressure_group"] = f"walk-forward band: {band}"
            stat_rows.append(piece)
    # Continuous-pressure weighted figures, with pressure/100 as an intensity weight.
    weighted_rows = []
    valid = frame.loc[frame["balls_faced_known"] & frame["striker_id"].notna() & frame["batting_pressure_weighted"].notna()].copy()
    valid["_w"] = pd.to_numeric(valid["batting_pressure_weighted"], errors="coerce").clip(0, 100) / 100
    valid["_w_runs"] = valid["batter_runs"] * valid["_w"]
    valid["_w_boundary"] = valid["boundary_count_known"] * valid["_w"]
    valid["_w_four"] = valid["fours_known"].astype(int) * valid["_w"]
    valid["_w_six"] = valid["sixes_known"].astype(int) * valid["_w"]
    valid["_w_dot"] = valid["dot_ball_batter_run_known"].astype(int) * valid["_w"]
    valid["_w_sq"] = valid["_w"] ** 2
    for keys, group in valid.groupby(player_keys, dropna=False):
        if not isinstance(keys, tuple): keys = (keys,)
        rec = dict(zip(player_keys, keys))
        s = group["_w"].sum()
        rec.update(weighted_pressure_exposure=s,
            effective_pressure_exposure=(s * s / group["_w_sq"].sum()) if group["_w_sq"].sum() else 0,
            weighted_balls=s, weighted_runs=group["_w_runs"].sum(),
            weighted_runs_per_ball=group["_w_runs"].sum() / s if s else np.nan,
            weighted_boundary_frequency=group["_w_boundary"].sum() / s if s else np.nan,
            weighted_four_frequency=group["_w_four"].sum() / s if s else np.nan,
            weighted_six_frequency=group["_w_six"].sum() / s if s else np.nan,
            weighted_dot_frequency=group["_w_dot"].sum() / s if s else np.nan,
            continuous_pressure_deliveries=len(group), weighted_matches=group["match_id"].nunique())
        weighted_rows.append(rec)
    continuous = pd.DataFrame(weighted_rows)
    pressure = pd.concat(stat_rows, ignore_index=True) if stat_rows else pd.DataFrame()
    if not pressure.empty:
        pressure = pressure.merge(display, on=["comp_id", "striker_id", "batting_team_id"], how="left", validate="many_to_one")
    if not pressure.empty and not continuous.empty:
        pressure = pressure.merge(continuous, on=player_keys, how="left", validate="many_to_one")
    # Personal baseline is time-safe and context-specific at each row; summarize exposure-weighted values.
    base = frame.loc[frame["balls_faced_known"]].groupby(player_keys, as_index=False, dropna=False).agg(
        baseline_rpb=("baseline_rpb_hierarchical", "mean"),
        baseline_boundary_frequency=("baseline_boundary_frequency_hierarchical", "mean"),
        baseline_four_frequency=("baseline_four_frequency_hierarchical", "mean"),
        baseline_six_frequency=("baseline_six_frequency_hierarchical", "mean"),
        baseline_dot_ball_frequency=("baseline_dot_ball_frequency_hierarchical", "mean"),
        baseline_dismissal_frequency=("baseline_dismissal_frequency_hierarchical", "mean"),
        baseline_history_balls=("baseline_player_history_balls", "max"),
        baseline_pool_balls=("baseline_pool_balls", "max"),
        baseline_shrinkage_fraction=("baseline_shrinkage_fraction", "mean"),
    )
    # Pressure and normal summaries drive transparent differentials.
    if not pressure.empty:
        p = pressure.pivot_table(index=player_keys, columns="pressure_group", values=["strike_rate", "runs_per_ball", "boundary_frequency", "dot_ball_frequency", "dismissal_frequency"], aggfunc="first")
        p.columns = [f"{metric}_{group}" for metric, group in p.columns]
        p = p.reset_index()
        base = base.merge(p, on=player_keys, how="left", validate="one_to_one")
        def col(metric, group): return f"{metric}_{group}"
        pairs = [("strike_rate", "walk-forward P90 pressure", "non-P90 known reference", True),
                 ("runs_per_ball", "walk-forward P90 pressure", "non-P90 known reference", True),
                 ("boundary_frequency", "walk-forward P90 pressure", "non-P90 known reference", True),
                 ("dot_ball_frequency", "walk-forward P90 pressure", "non-P90 known reference", False),
                 ("dismissal_frequency", "walk-forward P90 pressure", "non-P90 known reference", False)]
        for metric, pressure_group, normal_group, higher_better in pairs:
            high, normal = col(metric, pressure_group), col(metric, normal_group)
            base[f"pressure_{metric}"] = base.get(high, np.nan)
            base[f"normal_{metric}"] = base.get(normal, np.nan)
            base[f"{metric}_absolute_difference"] = base[f"pressure_{metric}"] - base[f"normal_{metric}"]
            base[f"{metric}_relative_change_pct"] = 100 * base[f"{metric}_absolute_difference"] / base[f"normal_{metric}"].abs().replace(0, np.nan)
    return match, pressure, base
