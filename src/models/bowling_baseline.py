"""Walk-forward bowling-pressure labels and prior-match baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import BowlingModelConfig

CONTEXT_KEYS = ["competition_group", "comp_id", "innings_phase", "innings_context"]
BASELINE_COUNTS = [
    "legal_balls_known", "runs_conceded", "bowler_wickets", "legal_bowler_dots",
    "boundary_deliveries", "bowler_extra_runs", "normal_matches", "normal_innings",
]
RATE_SPECS = {
    "economy": ("runs_conceded", "legal_balls_known", 6.0, "economy_prior_balls"),
    "wickets_per_over": ("bowler_wickets", "legal_balls_known", 6.0, "wicket_prior_balls"),
    "dot_rate": ("legal_bowler_dots", "legal_balls_known", 1.0, "event_rate_prior_balls"),
    "boundary_rate": ("boundary_deliveries", "legal_balls_known", 1.0, "event_rate_prior_balls"),
    "extras_per_over": ("bowler_extra_runs", "legal_balls_known", 6.0, "extras_prior_balls"),
}


def assign_walk_forward_pressure_bands(
    deliveries: pd.DataFrame,
    config: BowlingModelConfig | None = None,
    score_column: str | None = None,
    suffix: str = "",
    copy_frame: bool = True,
) -> pd.DataFrame:
    """Classify a month using earlier months only, within group and phase."""
    cfg = config or BowlingModelConfig()
    column = score_column or cfg.weighted_pressure_column
    band_col = f"bowling_pressure_band_walk_forward{suffix}"
    available_col = f"bowling_pressure_label_available{suffix}"
    count_col = f"bowling_pressure_reference_state_count{suffix}"
    frame = deliveries.copy() if copy_frame else deliveries
    month = pd.to_datetime(frame["match_date"], errors="coerce").dt.to_period("M")
    values = pd.to_numeric(frame[column], errors="coerce")
    frame[band_col] = "reference unavailable"
    frame[available_col] = False
    frame[count_col] = 0
    groups = frame.groupby(["competition_group", "innings_phase"], dropna=False, sort=False).groups
    low_q, median_q, normal_q, high_q = cfg.pressure_percentiles
    for indices in groups.values():
        idx = pd.Index(indices)
        group_month = month.loc[idx]
        history: list[float] = []
        for current_month in sorted(group_month.dropna().unique()):
            month_idx = idx[group_month.eq(current_month).to_numpy()]
            current = values.loc[month_idx].dropna()
            if len(history) >= max(1, cfg.pressure_reference_minimum):
                low, _median, normal, high = np.quantile(
                    history, [low_q, median_q, normal_q, high_q]
                )
                labels = np.select(
                    [current.lt(low), current.lt(normal), current.lt(high)],
                    ["low", "normal", "elevated"],
                    default="high",
                )
                frame.loc[current.index, band_col] = labels
                frame.loc[current.index, available_col] = True
                frame.loc[current.index, count_col] = len(history)
            history.extend(current.astype(float).tolist())
    return frame


def _daily_prior(
    source: pd.DataFrame,
    target: pd.DataFrame,
    keys: list[str],
) -> pd.DataFrame:
    """Join cumulative counts strictly before each target calendar date."""
    group_cols = [*keys, "match_date"]
    daily = source.groupby(group_cols, as_index=False, dropna=False).agg(
        **{column: (column, "sum") for column in BASELINE_COUNTS}
    ).sort_values("match_date", kind="stable")
    query = target[keys + ["match_date"]].drop_duplicates()
    # Add zero-exposure rows for target dates that contain no normal-pressure
    # history. Without them, an as-of baseline would incorrectly appear absent.
    daily = query.merge(daily, on=keys + ["match_date"], how="outer", validate="one_to_one")
    daily[BASELINE_COUNTS] = daily[BASELINE_COUNTS].fillna(0)
    daily = daily.sort_values("match_date", kind="stable")
    if keys:
        for column in BASELINE_COUNTS:
            daily[f"prior_{column}"] = daily.groupby(keys, dropna=False, sort=False)[column].cumsum() - daily[column]
    else:
        for column in BASELINE_COUNTS:
            daily[f"prior_{column}"] = daily[column].cumsum() - daily[column]
    prior_columns = [f"prior_{column}" for column in BASELINE_COUNTS]
    return query.merge(
        daily[keys + ["match_date"] + prior_columns],
        on=keys + ["match_date"], how="left", validate="many_to_one",
    )


def add_walk_forward_bowling_baselines(
    deliveries: pd.DataFrame,
    config: BowlingModelConfig | None = None,
    strategy: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach normal-pressure baselines using only matches on earlier dates.

    All matches on the current calendar date are excluded because the source
    does not provide reliable within-day chronology. Personal estimates are
    partially pooled toward peer rates with metric-specific exposure priors.
    """
    cfg = config or BowlingModelConfig()
    strategy = strategy or cfg.baseline_strategy
    if strategy not in {"strict_competition", "legacy_hierarchical", "format_hierarchical", "broader_t20"}:
        raise ValueError(f"Unknown bowling baseline strategy: {strategy}")
    # Allow deterministic scenario rescoring on a frame that already carries a
    # baseline strategy; never merge stale baseline fields into new estimates.
    stale_baseline_columns = [
        column for column in deliveries.columns
        if column.startswith(("baseline_", "own_", "peer_"))
        or column == "selected_baseline_level"
    ]
    frame = deliveries.drop(columns=stale_baseline_columns).copy()
    frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")
    history = frame.loc[
        frame["bowler_id"].notna()
        & frame["match_date"].notna()
        & frame["bowling_pressure_band_walk_forward"].eq("normal")
    ].copy()
    history["runs_conceded"] = pd.to_numeric(history["bowler_runs_conceded_lower"], errors="coerce").fillna(0)
    history["bowler_wickets"] = pd.to_numeric(history["bowler_wicket_calculated"], errors="coerce").fillna(0)
    history["legal_bowler_dots"] = pd.to_numeric(history["bowler_dot_legal"], errors="coerce").fillna(0)
    history["boundary_deliveries"] = pd.to_numeric(history["boundary_delivery"], errors="coerce").fillna(0)
    history["bowler_extra_runs"] = pd.to_numeric(history["bowler_extra_runs_known"], errors="coerce").fillna(0)
    history["legal_balls_known"] = pd.to_numeric(history["legal_ball_known"], errors="coerce").fillna(0)
    history["normal_matches"] = 0
    history["normal_innings"] = 0

    match_counts = history.groupby(
        [*CONTEXT_KEYS, "bowler_id", "match_date"], as_index=False, dropna=False
    ).agg(normal_matches=("match_id", "nunique"), normal_innings=("innings", "nunique"))
    # Match and innings counts are attached to the daily count table without
    # counting deliveries as independent observations.
    daily_counts = history.groupby(
        [*CONTEXT_KEYS, "bowler_id", "match_date"], as_index=False, dropna=False
    )[BASELINE_COUNTS[:6]].sum().merge(
        match_counts, on=[*CONTEXT_KEYS, "bowler_id", "match_date"],
        how="left", validate="one_to_one",
    )
    # A bowler can appear in several innings on one date; count unique match
    # innings with a compound key rather than distinct innings number alone.
    history["_match_innings_key"] = (
        history["match_id"].astype("string") + ":" + history["innings"].astype("string")
    )
    innings_counts = history.groupby(
        [*CONTEXT_KEYS, "bowler_id", "match_date"], as_index=False, dropna=False
    ).agg(normal_innings=("_match_innings_key", "nunique"))
    daily_counts = daily_counts.drop(columns="normal_innings").merge(
        innings_counts, on=[*CONTEXT_KEYS, "bowler_id", "match_date"],
        how="left", validate="one_to_one",
    )
    # Candidate baseline targets are one row per bowler/context/date.
    targets = frame.loc[
        frame["bowler_id"].notna() & frame["match_date"].notna(),
        [*CONTEXT_KEYS, "bowler_id", "match_date"],
    ].drop_duplicates().reset_index(drop=True)
    if targets.empty:
        return frame, pd.DataFrame()

    # comp_id is season/competition-specific in this pack. Retain it at the
    # narrowest peer level, then widen only through explicit, auditable pools.
    all_levels = [
        (0, ["competition_group", "comp_id", "innings_phase", "innings_context"]),
        (1, ["competition_group", "innings_phase", "innings_context"]),
        (2, ["competition_group", "innings_phase"]),
        (3, ["competition_group"]),
    ]
    if strategy == "strict_competition":
        levels = all_levels[:1]
        personal_keys = ["competition_group", "comp_id", "innings_phase", "innings_context", "bowler_id"]
        personal_scope = "same competition/season + format + phase + innings context"
    elif strategy == "legacy_hierarchical":
        levels = all_levels
        personal_keys = []
        personal_scope = "selected peer hierarchy (legacy; may reset across competition/season)"
    elif strategy == "broader_t20":
        # Sensitivity only: pool peer rates across competition groups while
        # retaining phase and innings context. Scoring environments may differ.
        levels = [(4, ["innings_phase", "innings_context"])]
        personal_keys = ["innings_phase", "innings_context", "bowler_id"]
        personal_scope = "all T20 groups + phase + innings context (sensitivity only)"
    else:
        levels = all_levels
        # Accumulate the player's own normal-pressure record across seasons
        # within the same format and tactical phase/context. Peer rates retain
        # the selected competition-aware hierarchy below.
        personal_keys = ["competition_group", "innings_phase", "innings_context", "bowler_id"]
        personal_scope = "same format + phase + innings context across competitions"
    candidates: list[pd.DataFrame] = []
    for level_id, level in levels:
        own_keys = [*level, "bowler_id"]
        own_prior = _daily_prior(daily_counts, targets, own_keys)
        pool_daily = daily_counts.groupby(
            [*level, "match_date"], as_index=False, dropna=False
        )[BASELINE_COUNTS].sum()
        pool_prior = _daily_prior(pool_daily, targets, level)
        target_keys = [*CONTEXT_KEYS, "bowler_id", "match_date"]
        joined = targets.merge(
            own_prior, on=own_keys + ["match_date"], how="left", validate="many_to_one"
        ).merge(
            pool_prior, on=level + ["match_date"], how="left", validate="many_to_one",
            suffixes=("_own", "_pool"),
        )
        row: dict[str, object] = {key: joined[key] for key in target_keys}
        for column in BASELINE_COUNTS:
            own = pd.to_numeric(joined.get(f"prior_{column}_own"), errors="coerce").fillna(0)
            pool = pd.to_numeric(joined.get(f"prior_{column}_pool"), errors="coerce").fillna(0)
            row[f"own_{column}"] = own.to_numpy()
            row[f"peer_{column}"] = (pool - own).clip(lower=0).to_numpy()
        row["pool_source"] = "+".join(level) if level else "all prior normal-pressure data"
        candidate = pd.DataFrame(row)
        candidate["pool_source"] = row["pool_source"]
        candidate["_level_id"] = level_id
        candidates.append(candidate)

    selected = pd.DataFrame(index=targets.index)
    selected["pool_source"] = pd.NA
    selected["selected_baseline_level"] = pd.NA
    for column in BASELINE_COUNTS:
        selected[f"own_{column}"] = np.nan
        selected[f"peer_{column}"] = np.nan
    for candidate in candidates:
        eligible = selected["pool_source"].isna() & candidate["peer_legal_balls_known"].ge(
            cfg.minimum_peer_baseline_balls
        )
        for column in BASELINE_COUNTS:
            selected.loc[eligible, f"own_{column}"] = candidate.loc[eligible, f"own_{column}"]
            selected.loc[eligible, f"peer_{column}"] = candidate.loc[eligible, f"peer_{column}"]
        selected.loc[eligible, "pool_source"] = candidate.loc[eligible, "pool_source"]
        selected.loc[eligible, "selected_baseline_level"] = candidate.loc[eligible, "_level_id"]
    # If no candidate meets the minimum peer exposure, retain the broadest
    # non-empty peer pool rather than silently dropping the player.
    for candidate in reversed(candidates):
        fallback = selected["pool_source"].isna() & candidate["peer_legal_balls_known"].gt(0)
        for column in BASELINE_COUNTS:
            selected.loc[fallback, f"own_{column}"] = candidate.loc[fallback, f"own_{column}"]
            selected.loc[fallback, f"peer_{column}"] = candidate.loc[fallback, f"peer_{column}"]
        selected.loc[fallback, "pool_source"] = candidate.loc[fallback, "pool_source"]
        selected.loc[fallback, "selected_baseline_level"] = candidate.loc[fallback, "_level_id"]
    # With no historical peers, use the bowler's own earlier normal-pressure
    # rate as a clearly flagged, unpooled fallback.
    for candidate in candidates:
        fallback = selected["pool_source"].isna() & candidate["own_legal_balls_known"].gt(0)
        for column in BASELINE_COUNTS:
            selected.loc[fallback, f"own_{column}"] = candidate.loc[fallback, f"own_{column}"]
            selected.loc[fallback, f"peer_{column}"] = 0.0
        selected.loc[fallback, "pool_source"] = "player-only prior history"
        selected.loc[fallback, "selected_baseline_level"] = candidate.loc[fallback, "_level_id"]
    if strategy in {"format_hierarchical", "broader_t20"}:
        personal_prior = _daily_prior(daily_counts, targets, personal_keys)
        personal = targets.merge(
            personal_prior, on=personal_keys + ["match_date"],
            how="left", validate="many_to_one",
        )
        for column in BASELINE_COUNTS:
            selected[f"own_{column}"] = pd.to_numeric(
                personal[f"prior_{column}"], errors="coerce"
            ).fillna(0).to_numpy()
    selected["baseline_pool_source"] = selected["pool_source"].fillna("no prior peer history")
    selected["baseline_history_balls"] = selected["own_legal_balls_known"].fillna(0)
    selected["baseline_history_matches"] = selected["own_normal_matches"].fillna(0)
    selected["baseline_history_innings"] = selected["own_normal_innings"].fillna(0)
    selected["baseline_peer_balls"] = selected["peer_legal_balls_known"].fillna(0)
    selected["baseline_selected_level"] = selected["selected_baseline_level"]
    selected["baseline_strategy"] = strategy
    selected["baseline_personal_history_scope"] = personal_scope

    for metric, (numerator, denominator, scale, prior_attr) in RATE_SPECS.items():
        prior_strength = float(getattr(cfg, prior_attr))
        own_den = selected[f"own_{denominator}"].fillna(0)
        peer_den = selected[f"peer_{denominator}"].fillna(0)
        own_num = selected[f"own_{numerator}"].fillna(0)
        peer_num = selected[f"peer_{numerator}"].fillna(0)
        own_raw = own_num.div(own_den.replace(0, np.nan)) * scale
        peer_rate = peer_num.div(peer_den.replace(0, np.nan))
        shrunk = (own_num + peer_rate * prior_strength).div(own_den + prior_strength) * scale
        baseline = shrunk.where(peer_rate.notna(), own_raw)
        selected[f"baseline_{metric}"] = baseline
        selected[f"baseline_{metric}_raw"] = own_raw
        selected[f"baseline_{metric}_peer"] = peer_rate * scale
        selected[f"baseline_{metric}_prior_balls"] = prior_strength
        selected[f"baseline_{metric}_shrinkage_fraction"] = own_den.div(own_den + prior_strength)
        peer_rate = peer_num.div(peer_den.replace(0, np.nan))
        combined_den = own_den + prior_strength
        own_variance = own_num.clip(lower=0)
        peer_variance = peer_rate.clip(lower=0).div(peer_den.replace(0, np.nan))
        variance = (own_variance + prior_strength**2 * peer_variance).div(
            combined_den.pow(2).replace(0, np.nan)
        ) * scale**2
        own_only_variance = own_variance.div(own_den.pow(2).replace(0, np.nan)) * scale**2
        variance = variance.where(peer_rate.notna(), own_only_variance)
        selected[f"baseline_{metric}_approx_se"] = np.sqrt(variance.clip(lower=0))
    selected["baseline_available"] = selected[[f"baseline_{metric}" for metric in RATE_SPECS]].notna().any(axis=1)
    lookup = targets.merge(selected.reset_index(drop=True), left_index=True, right_index=True, validate="one_to_one")
    baseline_columns = [
        column for column in lookup.columns
        if column.startswith("baseline_") or column.startswith("own_") or column.startswith("peer_")
    ]
    frame = frame.merge(
        lookup[[*CONTEXT_KEYS, "bowler_id", "match_date", *baseline_columns]],
        on=[*CONTEXT_KEYS, "bowler_id", "match_date"], how="left", validate="many_to_one",
    )
    audit = frame.groupby(
        ["competition_group", "innings_phase", "innings_context", "bowler_id"],
        as_index=False, dropna=False,
    ).agg(
        delivery_rows=("match_id", "size"),
        baseline_available_rows=("baseline_available", "sum"),
        baseline_coverage_fraction=("baseline_available", "mean"),
        median_baseline_history_balls=("baseline_history_balls", "median"),
        median_baseline_peer_balls=("baseline_peer_balls", "median"),
        median_baseline_selected_level=("baseline_selected_level", "median"),
        baseline_source_count=("baseline_pool_source", "nunique"),
        baseline_pool_source=("baseline_pool_source", lambda values: values.mode(dropna=True).iloc[0] if values.notna().any() else "no baseline"),
        baseline_economy=("baseline_economy", "median"),
        baseline_wickets_per_over=("baseline_wickets_per_over", "median"),
        baseline_dot_rate=("baseline_dot_rate", "median"),
        baseline_boundary_rate=("baseline_boundary_rate", "median"),
        baseline_extras_per_over=("baseline_extras_per_over", "median"),
        baseline_economy_approx_se=("baseline_economy_approx_se", "median"),
        baseline_wickets_per_over_approx_se=("baseline_wickets_per_over_approx_se", "median"),
        baseline_dot_rate_approx_se=("baseline_dot_rate_approx_se", "median"),
        baseline_boundary_rate_approx_se=("baseline_boundary_rate_approx_se", "median"),
        baseline_extras_per_over_approx_se=("baseline_extras_per_over_approx_se", "median"),
    )
    return frame, audit
