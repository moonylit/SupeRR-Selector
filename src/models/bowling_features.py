"""Delivery-level bowling outcomes joined to the Phase 4 pre-delivery state."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.analysis.exploratory_analysis import competition_group
from src.data.cricket_calculations import calculate_delivery_metrics
from src.data.preprocessing import build_delivery_table
from src.models.batting_features import build_batting_delivery_features

DELIVERY_KEYS = ["match_id", "innings", "innings_ball_num"]
PRESSURE_CONTEXT_COLUMNS = [
    "runs_scored_before", "wickets_lost_before", "wickets_remaining",
    "legal_balls_bowled_known", "legal_balls_bowled_possible",
    "legal_balls_remaining_min", "legal_balls_remaining_max", "legal_balls_remaining",
    "current_run_rate", "required_run_rate", "runs_required", "runs_to_defend",
    "runs_required_per_remaining_wicket", "prior_legality_uncertain_deliveries",
    "state_reliability", "is_chasing", "is_super_over", "innings_phase",
    "target_source", "limit_source", "historical_boundary_event_rate_prior",
    "resource_depletion_score", "resource_depletion_lower", "resource_depletion_upper",
    "bowling_threat_component", "bowling_resource_component",
]
PRESSURE_SCORE_COLUMNS = [
    "bowling_pressure_weighted", "bowling_pressure_nonlinear",
    "bowling_pressure_weighted_lower", "bowling_pressure_weighted_upper",
    "bowling_pressure_nonlinear_lower", "bowling_pressure_nonlinear_upper",
    "bowling_pressure_weighted_context_percentile",
    "bowling_pressure_nonlinear_context_percentile",
    "bowling_threat_component", "bowling_resource_component",
    "bowling_high_weighted_context_p90", "bowling_high_weighted_context_p95",
    "bowling_high_nonlinear_context_p90", "bowling_high_nonlinear_context_p95",
]


def _unique(frame: pd.DataFrame, keys: list[str], label: str) -> pd.DataFrame:
    if frame.duplicated(keys).any():
        raise ValueError(f"{label} contains duplicate keys {keys}")
    return frame


def _attach_bowler_names(tables: dict[str, pd.DataFrame], frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    mapping = tables["competition_team_player_mapping"].copy()
    for column in ("comp_id", "team_id", "player_id"):
        mapping[column] = pd.to_numeric(mapping[column], errors="coerce")
    dimensions = mapping[["comp_id", "team_id", "player_id", "player_name"]].drop_duplicates()
    if dimensions.duplicated(["comp_id", "team_id", "player_id"]).any():
        raise ValueError("Bowler mapping has conflicting names for a competition/team/player ID")
    dimensions = dimensions.rename(columns={
        "team_id": "bowling_team_id", "player_id": "bowler_id", "player_name": "mapped_bowler_name"
    })
    before = len(frame)
    result = frame.merge(
        dimensions, on=["comp_id", "bowling_team_id", "bowler_id"], how="left",
        validate="many_to_one", indicator="_bowler_mapping_join",
    )
    if len(result) != before:
        raise ValueError("Bowler identity join changed the delivery row count")
    result["bowler_name"] = result["mapped_bowler_name"].fillna(result.get("bowler_name"))
    result["bowler_mapping_status"] = np.select(
        [result["bowler_id"].isna(), result["_bowler_mapping_join"].eq("both")],
        ["missing stable bowler ID", "matched by competition, bowling team, and player ID"],
        default="unmatched mapping; source name retained",
    )
    diagnostics = {
        "missing_bowler_id_rows": int(result["bowler_id"].isna().sum()),
        "unmatched_bowler_mapping_rows": int(result["_bowler_mapping_join"].eq("left_only").sum()),
        "bowler_mapping_unique_keys": int(len(dimensions)),
    }
    return result.drop(columns=["mapped_bowler_name", "_bowler_mapping_join"]), diagnostics


def build_bowling_delivery_features(
    tables: dict[str, pd.DataFrame],
    pressure_states: pd.DataFrame,
    pressure_features: pd.DataFrame,
    pressure_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Preserve source delivery IDs while adding bowling outcomes and P4 context."""
    # Reuse Phase 5's tested source/context joins, then add the distinct
    # bowling-side index. Batting outcomes are retained for audit only.
    frame, diagnostics = build_batting_delivery_features(
        tables, pressure_states, pressure_features, pressure_scores
    )
    keep_columns = {
        *DELIVERY_KEYS,
        "_source_row_id", "comp_id", "competition_group", "season", "match_date",
        "batting_team_id", "bowling_team_id", "bowler_id", "bowler_name",
        "innings_phase", "innings_context", "is_chasing", "is_super_over",
        "striker_id", "legal_delivery_calculated", "bowler_runs_conceded_known",
        "bowler_runs_attribution_unknown", "extras_calculated", "total_runs_calculated",
        "bowler_wicket_calculated", "wicket_attribution_unknown", "is_four", "is_six",
        "extras_type", "runs", "is_wicket", "bowler_pressure_weighted",
        "bowling_pressure_nonlinear", "bowling_pressure_weighted_lower",
        "bowling_pressure_weighted_upper", "bowling_pressure_nonlinear_lower",
        "bowling_pressure_nonlinear_upper", "bowling_pressure_weighted_context_percentile",
        "bowling_pressure_nonlinear_context_percentile", "bowling_high_weighted_context_p90",
        "bowling_high_weighted_context_p95", "bowling_high_nonlinear_context_p90",
        "bowling_high_nonlinear_context_p95", *PRESSURE_CONTEXT_COLUMNS,
    }
    frame = frame.loc[:, [column for column in frame.columns if column in keep_columns]].copy()
    score_columns = [column for column in PRESSURE_SCORE_COLUMNS if column in pressure_scores]
    score_table = pressure_scores[DELIVERY_KEYS + score_columns].copy()
    _unique(score_table, DELIVERY_KEYS, "Phase 4 pressure scores")
    before = len(frame)
    frame = frame.merge(
        score_table, on=DELIVERY_KEYS, how="left", validate="one_to_one",
        indicator="_bowling_pressure_join", suffixes=("", "_p4"),
    )
    diagnostics["bowling_pressure_rows_matched"] = int(frame["_bowling_pressure_join"].eq("both").sum())
    diagnostics["bowling_pressure_rows_unmatched"] = int(frame["_bowling_pressure_join"].ne("both").sum())
    frame = frame.drop(columns="_bowling_pressure_join")
    if len(frame) != before:
        raise ValueError("Bowling pressure join changed the source delivery row count")

    for column in ("comp_id", "bowler_id", "bowling_team_id"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame, mapping_diagnostics = _attach_bowler_names(tables, frame)
    diagnostics.update(mapping_diagnostics)

    extras_type = frame["extras_type"].astype("string").str.strip().str.casefold()
    extras = pd.to_numeric(frame["extras_calculated"], errors="coerce").fillna(0).clip(lower=0)
    frame["bowler_runs_conceded_lower"] = pd.to_numeric(
        frame["bowler_runs_conceded_known"], errors="coerce"
    ).fillna(0).clip(lower=0)
    uncertain_extras = frame["bowler_runs_attribution_unknown"].fillna(False).astype(bool)
    frame["bowler_runs_attribution_uncertain_upper"] = extras.where(uncertain_extras, 0)
    frame["bowler_runs_conceded_upper"] = (
        frame["bowler_runs_conceded_lower"] + frame["bowler_runs_attribution_uncertain_upper"]
    )

    legal = frame["legal_delivery_calculated"]
    frame["legal_ball_known"] = legal.eq(True).fillna(False)
    frame["legal_ball_possible"] = legal.eq(True).fillna(False) | legal.isna()
    frame["legal_ball_uncertain"] = legal.isna()
    frame["bowler_dot_legal"] = (
        frame["legal_ball_known"] & frame["bowler_runs_conceded_upper"].eq(0)
    )
    frame["team_dot_legal"] = (
        frame["legal_ball_known"] & pd.to_numeric(frame["total_runs_calculated"], errors="coerce").eq(0)
    )
    frame["boundary_delivery"] = (
        frame["is_four"].fillna(False).astype(bool) | frame["is_six"].fillna(False).astype(bool)
    )
    frame["four_conceded"] = frame["is_four"].fillna(False).astype(bool)
    frame["six_conceded"] = frame["is_six"].fillna(False).astype(bool)
    frame["wide_delivery"] = extras_type.eq("wide").fillna(False)
    frame["no_ball_delivery"] = extras_type.eq("overstep").fillna(False)
    frame["bowler_extra_runs_known"] = (
        extras.where(frame["wide_delivery"] | frame["no_ball_delivery"], 0)
    )
    frame["bowler_pressure_score"] = pd.to_numeric(frame["bowling_pressure_weighted"], errors="coerce")
    frame["bowler_pressure_alternate_score"] = pd.to_numeric(
        frame["bowling_pressure_nonlinear"], errors="coerce"
    )
    frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")
    frame["innings_context"] = np.where(
        frame["is_super_over"].fillna(False),
        np.where(frame["is_chasing"].fillna(False), "tie-break chase", "tie-break first"),
        np.where(frame["is_chasing"].fillna(False), "chasing", "first innings"),
    )
    diagnostics.update({
        "source_delivery_rows": int(len(frame)),
        "unique_delivery_keys": int(frame[DELIVERY_KEYS].drop_duplicates().shape[0]),
        "legal_ball_known_rows": int(frame["legal_ball_known"].sum()),
        "legal_ball_uncertain_rows": int(frame["legal_ball_uncertain"].sum()),
        "uncertain_bowler_extra_rows": int(uncertain_extras.sum()),
        "bowler_attributable_wicket_rows": int(frame["bowler_wicket_calculated"].sum()),
        "unknown_wicket_attribution_rows": int(frame["wicket_attribution_unknown"].sum()),
        "bowling_pressure_missing_rows": int(frame["bowler_pressure_score"].isna().sum()),
    })
    if diagnostics["unique_delivery_keys"] != len(frame):
        raise ValueError("Source delivery keys are not unique by match, innings, and innings ball")
    return frame.sort_values("_source_row_id", kind="stable").reset_index(drop=True), diagnostics


def aggregate_bowling_counts(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """Aggregate cricket counts first; derive rates only after aggregation."""
    work = frame.copy()
    if "legal_ball_known" not in work:
        legal = work["legal_delivery_calculated"]
        work["legal_ball_known"] = legal.eq(True).fillna(False)
        work["legal_ball_possible"] = legal.eq(True).fillna(False) | legal.isna()
    if "bowler_runs_conceded_lower" not in work:
        work["bowler_runs_conceded_lower"] = work["bowler_runs_conceded_known"]
        work["bowler_runs_conceded_upper"] = work["bowler_runs_conceded_known"] + (
            work["extras_calculated"].where(work["bowler_runs_attribution_unknown"], 0)
        )
    if "boundary_runs_conceded" not in work:
        work["boundary_runs_conceded"] = (
            4 * work["four_conceded"].fillna(False).astype(int)
            + 6 * work["six_conceded"].fillna(False).astype(int)
        )
    work["_match_innings_key"] = (
        work["match_id"].astype("string") + ":" + work["innings"].astype("string")
    )
    flags = {
        "legal_balls_known": "legal_ball_known",
        "legal_balls_possible": "legal_ball_possible",
        "runs_conceded_lower": "bowler_runs_conceded_lower",
        "runs_conceded_upper": "bowler_runs_conceded_upper",
        "boundary_runs_conceded": "boundary_runs_conceded",
        "bowler_wickets": "bowler_wicket_calculated",
        "legal_bowler_dots": "bowler_dot_legal",
        "legal_team_dots": "team_dot_legal",
        "fours_conceded": "four_conceded",
        "sixes_conceded": "six_conceded",
        "boundary_deliveries": "boundary_delivery",
        "wide_deliveries": "wide_delivery",
        "no_ball_deliveries": "no_ball_delivery",
        "bowler_extra_runs_known": "bowler_extra_runs_known",
        "uncertain_extra_runs_upper": "bowler_runs_attribution_uncertain_upper",
        "legal_ball_uncertain_rows": "legal_ball_uncertain",
        "uncertain_extra_rows": "bowler_runs_attribution_unknown",
        "wicket_attribution_unknown_rows": "wicket_attribution_unknown",
    }
    agg = work.groupby(groups, as_index=False, dropna=False).agg(
        delivery_rows=("match_id", "size"),
        **{name: (column, "sum") for name, column in flags.items()},
        matches=("match_id", "nunique"),
        innings=("_match_innings_key", "nunique"),
    )
    legal = agg["legal_balls_known"].replace(0, np.nan)
    possible = agg["legal_balls_possible"].replace(0, np.nan)
    agg["overs_known"] = agg["legal_balls_known"] / 6
    agg["overs_possible"] = agg["legal_balls_possible"] / 6
    agg["economy"] = 6 * agg["runs_conceded_lower"] / legal
    agg["runs_per_legal_ball"] = agg["runs_conceded_lower"] / legal
    agg["economy_lower_bound"] = 6 * agg["runs_conceded_lower"] / possible
    agg["economy_upper_bound"] = 6 * agg["runs_conceded_upper"] / legal
    agg["wickets_per_legal_over"] = 6 * agg["bowler_wickets"] / legal
    agg["wickets_per_100_legal_balls"] = 100 * agg["bowler_wickets"] / legal
    agg["bowling_strike_rate_balls_per_wicket"] = legal / agg["bowler_wickets"].replace(0, np.nan)
    agg["bowler_dot_rate"] = agg["legal_bowler_dots"] / legal
    agg["team_dot_rate"] = agg["legal_team_dots"] / legal
    agg["boundary_concession_rate"] = agg["boundary_deliveries"] / legal
    agg["four_rate"] = agg["fours_conceded"] / legal
    agg["six_rate"] = agg["sixes_conceded"] / legal
    agg["wides_per_legal_over"] = 6 * agg["wide_deliveries"] / legal
    agg["no_balls_per_legal_over"] = 6 * agg["no_ball_deliveries"] / legal
    agg["bowler_extras_per_legal_over"] = (
        6 * agg["bowler_extra_runs_known"] / legal
    )
    agg["bowler_extra_runs_per_legal_over_upper"] = 6 * (
        agg["bowler_extra_runs_known"] + agg["uncertain_extra_runs_upper"]
    ) / legal
    agg["economy_uncertainty_width"] = agg["economy_upper_bound"] - agg["economy_lower_bound"]
    return agg
