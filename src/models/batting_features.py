"""Delivery-level batting features with validated one-to-one joins."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.analysis.exploratory_analysis import competition_group
from src.data.cricket_calculations import calculate_delivery_metrics
from src.data.preprocessing import build_delivery_table

PRESSURE_STATE_COLUMNS = (
    "runs_scored_before", "wickets_lost_before", "wickets_remaining",
    "legal_balls_bowled_known", "legal_balls_bowled_possible",
    "legal_balls_remaining_min", "legal_balls_remaining_max", "legal_balls_remaining",
    "completed_overs_lower", "completed_overs_upper", "balls_into_over_lower", "balls_into_over_upper",
    "current_run_rate_lower", "current_run_rate_upper", "current_run_rate",
    "chase_target", "runs_required", "runs_to_defend", "runs_required_per_remaining_wicket",
    "required_run_rate_lower", "required_run_rate_upper", "required_run_rate",
    "prior_legality_uncertain_deliveries", "state_reliability", "is_super_over",
    "is_chasing", "innings_phase", "match_situation", "target_source", "limit_source",
)
PRESSURE_CONTEXT_COLUMNS = (
    "historical_boundary_event_rate_prior", "resource_depletion_score",
    "resource_depletion_lower", "resource_depletion_upper", "innings_progress_fraction",
    "wicket_loss_fraction", "chase_rate_gap", "chase_rate_gap_percentile",
    "remaining_target_fraction", "batting_demand_component",
)
PRESSURE_SCORE_COLUMNS = (
    "batting_pressure_weighted", "batting_pressure_nonlinear",
    "batting_pressure_weighted_lower", "batting_pressure_weighted_upper",
    "batting_pressure_nonlinear_lower", "batting_pressure_nonlinear_upper",
    "batting_high_weighted_context_p90", "batting_high_nonlinear_context_p90",
    "batting_pressure_weighted_context_percentile",
    "batting_pressure_nonlinear_context_percentile",
)

DISMISSAL_TYPES = {
    "bowled", "caught", "caught and bowled", "lbw", "stump", "stumped",
    "hit wicket", "runout", "run out", "retire out", "retired out",
    "obstructing the field", "obstracting the field", "timed out",
    "hit the ball twice", "handled the ball",
}
NON_DISMISSAL_TYPES = {"retired hurt", "retired", "absent hurt", "absent injured"}
NON_LEGAL_EXTRAS = {"wide", "overstep"}
KNOWN_FACE_EXTRAS = {"overstep", "byes", "legbyes"}


def _ensure_unique_dimension(
    frame: pd.DataFrame, keys: list[str], label: str
) -> pd.DataFrame:
    duplicate_rows = int(frame.duplicated(keys, keep=False).sum())
    if duplicate_rows:
        raise ValueError(f"{label} dimension has {duplicate_rows} rows with duplicate keys {keys}")
    return frame


def _attach_competition_group(tables: dict[str, pd.DataFrame], deliveries: pd.DataFrame) -> pd.DataFrame:
    competitions = tables["competitions"][["comp_id", "comp_name", "match_format", "season"]].drop_duplicates("comp_id").copy()
    competitions["competition_group"] = competition_group(competitions)
    if deliveries.columns.duplicated().any():
        deliveries = deliveries.loc[:, ~deliveries.columns.duplicated()].copy()
    deliveries = deliveries.drop(columns=["comp_name", "match_format", "season", "competition_group"], errors="ignore")
    return deliveries.merge(
        competitions,
        on="comp_id",
        how="left",
        validate="many_to_one",
        indicator="_competition_join",
    )


def _attach_player_and_team_names(
    tables: dict[str, pd.DataFrame], frame: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    mapping = tables["competition_team_player_mapping"].copy()
    mapping["comp_id"] = pd.to_numeric(mapping["comp_id"], errors="coerce")
    mapping["team_id"] = pd.to_numeric(mapping["team_id"], errors="coerce")
    mapping["player_id"] = pd.to_numeric(mapping["player_id"], errors="coerce")
    mapping = _ensure_unique_dimension(mapping, ["comp_id", "team_id", "player_id"], "competition-team-player")
    player_dim = mapping[["comp_id", "team_id", "player_id", "player_name", "team_name"]]

    before = len(frame)
    frame = frame.merge(
        player_dim.rename(
            columns={
                "team_id": "batting_team_id",
                "player_id": "striker_id",
                "player_name": "mapped_player_name",
                "team_name": "mapped_batting_team_name",
            }
        ),
        on=["comp_id", "batting_team_id", "striker_id"],
        how="left",
        validate="many_to_one",
        indicator="_player_join",
    )
    if len(frame) != before:
        raise ValueError("Player mapping join changed the delivery row count")

    team_dim = tables["teams"][["comp_id", "team_id", "team_name"]].copy()
    team_dim = team_dim.groupby(["comp_id", "team_id"], as_index=False, dropna=False).agg(
        team_name=("team_name", "first"),
        distinct_team_names=("team_name", "nunique"),
    )
    team_dim = _ensure_unique_dimension(team_dim, ["comp_id", "team_id"], "team")
    batting_dim = team_dim.rename(
        columns={
            "team_id": "batting_team_id",
            "team_name": "batting_team_name_catalog",
            "distinct_team_names": "batting_team_name_conflicts",
        }
    )
    bowling_dim = team_dim.rename(
        columns={
            "team_id": "bowling_team_id",
            "team_name": "opponent_team_name",
            "distinct_team_names": "opponent_team_name_conflicts",
        }
    )
    frame = frame.merge(
        batting_dim, on=["comp_id", "batting_team_id"], how="left", validate="many_to_one"
    ).merge(
        bowling_dim, on=["comp_id", "bowling_team_id"], how="left", validate="many_to_one"
    )
    frame["player_name"] = frame["mapped_player_name"].fillna(frame["striker_name"])
    frame["batting_team_name"] = frame["batting_team_name_catalog"].fillna(frame["mapped_batting_team_name"])
    frame["player_identifier_join_status"] = np.where(
        frame["striker_id"].notna() & frame["_player_join"].eq("both"),
        "matched by competition, team, and player ID",
        np.where(frame["striker_id"].notna(), "unmatched mapping; source name retained", "missing stable player ID"),
    )
    diagnostics = {
        "unmatched_player_mapping_rows": int(frame["_player_join"].ne("both").sum()),
        "missing_player_id_rows": int(frame["striker_id"].isna().sum()),
        "missing_batting_team_name_rows": int(frame["batting_team_name"].isna().sum()),
        "missing_opponent_team_name_rows": int(frame["opponent_team_name"].isna().sum()),
        "batting_team_catalog_name_conflicts": int(frame["batting_team_name_conflicts"].fillna(0).gt(1).sum()),
        "opponent_team_catalog_name_conflicts": int(frame["opponent_team_name_conflicts"].fillna(0).gt(1).sum()),
    }
    frame = frame.drop(
        columns=[
            "_player_join", "mapped_player_name", "mapped_batting_team_name",
            "batting_team_name_catalog", "batting_team_name_conflicts", "opponent_team_name_conflicts",
        ],
        errors="ignore",
    )
    return frame, diagnostics


def derive_batting_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply scorekeeping conventions and explicit uncertainty flags to deliveries."""
    frame = frame.copy()
    for id_column in ("striker_id", "dismissed_player_id"):
        if id_column in frame:
            frame[id_column] = pd.to_numeric(frame[id_column], errors="coerce")
    extras_type = frame["extras_type"].astype("string").str.strip().str.casefold()
    extras_one = pd.to_numeric(frame["extras1_runs"], errors="coerce").fillna(0)
    extras_two = pd.to_numeric(frame["extras_2_runs"], errors="coerce").fillna(0)
    extras_present = extras_one.ne(0) | extras_two.ne(0)
    is_wide = extras_type.eq("wide").fillna(False)
    no_extra_label = extras_type.isna() | extras_type.isin({"none", "no extras", ""})
    face_known = extras_type.isin(KNOWN_FACE_EXTRAS) | (no_extra_label & ~extras_present)
    frame["balls_faced_known"] = face_known & ~is_wide
    frame["balls_faced_uncertain"] = ~face_known & ~is_wide
    frame["batter_runs"] = pd.to_numeric(frame["runs"], errors="coerce").fillna(0)
    frame["boundary_event"] = frame["is_four"].fillna(False) | frame["is_six"].fillna(False)
    frame["fours_known"] = frame["is_four"].fillna(False).astype(bool) & frame["balls_faced_known"]
    frame["sixes_known"] = frame["is_six"].fillna(False).astype(bool) & frame["balls_faced_known"]
    frame["boundary_count_known"] = frame["fours_known"].astype("int8") + frame["sixes_known"].astype("int8")
    frame["boundary_runs_known"] = 4 * frame["fours_known"].astype("int8") + 6 * frame["sixes_known"].astype("int8")
    frame["dot_ball_batter_run_known"] = frame["balls_faced_known"] & frame["batter_runs"].eq(0)
    frame["zero_total_run_ball_known"] = frame["balls_faced_known"] & frame["total_runs_calculated"].eq(0)
    frame["boundary_on_uncertain_faced_delivery"] = frame["boundary_event"].fillna(False) & frame["balls_faced_uncertain"]
    dismissal_type = frame["dismissal_type"].astype("string").str.strip().str.casefold()
    is_wicket = frame["is_wicket"].fillna(False).astype(bool)
    frame["valid_dismissal_type"] = dismissal_type.isin(DISMISSAL_TYPES)
    frame["non_dismissal_event_type"] = dismissal_type.isin(NON_DISMISSAL_TYPES)
    frame["dismissal_event_valid"] = is_wicket & frame["valid_dismissal_type"]
    frame["dismissal_attribution_uncertain"] = is_wicket & (
        dismissal_type.isna() | (~frame["valid_dismissal_type"] & ~frame["non_dismissal_event_type"])
        | (frame["valid_dismissal_type"] & frame["dismissed_player_id"].isna()))
    frame["dismissal_for_striker"] = frame["dismissal_event_valid"] & frame["dismissed_player_id"].eq(frame["striker_id"]).fillna(False)
    frame["retired_hurt_not_counted_as_dismissal"] = is_wicket & frame["non_dismissal_event_type"]
    frame["legal_status_uncertain"] = frame["legal_delivery_calculated"].isna()
    frame["bowler_extra_attribution_uncertain"] = frame["bowler_runs_attribution_unknown"].fillna(False)
    frame["pressure_state_reliability"] = frame["state_reliability"].fillna("missing pressure state")
    frame["pressure_score_missing"] = frame.get("batting_pressure_weighted", pd.Series(pd.NA,index=frame.index)).isna() | frame.get("batting_pressure_nonlinear", pd.Series(pd.NA,index=frame.index)).isna()
    frame["pressure_features_missing"] = frame.get("resource_depletion_score", pd.Series(pd.NA,index=frame.index)).isna() | frame.get("batting_demand_component", pd.Series(pd.NA,index=frame.index)).isna()
    frame["innings_context"] = np.where(frame["is_super_over"].fillna(False),
        np.where(frame["is_chasing"].fillna(False), "tie-break chase", "tie-break first"),
        np.where(frame["is_chasing"].fillna(False), "chasing", "first innings"))
    frame["season"] = frame["season"].astype("string")
    return frame


def build_batting_delivery_features(
    tables: dict[str, pd.DataFrame],
    pressure_states: pd.DataFrame,
    pressure_features: pd.DataFrame,
    pressure_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join source deliveries, pre-delivery states, and pressure outputs safely."""
    source = build_delivery_table(tables)
    source["_source_row_id"] = np.arange(len(source), dtype="int64")
    frame = calculate_delivery_metrics(source)
    # Normalize ID columns once so dismissed-player and striker aggregations use the same key type.
    for id_column in ("comp_id", "batting_team_id", "bowling_team_id", "striker_id", "dismissed_player_id"):
        if id_column in frame:
            frame[id_column] = pd.to_numeric(frame[id_column], errors="coerce")
    frame = _attach_competition_group(tables, frame)

    diagnostics: dict[str, Any] = {
        "source_delivery_rows": int(len(frame)),
        "duplicate_source_delivery_keys": int(frame.duplicated(["match_id", "innings", "innings_ball_num"]).sum()),
        "pressure_join_rows": {},
    }
    if diagnostics["duplicate_source_delivery_keys"]:
        raise ValueError("Source delivery keys are not unique by match, innings, and innings_ball_num")

    join_keys = ["match_id", "innings", "innings_ball_num"]
    frames = (
        ("state", pressure_states, PRESSURE_STATE_COLUMNS),
        ("feature", pressure_features, PRESSURE_CONTEXT_COLUMNS),
        ("score", pressure_scores, PRESSURE_SCORE_COLUMNS),
    )
    for label, right, columns in frames:
        selected = right[join_keys + [column for column in columns if column in right.columns]].copy()
        duplicate_keys = int(selected.duplicated(join_keys).sum())
        if duplicate_keys:
            raise ValueError(f"Phase 4 {label} table contains {duplicate_keys} duplicate delivery keys")
        row_count = len(frame)
        frame = frame.merge(
            selected,
            on=join_keys,
            how="left",
            validate="one_to_one",
            indicator=f"_{label}_join",
            suffixes=("", f"_{label}"),
        )
        matched = frame[f"_{label}_join"].eq("both")
        diagnostics["pressure_join_rows"][label] = {
            "matched": int(matched.sum()),
            "unmatched": int((~matched).sum()),
            "right_duplicate_keys": duplicate_keys,
        }
        if len(frame) != row_count:
            raise ValueError(f"Phase 4 {label} join changed delivery row count")
        frame = frame.drop(columns=[f"_{label}_join"])

    frame, name_diagnostics = _attach_player_and_team_names(tables, frame)
    diagnostics.update(name_diagnostics)
    diagnostics["delivery_rows_after_joins"] = int(len(frame))

    frame = derive_batting_outcomes(frame)

    diagnostics.update(
        {
            "known_faced_rows": int(frame["balls_faced_known"].sum()),
            "uncertain_faced_rows": int(frame["balls_faced_uncertain"].sum()),
            "wide_rows_excluded_from_balls_faced": int(frame["extras_type"].astype("string").str.casefold().eq("wide").fillna(False).sum()),
            "legal_status_uncertain_rows": int(frame["legal_status_uncertain"].sum()),
            "bowler_extra_attribution_uncertain_rows": int(frame["bowler_extra_attribution_uncertain"].sum()),
            "valid_batter_dismissal_events": int(frame["dismissal_event_valid"].sum()),
            "dismissal_attribution_uncertain_events": int(frame["dismissal_attribution_uncertain"].sum()),
            "retired_hurt_or_non_dismissal_rows_not_counted": int(frame["retired_hurt_not_counted_as_dismissal"].sum()),
            "unmatched_pressure_state_rows": diagnostics["pressure_join_rows"]["state"]["unmatched"],
            "unmatched_pressure_feature_rows": diagnostics["pressure_join_rows"]["feature"]["unmatched"],
            "unmatched_pressure_score_rows": diagnostics["pressure_join_rows"]["score"]["unmatched"],
        }
    )
    frame = frame.sort_values("_source_row_id", kind="stable").reset_index(drop=True)
    return frame, diagnostics
