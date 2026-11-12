"""Pre-delivery ball-by-ball match-state reconstruction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.exploratory_analysis import competition_group
from src.data.cricket_calculations import calculate_delivery_metrics
from src.data.preprocessing import build_delivery_table

DEFAULT_T20_BALL_LIMIT = 120
SUPER_OVER_BALL_LIMIT = 6


def build_delivery_match_states(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return all delivery rows with state strictly from earlier rows.

    Unknown legality is carried as a lower/upper bound on balls used and
    remaining. The current delivery's legality is recorded but never used in
    the state before that delivery.
    """
    deliveries = calculate_delivery_metrics(build_delivery_table(tables))
    competitions = tables["competitions"][["comp_id", "comp_name", "match_format", "season"]].copy()
    competitions["competition_group"] = competition_group(competitions)
    deliveries = deliveries.drop(columns=["comp_name", "match_format", "season"], errors="ignore").merge(
        competitions, on="comp_id", how="left", validate="many_to_one"
    )
    deliveries["_source_row_order"] = np.arange(len(deliveries), dtype="int64")
    deliveries = deliveries.sort_values(["match_id", "innings", "innings_ball_num", "_source_row_order"], kind="stable")

    innings_totals = deliveries.groupby(["match_id", "innings"], sort=False)["total_runs_calculated"].sum()
    matches = tables["matches"].set_index("match_id")
    innings_number = pd.to_numeric(deliveries["innings"], errors="coerce").fillna(0).astype(int)
    match_ids = deliveries["match_id"]
    revised_target = match_ids.map(matches["revised_target"])
    first_total = match_ids.map(innings_totals.xs(1, level="innings") if 1 in innings_totals.index.get_level_values("innings") else pd.Series(dtype=float))
    previous_keys = pd.MultiIndex.from_arrays([match_ids, innings_number - 1], names=["match_id", "innings"])
    previous_total = innings_totals.reindex(previous_keys).to_numpy(dtype=float)
    is_chasing = innings_number.ge(2) & innings_number.mod(2).eq(0)
    second_innings_target = revised_target.where(revised_target.notna(), first_total + 1)
    chase_target = pd.Series(previous_total + 1, index=deliveries.index)
    chase_target.loc[innings_number.eq(2)] = second_innings_target.loc[innings_number.eq(2)]
    deliveries["chase_target"] = chase_target.where(is_chasing)
    deliveries["target_source"] = np.select(
        [
            ~is_chasing,
            innings_number.eq(2) & revised_target.notna(),
            deliveries["chase_target"].isna() & is_chasing,
        ],
        ["not applicable", "recorded revised_target", "unknown previous innings total"],
        default="previous innings total + 1",
    )
    deliveries["is_chasing"] = is_chasing
    deliveries["is_super_over"] = deliveries["innings"].ge(3)
    first_limits = pd.to_numeric(match_ids.map(matches["first_inn_revised_overs"]), errors="coerce")
    second_limits = pd.to_numeric(match_ids.map(matches["second_inn_revised_overs"]), errors="coerce")
    regular_limit = first_limits.where(innings_number.eq(1), second_limits)
    deliveries["innings_ball_limit"] = np.where(
        deliveries["is_super_over"], SUPER_OVER_BALL_LIMIT,
        regular_limit.fillna(DEFAULT_T20_BALL_LIMIT),
    ).astype(float)
    deliveries["limit_source"] = np.select(
        [
            deliveries["is_super_over"],
            regular_limit.notna(),
        ],
        [
            "super-over default 6 legal balls",
            "recorded revised ball limit; field name is ambiguous",
        ],
        default="default 120 legal balls",
    )
    deliveries["boundary_event"] = deliveries["is_four"].fillna(False) | deliveries["is_six"].fillna(False)

    keys = [deliveries["match_id"], deliveries["innings"]]
    runs = deliveries["total_runs_calculated"].fillna(0)
    wickets = deliveries["is_wicket"].fillna(False).astype("int8")
    legal = deliveries["legal_delivery_calculated"].eq(True).fillna(False).astype("int8")
    unknown = deliveries["legal_delivery_calculated"].isna().astype("int8")
    deliveries["runs_scored_before"] = runs.groupby(keys, sort=False).cumsum() - runs
    deliveries["wickets_lost_before"] = wickets.groupby(keys, sort=False).cumsum() - wickets
    deliveries["legal_balls_bowled_known"] = legal.groupby(keys, sort=False).cumsum() - legal
    deliveries["prior_legality_uncertain_deliveries"] = unknown.groupby(keys, sort=False).cumsum() - unknown
    deliveries["legal_balls_bowled_possible"] = (
        deliveries["legal_balls_bowled_known"] + deliveries["prior_legality_uncertain_deliveries"]
    )
    deliveries["completed_overs_lower"] = deliveries["legal_balls_bowled_known"] // 6
    deliveries["completed_overs_upper"] = deliveries["legal_balls_bowled_possible"] // 6
    deliveries["balls_into_over_lower"] = deliveries["legal_balls_bowled_known"] % 6
    deliveries["balls_into_over_upper"] = deliveries["legal_balls_bowled_possible"] % 6
    deliveries["legal_balls_remaining_min"] = (
        deliveries["innings_ball_limit"] - deliveries["legal_balls_bowled_possible"]
    ).clip(lower=0)
    deliveries["legal_balls_remaining_max"] = (
        deliveries["innings_ball_limit"] - deliveries["legal_balls_bowled_known"]
    ).clip(lower=0)
    deliveries["legal_balls_remaining"] = (
        deliveries["legal_balls_remaining_min"] + deliveries["legal_balls_remaining_max"]
    ) / 2
    deliveries["wickets_remaining"] = (10 - deliveries["wickets_lost_before"]).clip(lower=0)
    deliveries["current_run_rate_lower"] = (
        deliveries["runs_scored_before"] * 6
        / (deliveries["legal_balls_bowled_possible"].replace(0, np.nan))
    )
    deliveries["current_run_rate_upper"] = (
        deliveries["runs_scored_before"] * 6
        / (deliveries["legal_balls_bowled_known"].replace(0, np.nan))
    )
    deliveries.loc[
        deliveries["legal_balls_bowled_known"].eq(0) & deliveries["runs_scored_before"].gt(0),
        "current_run_rate_upper",
    ] = np.inf
    deliveries["current_run_rate"] = (
        deliveries["current_run_rate_lower"] + deliveries["current_run_rate_upper"]
    ) / 2
    deliveries["runs_required"] = (
        deliveries["chase_target"] - deliveries["runs_scored_before"]
    ).clip(lower=0).where(deliveries["is_chasing"])
    deliveries["runs_to_defend"] = (deliveries["runs_required"] - 1).clip(lower=0)
    deliveries["runs_required_per_remaining_wicket"] = (
        deliveries["runs_required"] / deliveries["wickets_remaining"].replace(0, np.nan)
    )
    deliveries["required_run_rate_lower"] = (
        deliveries["runs_required"] * 6 / deliveries["legal_balls_remaining_max"].replace(0, np.nan)
    )
    deliveries["required_run_rate_upper"] = (
        deliveries["runs_required"] * 6 / deliveries["legal_balls_remaining_min"].replace(0, np.nan)
    )
    # A positive requirement with no possible legal balls has infinite required
    # rate (unattainable); zero requirement remains zero even after the target.
    for column, remaining_column in (
        ("required_run_rate_lower", "legal_balls_remaining_max"),
        ("required_run_rate_upper", "legal_balls_remaining_min"),
    ):
        no_balls_left = deliveries[remaining_column].eq(0)
        deliveries.loc[no_balls_left & deliveries["runs_required"].gt(0), column] = np.inf
        deliveries.loc[no_balls_left & deliveries["runs_required"].eq(0), column] = 0.0
    deliveries["required_run_rate"] = (
        deliveries["required_run_rate_lower"] + deliveries["required_run_rate_upper"]
    ) / 2
    deliveries["known_balls_used_over_bin"] = (deliveries["legal_balls_bowled_known"] // 6).astype(int)
    known_position = deliveries["legal_balls_bowled_known"]
    possible_position = deliveries["legal_balls_bowled_possible"]
    phase_min = pd.cut(known_position, [-1, 35, 89, np.inf], labels=["Powerplay (overs 1-6)", "Middle (overs 7-15)", "Death (overs 16+)"])
    phase_max = pd.cut(possible_position, [-1, 35, 89, np.inf], labels=["Powerplay (overs 1-6)", "Middle (overs 7-15)", "Death (overs 16+)"])
    deliveries["innings_phase"] = phase_min.astype("string").where(phase_min.eq(phase_max), "Uncertain phase")
    deliveries.loc[deliveries["is_super_over"], "innings_phase"] = "Tie-break / super-over innings"
    deliveries["legal_status_unknown_at_delivery"] = deliveries["legal_delivery_calculated"].isna()
    deliveries["state_reliability"] = np.select(
        [
            deliveries["prior_legality_uncertain_deliveries"].gt(0),
            deliveries["is_chasing"] & deliveries["chase_target"].isna(),
            deliveries["limit_source"].str.startswith("default"),
        ],
        [
            "uncertain prior legal-ball count",
            "chase target unknown",
            "assumed default innings limit",
        ],
        default="known pre-delivery state",
    )
    deliveries["match_situation"] = np.select(
        [
            ~deliveries["is_chasing"],
            deliveries["chase_target"].isna(),
            deliveries["runs_required"].eq(0),
            deliveries["wickets_remaining"].eq(0),
            deliveries["legal_balls_remaining_max"].eq(0),
        ],
        ["first innings", "target unknown", "target already reached", "all out", "innings limit reached"],
        default="active chase",
    )
    return deliveries.drop(columns=["_source_row_order"]).reset_index(drop=True)
