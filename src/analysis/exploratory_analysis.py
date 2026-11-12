"""Descriptive IPL, T20I, and other-T20 exploratory analytics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.cricket_calculations import calculate_delivery_metrics
from src.data.loader import load_data_pack
from src.data.preprocessing import build_delivery_table

LOGGER = logging.getLogger(__name__)

POWERPLAY_END = 36
MIDDLE_END = 90
DEFAULT_T20_BALL_LIMIT = 120
MIN_BATTER_BALLS_FOR_COMPARISON = 60
MIN_BOWLER_BALLS_FOR_COMPARISON = 24
MIN_MATCHES_FOR_CONSISTENCY = 5
MIN_BOWLER_SPELL_BALLS_FOR_CONSISTENCY = 12


def competition_group(frame: pd.DataFrame) -> pd.Series:
    """Classify catalog competitions without treating all T20 events as IPL."""
    is_ipl = frame["match_format"].eq("T20") & frame["comp_name"].astype("string").str.fullmatch(
        r"IPL\s+\d{4}", case=False, na=False
    )
    return pd.Series(
        np.select(
            [frame["match_format"].eq("T20I"), is_ipl, frame["match_format"].eq("T20")],
            ["T20I", "IPL", "Other T20"],
            default="Other",
        ),
        index=frame.index,
        dtype="string",
    )


def _phase_for_legal_ball_position(position: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [position.lt(POWERPLAY_END), position.lt(MIDDLE_END)],
            ["Powerplay (overs 1-6)", "Middle (overs 7-15)"],
            default="Death (overs 16+)",
        ),
        index=position.index,
        dtype="string",
    )


def add_delivery_context(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Add inspected scoring fields, format labels, legal-ball phases, and roles."""
    deliveries = build_delivery_table(tables)
    deliveries = calculate_delivery_metrics(deliveries)
    competition_meta = tables["competitions"][
        ["comp_id", "comp_name", "match_format", "season"]
    ].copy()
    competition_meta["competition_group"] = competition_group(competition_meta)
    deliveries = deliveries.drop(
        columns=["comp_name", "match_format", "season"], errors="ignore"
    ).merge(competition_meta, on="comp_id", how="left", validate="many_to_one")
    deliveries = deliveries.sort_values(
        ["match_id", "innings", "innings_ball_num"], kind="stable"
    ).reset_index(drop=True)
    keys = [deliveries["match_id"], deliveries["innings"]]

    known_legal = deliveries["legal_delivery_calculated"].eq(True).fillna(False).astype("int8")
    unknown_legal = deliveries["legal_delivery_calculated"].isna().astype("int8")
    deliveries["known_legal_balls_before"] = known_legal.groupby(keys, sort=False).cumsum() - known_legal
    deliveries["unknown_legal_balls_before"] = unknown_legal.groupby(keys, sort=False).cumsum() - unknown_legal

    min_phase = _phase_for_legal_ball_position(deliveries["known_legal_balls_before"])
    max_phase = _phase_for_legal_ball_position(
        deliveries["known_legal_balls_before"] + deliveries["unknown_legal_balls_before"]
    )
    stable_phase = min_phase.eq(max_phase)
    deliveries["innings_phase"] = min_phase.where(stable_phase, "Uncertain phase")
    deliveries.loc[deliveries["innings"].gt(2), "innings_phase"] = "Tie-break / super-over innings"
    deliveries["phase_assignment_uncertain"] = deliveries["innings_phase"].eq("Uncertain phase")

    extras_type = deliveries["extras_type"].astype("string").str.strip().str.casefold()
    extras_present = deliveries["extras1_runs"].fillna(0).ne(0) | deliveries["extras_2_runs"].fillna(0).ne(0)
    known_faced = extras_type.isin(["overstep", "byes", "legbyes"]) | (
        extras_type.isna() & ~extras_present
    )
    deliveries["batter_ball_faced_known"] = known_faced
    deliveries.loc[extras_type.eq("wide"), "batter_ball_faced_known"] = False
    is_wide = extras_type.eq("wide").fillna(False)
    deliveries["batter_ball_faced_uncertain"] = ~known_faced & ~is_wide
    deliveries["batter_dot_known"] = (
        deliveries["batter_ball_faced_known"] & deliveries["runs"].eq(0)
    )
    deliveries["known_total_run_dot"] = (
        deliveries["legal_delivery_calculated"].eq(True).fillna(False)
        & deliveries["total_runs_calculated"].eq(0)
    )
    deliveries["boundary_event"] = deliveries["is_four"].fillna(False) | deliveries["is_six"].fillna(False)
    deliveries["innings_context"] = np.select(
        [deliveries["innings"].eq(1), deliveries["innings"].eq(2)],
        ["First innings / batting first", "Second innings / chasing"],
        default="Tie-break / super-over innings",
    )
    return deliveries


def _safe_ratio(numerator: pd.Series, denominator: pd.Series, multiplier: float = 1.0) -> pd.Series:
    return (numerator * multiplier).div(denominator.where(denominator.ne(0)))


def _player_dimension(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    mapping = tables["competition_team_player_mapping"].merge(
        tables["competitions"][["comp_id", "match_format"]],
        on="comp_id", how="left", validate="many_to_one",
    )
    mapping["competition_group"] = competition_group(mapping)
    return (
        mapping.groupby(["competition_group", "player_id"], as_index=False, dropna=False)
        .agg(
            player_name=("player_name", "first"),
            teams=("team_name", lambda values: " | ".join(sorted(set(values.dropna().astype(str))))),
        )
    )


def _attach_dimension(summary: pd.DataFrame, dimension: pd.DataFrame) -> pd.DataFrame:
    return summary.merge(
        dimension, on=["competition_group", "player_id"], how="left", validate="many_to_one"
    )


def _batting_aggregate(deliveries: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    frame = deliveries.copy()
    frame["_runs"] = frame["runs"].fillna(0)
    frame["_fours"] = frame["is_four"].fillna(False).astype("int8")
    frame["_sixes"] = frame["is_six"].fillna(False).astype("int8")
    frame["_boundary_runs"] = frame["_runs"].where(frame["boundary_event"], 0)
    frame["_known_faced"] = frame["batter_ball_faced_known"].astype("int8")
    frame["_uncertain_faced"] = frame["batter_ball_faced_uncertain"].astype("int8")
    frame["_dot"] = frame["batter_dot_known"].astype("int8")
    frame["_known_boundary"] = (frame["boundary_event"] & frame["batter_ball_faced_known"]).astype("int8")
    output_group_columns = ["player_id" if column == "striker_id" else column for column in group_columns]
    summary = (
        frame.groupby(group_columns, as_index=False, dropna=False)
        .agg(
            matches_with_batting=("match_id", "nunique"),
            innings_with_batting=("innings", lambda s: frame.loc[s.index, ["match_id", "innings"]].drop_duplicates().shape[0]),
            runs=("_runs", "sum"),
            balls_faced_known=("_known_faced", "sum"),
            balls_faced_uncertain=("_uncertain_faced", "sum"),
            fours=("_fours", "sum"),
            sixes=("_sixes", "sum"),
            boundary_balls_known=("_known_boundary", "sum"),
            boundary_runs=("_boundary_runs", "sum"),
            batter_dot_balls_known=("_dot", "sum"),
        )
    ).rename(columns={"striker_id": "player_id"})
    dismissals = frame.loc[
        frame["is_wicket"].fillna(False) & frame["dismissed_player_id"].notna()
    ].copy()
    dismissals["player_id"] = dismissals["dismissed_player_id"].astype("int64")
    dismissal_counts = (
        dismissals.groupby(output_group_columns, as_index=False, dropna=False)
        .size().rename(columns={"size": "dismissals"})
    )
    summary = summary.merge(
        dismissal_counts, on=output_group_columns, how="left", validate="one_to_one"
    )
    summary["dismissals"] = summary["dismissals"].fillna(0).astype(int)
    summary["balls_faced_max"] = summary["balls_faced_known"] + summary["balls_faced_uncertain"]
    summary["batting_average"] = _safe_ratio(summary["runs"], summary["dismissals"])
    summary["strike_rate_known_balls"] = _safe_ratio(summary["runs"], summary["balls_faced_known"], 100)
    summary["strike_rate_if_all_uncertain_faced"] = _safe_ratio(summary["runs"], summary["balls_faced_max"], 100)
    summary["boundary_ball_percentage_known_balls"] = _safe_ratio(
        summary["boundary_balls_known"], summary["balls_faced_known"], 100
    )
    summary["boundary_run_percentage"] = _safe_ratio(summary["boundary_runs"], summary["runs"], 100)
    summary["dot_ball_percentage_known_balls"] = _safe_ratio(
        summary["batter_dot_balls_known"], summary["balls_faced_known"], 100
    )
    summary["dismissals_per_100_known_balls"] = _safe_ratio(
        summary["dismissals"], summary["balls_faced_known"], 100
    )
    summary["minimum_60_known_balls"] = summary["balls_faced_known"].ge(MIN_BATTER_BALLS_FOR_COMPARISON)
    summary["minimum_opportunity_rule"] = f"{MIN_BATTER_BALLS_FOR_COMPARISON} known balls faced; descriptive comparison only"
    return summary


def _bowling_aggregate(deliveries: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    frame = deliveries.copy()
    frame["_known_legal"] = frame["legal_delivery_calculated"].eq(True).fillna(False).astype("int8")
    frame["_unknown_legal"] = frame["legal_delivery_calculated"].isna().astype("int8")
    frame["_bowler_wicket"] = frame["bowler_wicket_calculated"].astype("int8")
    frame["_all_wicket"] = frame["is_wicket"].fillna(False).astype("int8")
    frame["_known_bowler_runs"] = frame["bowler_runs_conceded_known"].fillna(0)
    frame["_uncertain_extra"] = frame["bowler_runs_attribution_unknown"].astype("int8")
    frame["_dot"] = frame["known_total_run_dot"].astype("int8")
    frame["_known_status_delivery"] = frame["legal_delivery_calculated"].notna().astype("int8")
    frame["_boundary_known_status"] = (frame["boundary_event"] & frame["legal_delivery_calculated"].notna()).astype("int8")
    summary = (
        frame.groupby(group_columns, as_index=False, dropna=False)
        .agg(
            matches_bowled=("match_id", "nunique"),
            innings_bowled=("innings", lambda s: frame.loc[s.index, ["match_id", "innings"]].drop_duplicates().shape[0]),
            legal_deliveries_known=("_known_legal", "sum"),
            legal_status_unknown_deliveries=("_unknown_legal", "sum"),
            bowling_runs_conceded_known=("_known_bowler_runs", "sum"),
            bowler_attribution_unknown_deliveries=("_uncertain_extra", "sum"),
            bowler_attributable_wickets=("_bowler_wicket", "sum"),
            total_wicket_events=("_all_wicket", "sum"),
            dot_balls_known=("_dot", "sum"),
            deliveries_with_known_legality=("_known_status_delivery", "sum"),
            boundary_events_known_legality=("_boundary_known_status", "sum"),
        )
    ).rename(columns={"bowler_id": "player_id"})
    summary["overs_known"] = summary["legal_deliveries_known"] / 6
    summary["economy_known"] = _safe_ratio(
        summary["bowling_runs_conceded_known"], summary["legal_deliveries_known"], 6
    )
    summary["bowling_strike_rate_known"] = _safe_ratio(
        summary["legal_deliveries_known"], summary["bowler_attributable_wickets"]
    )
    summary["bowling_average_known"] = _safe_ratio(
        summary["bowling_runs_conceded_known"], summary["bowler_attributable_wickets"]
    )
    summary["dot_ball_percentage_known_legal"] = _safe_ratio(
        summary["dot_balls_known"], summary["legal_deliveries_known"], 100
    )
    summary["boundary_concession_percentage_known_status_deliveries"] = _safe_ratio(
        summary["boundary_events_known_legality"], summary["deliveries_with_known_legality"], 100
    )
    summary["wickets_per_innings"] = _safe_ratio(
        summary["bowler_attributable_wickets"], summary["innings_bowled"]
    )
    summary["minimum_24_known_legal_balls"] = summary["legal_deliveries_known"].ge(MIN_BOWLER_BALLS_FOR_COMPARISON)
    summary["minimum_opportunity_rule"] = f"{MIN_BOWLER_BALLS_FOR_COMPARISON} known legal balls; descriptive comparison only"
    return summary


def _context_metrics(deliveries: pd.DataFrame, context: str) -> pd.DataFrame:
    frame = deliveries.loc[deliveries["innings"].isin([1, 2])].copy()
    frame["_known_legal"] = frame["legal_delivery_calculated"].eq(True).fillna(False).astype("int8")
    frame["_unknown_legal"] = frame["legal_delivery_calculated"].isna().astype("int8")
    frame["_bowler_wickets"] = frame["bowler_wicket_calculated"].astype("int8")
    frame["_dots"] = frame["known_total_run_dot"].astype("int8")
    frame["_known_bowler_runs"] = frame["bowler_runs_conceded_known"].fillna(0)
    frame["_known_faced"] = frame["batter_ball_faced_known"].astype("int8")
    frame["_uncertain_faced"] = frame["batter_ball_faced_uncertain"].astype("int8")
    frame["_batter_dots"] = frame["batter_dot_known"].astype("int8")
    frame["_batter_boundary_known"] = (
        frame["boundary_event"] & frame["batter_ball_faced_known"]
    ).astype("int8")
    group_columns = ["competition_group", "innings"]
    summary = (
        frame.groupby(group_columns, as_index=False)
        .agg(
            deliveries=("match_id", "size"),
            matches=("match_id", "nunique"),
            batter_runs=("runs", "sum"),
            total_runs=("total_runs_calculated", "sum"),
            legal_balls_known=("_known_legal", "sum"),
            legal_status_unknown_deliveries=("_unknown_legal", "sum"),
            wickets=("is_wicket", "sum"),
            bowler_attributable_wickets=("_bowler_wickets", "sum"),
            total_run_dot_balls=("_dots", "sum"),
            bowler_runs_known=("_known_bowler_runs", "sum"),
            boundary_events=("boundary_event", "sum"),
            batter_balls_faced_known=("_known_faced", "sum"),
            batter_balls_faced_uncertain=("_uncertain_faced", "sum"),
            batter_dot_balls_known=("_batter_dots", "sum"),
            batter_boundary_balls_known=("_batter_boundary_known", "sum"),
        )
    )
    summary["innings_context"] = np.where(summary["innings"].eq(1), "First innings / defending", "Second innings / chasing")
    summary["run_rate_known_balls"] = _safe_ratio(summary["total_runs"], summary["legal_balls_known"], 6)
    summary["bowling_economy_known_balls"] = _safe_ratio(summary["bowler_runs_known"], summary["legal_balls_known"], 6)
    summary["dot_ball_percentage_known_legal"] = _safe_ratio(summary["total_run_dot_balls"], summary["legal_balls_known"], 100)
    summary["wickets_per_innings_match"] = _safe_ratio(summary["wickets"], summary["matches"])
    summary["bowler_attributable_wickets_per_match"] = _safe_ratio(
        summary["bowler_attributable_wickets"], summary["matches"]
    )
    summary["boundary_events_per_100_deliveries"] = _safe_ratio(summary["boundary_events"], summary["deliveries"], 100)
    summary["batter_strike_rate_known_balls"] = _safe_ratio(
        summary["batter_runs"], summary["batter_balls_faced_known"], 100
    )
    summary["batter_dot_ball_percentage_known_balls"] = _safe_ratio(
        summary["batter_dot_balls_known"], summary["batter_balls_faced_known"], 100
    )
    summary["batter_boundary_ball_percentage_known_balls"] = _safe_ratio(
        summary["batter_boundary_balls_known"], summary["batter_balls_faced_known"], 100
    )
    summary["analysis_scope"] = context
    return summary


def _build_innings_summary(deliveries: pd.DataFrame) -> pd.DataFrame:
    group_keys = ["match_id", "comp_id", "comp_name", "competition_group", "match_format", "season", "innings"]
    work = deliveries.copy()
    work["_known_legal"] = work["legal_delivery_calculated"].eq(True).fillna(False).astype("int8")
    work["_unknown_legal"] = work["legal_delivery_calculated"].isna().astype("int8")
    work["_wickets"] = work["is_wicket"].fillna(False).astype("int8")
    work["_bowler_wickets"] = work["bowler_wicket_calculated"].astype("int8")
    return (
        work.groupby(group_keys, as_index=False, dropna=False)
        .agg(
            deliveries=("match_id", "size"),
            batter_runs=("runs", "sum"),
            extras=("extras_calculated", "sum"),
            total_runs=("total_runs_calculated", "sum"),
            legal_balls_known=("_known_legal", "sum"),
            legal_status_unknown_deliveries=("_unknown_legal", "sum"),
            wickets=("_wickets", "sum"),
            bowler_attributable_wickets=("_bowler_wickets", "sum"),
        )
    )


def _competition_analysis(
    tables: dict[str, pd.DataFrame], innings: pd.DataFrame
) -> pd.DataFrame:
    competitions = tables["competitions"].copy()
    competitions["competition_group"] = competition_group(competitions)
    match_counts = tables["matches"].groupby("comp_id").agg(
        matches=("match_id", "nunique"),
        first_match_date=("match_date", "min"),
        last_match_date=("match_date", "max"),
    )
    team_counts = tables["teams"].groupby("comp_id").team_id.nunique().rename("teams")
    player_counts = tables["competition_team_player_mapping"].groupby("comp_id").player_id.nunique().rename("unique_players")
    all_innings = innings.groupby("comp_id").agg(
        recorded_innings=("innings", "size"),
        innings_total_runs=("total_runs", "sum"),
        average_all_innings_score=("total_runs", "mean"),
        average_all_innings_wickets=("wickets", "mean"),
        legal_balls_known=("legal_balls_known", "sum"),
        legal_status_unknown_deliveries=("legal_status_unknown_deliveries", "sum"),
        deliveries=("deliveries", "sum"),
    )
    regular = innings.loc[innings["innings"].isin([1, 2])].groupby("comp_id").agg(
        regular_innings=("innings", "size"),
        regular_innings_runs=("total_runs", "sum"),
        average_regular_innings_score=("total_runs", "mean"),
        average_regular_innings_wickets=("wickets", "mean"),
        average_regular_innings_known_legal_balls=("legal_balls_known", "mean"),
        average_regular_innings_unknown_deliveries=("legal_status_unknown_deliveries", "mean"),
        regular_legal_balls=("legal_balls_known", "sum"),
    )
    result = competitions.merge(match_counts, on="comp_id", how="left").merge(
        team_counts, on="comp_id", how="left"
    ).merge(player_counts, on="comp_id", how="left").merge(
        all_innings, on="comp_id", how="left"
    ).merge(regular, on="comp_id", how="left")
    count_columns = ["matches", "teams", "unique_players", "recorded_innings", "regular_innings", "deliveries", "legal_balls_known", "legal_status_unknown_deliveries"]
    result[count_columns] = result[count_columns].fillna(0).astype(int)
    result["average_regular_innings_run_rate_known"] = _safe_ratio(
        result["regular_innings_runs"].fillna(0), result["regular_legal_balls"].fillna(0), 6
    )
    result["average_regular_innings_run_rate_per_innings"] = _safe_ratio(
        result["average_regular_innings_score"].fillna(0),
        result["average_regular_innings_known_legal_balls"].fillna(0),
        6,
    )
    result["average_regular_innings_overs_known"] = result["average_regular_innings_known_legal_balls"] / 6
    result["analysis_note"] = "Average score/wickets use innings 1-2; all recorded innings are reported separately to keep tie-break innings visible."
    return result.sort_values(["competition_group", "season", "comp_name"]).reset_index(drop=True)


def _build_match_states(deliveries: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Reconstruct delivery states without joining final match outcomes."""
    first_totals = (
        deliveries.loc[deliveries["innings"].eq(1)]
        .groupby("match_id").total_runs_calculated.sum().rename("first_innings_score")
    )
    matches = tables["matches"].copy().set_index("match_id")
    matches["first_innings_score"] = first_totals
    matches["chase_target"] = matches["revised_target"].where(
        matches["revised_target"].notna(), matches["first_innings_score"] + 1
    )
    work = deliveries.loc[deliveries["innings"].isin([1, 2])].copy()
    work = work.merge(
        matches[["chase_target", "revised_target", "first_inn_revised_overs", "second_inn_revised_overs"]].reset_index(),
        on="match_id", how="left", validate="many_to_one",
    )
    work["innings_ball_limit"] = np.where(
        work["innings"].eq(1), work["first_inn_revised_overs"], work["second_inn_revised_overs"]
    )
    work["innings_ball_limit"] = pd.to_numeric(work["innings_ball_limit"], errors="coerce").fillna(DEFAULT_T20_BALL_LIMIT)
    work["limit_source"] = np.where(
        work["innings"].eq(1),
        np.where(work["first_inn_revised_overs"].notna(), "recorded revised ball limit", "default 120 legal balls"),
        np.where(work["second_inn_revised_overs"].notna(), "recorded revised ball limit", "default 120 legal balls"),
    )
    group_keys = [work["match_id"], work["innings"]]
    work["runs_before"] = work["total_runs_calculated"].groupby(group_keys, sort=False).cumsum() - work["total_runs_calculated"]
    wicket = work["is_wicket"].fillna(False).astype("int8")
    work["wickets_lost_before"] = wicket.groupby(group_keys, sort=False).cumsum() - wicket
    known = work["legal_delivery_calculated"].eq(True).fillna(False).astype("int8")
    unknown = work["legal_delivery_calculated"].isna().astype("int8")
    work["known_legal_balls_before"] = known.groupby(group_keys, sort=False).cumsum() - known
    work["unknown_legal_balls_before"] = unknown.groupby(group_keys, sort=False).cumsum() - unknown
    work["balls_remaining_min"] = (
        work["innings_ball_limit"] - work["known_legal_balls_before"] - work["unknown_legal_balls_before"]
    ).clip(lower=0)
    work["balls_remaining_max"] = (
        work["innings_ball_limit"] - work["known_legal_balls_before"]
    ).clip(lower=0)
    work["wickets_remaining"] = (10 - work["wickets_lost_before"]).clip(lower=0)
    work["current_run_rate_lower"] = _safe_ratio(
        work["runs_before"], work["known_legal_balls_before"] + work["unknown_legal_balls_before"], 6
    )
    work["current_run_rate_upper"] = _safe_ratio(
        work["runs_before"], work["known_legal_balls_before"], 6
    )
    work["required_runs"] = np.where(
        work["innings"].eq(2), (work["chase_target"] - work["runs_before"]).clip(lower=0), np.nan
    )
    work["required_run_rate_lower"] = _safe_ratio(
        work["required_runs"], work["balls_remaining_max"], 6
    )
    work["required_run_rate_upper"] = _safe_ratio(
        work["required_runs"], work["balls_remaining_min"], 6
    )
    work["known_balls_used_over_bin"] = (work["known_legal_balls_before"] // 6).astype(int)
    work["target_source"] = np.where(
        work["revised_target"].notna(), "recorded revised_target", "first-innings total + 1"
    )
    work["target_source"] = np.where(work["innings"].eq(2), work["target_source"], "not applicable in first innings")
    return work


def _match_context_summary(states: pd.DataFrame) -> pd.DataFrame:
    work = states.copy()
    work["chase_has_known_target"] = work["innings"].eq(2) & work["chase_target"].notna()
    keys = ["competition_group", "innings", "known_balls_used_over_bin", "limit_source"]
    return (
        work.groupby(keys, as_index=False, dropna=False)
        .agg(
            delivery_states=("match_id", "size"),
            matches=("match_id", "nunique"),
            target_known_states=("chase_has_known_target", "sum"),
            median_current_score_before_delivery=("runs_before", "median"),
            median_wickets_remaining=("wickets_remaining", "median"),
            median_balls_remaining_min=("balls_remaining_min", "median"),
            median_balls_remaining_max=("balls_remaining_max", "median"),
            median_current_run_rate_lower=("current_run_rate_lower", "median"),
            median_current_run_rate_upper=("current_run_rate_upper", "median"),
            median_required_runs=("required_runs", "median"),
            median_required_run_rate_lower=("required_run_rate_lower", "median"),
            median_required_run_rate_upper=("required_run_rate_upper", "median"),
            deliveries_with_prior_legal_uncertainty=("unknown_legal_balls_before", lambda s: s.gt(0).sum()),
        )
        .sort_values(keys)
        .reset_index(drop=True)
    )


def _player_match_consistency(
    deliveries: pd.DataFrame, dimension: pd.DataFrame
) -> pd.DataFrame:
    batting = deliveries.copy()
    batting["_runs"] = batting["runs"].fillna(0)
    batting["_known_balls"] = batting["batter_ball_faced_known"].astype("int8")
    batting["_uncertain_balls"] = batting["batter_ball_faced_uncertain"].astype("int8")
    batter_matches = (
        batting.groupby(["competition_group", "striker_id", "match_id"], as_index=False)
        .agg(
            match_runs=("_runs", "sum"),
            match_balls_faced_known=("_known_balls", "sum"),
            match_balls_faced_uncertain=("_uncertain_balls", "sum"),
        ).rename(columns={"striker_id": "player_id"})
    )
    batter_matches["match_strike_rate_known"] = _safe_ratio(
        batter_matches["match_runs"], batter_matches["match_balls_faced_known"], 100
    )
    batter_consistency = (
        batter_matches.groupby(["competition_group", "player_id"], as_index=False)
        .agg(
            batting_matches=("match_id", "nunique"),
            mean_match_runs=("match_runs", "mean"),
            median_match_runs=("match_runs", "median"),
            match_runs_std=("match_runs", "std"),
            match_runs_variance=("match_runs", "var"),
            low_score_matches=("match_runs", lambda s: s.le(10).sum()),
            high_impact_matches_30_plus=("match_runs", lambda s: s.ge(30).sum()),
            match_strike_rate_std=("match_strike_rate_known", "std"),
            match_strike_rate_median=("match_strike_rate_known", "median"),
            uncertain_balls_faced=("match_balls_faced_uncertain", "sum"),
        )
    )
    batter_consistency["low_score_match_percentage"] = _safe_ratio(
        batter_consistency["low_score_matches"], batter_consistency["batting_matches"], 100
    )
    batter_consistency["high_impact_match_percentage"] = _safe_ratio(
        batter_consistency["high_impact_matches_30_plus"], batter_consistency["batting_matches"], 100
    )
    batter_consistency["batting_consistency_minimum_met"] = batter_consistency["batting_matches"].ge(MIN_MATCHES_FOR_CONSISTENCY)

    bowling = deliveries.copy()
    bowling["_known_legal"] = bowling["legal_delivery_calculated"].eq(True).fillna(False).astype("int8")
    bowling["_runs_known"] = bowling["bowler_runs_conceded_known"].fillna(0)
    bowling["_wickets"] = bowling["bowler_wicket_calculated"].astype("int8")
    bowling["_dots"] = bowling["known_total_run_dot"].astype("int8")
    bowler_matches = (
        bowling.groupby(["competition_group", "bowler_id", "match_id"], as_index=False)
        .agg(
            match_legal_balls_known=("_known_legal", "sum"),
            match_bowler_runs_known=("_runs_known", "sum"),
            match_bowler_wickets=("_wickets", "sum"),
            match_dot_balls=("_dots", "sum"),
        ).rename(columns={"bowler_id": "player_id"})
    )
    bowler_matches["match_economy_known"] = _safe_ratio(
        bowler_matches["match_bowler_runs_known"], bowler_matches["match_legal_balls_known"], 6
    )
    bowler_matches["match_dot_percentage"] = _safe_ratio(
        bowler_matches["match_dot_balls"], bowler_matches["match_legal_balls_known"], 100
    )
    qualified = bowler_matches.loc[
        bowler_matches["match_legal_balls_known"].ge(MIN_BOWLER_SPELL_BALLS_FOR_CONSISTENCY)
    ].copy()
    bowler_consistency = (
        bowler_matches.groupby(["competition_group", "player_id"], as_index=False)
        .agg(
            bowling_matches=("match_id", "nunique"),
            mean_match_wickets=("match_bowler_wickets", "mean"),
            median_match_wickets=("match_bowler_wickets", "median"),
            match_wickets_std=("match_bowler_wickets", "std"),
        )
    )
    spell_consistency = (
        qualified.groupby(["competition_group", "player_id"], as_index=False)
        .agg(
            qualified_bowling_matches=("match_id", "nunique"),
            median_qualified_match_economy=("match_economy_known", "median"),
            qualified_match_economy_std=("match_economy_known", "std"),
            expensive_matches_economy_10_plus=("match_economy_known", lambda s: s.ge(10).sum()),
            wicket_taking_qualified_matches=("match_bowler_wickets", lambda s: s.gt(0).sum()),
            median_qualified_match_dot_percentage=("match_dot_percentage", "median"),
        )
    )
    bowler_consistency = bowler_consistency.merge(
        spell_consistency, on=["competition_group", "player_id"], how="left", validate="one_to_one"
    )
    bowler_consistency["expensive_match_percentage"] = _safe_ratio(
        bowler_consistency["expensive_matches_economy_10_plus"],
        bowler_consistency["qualified_bowling_matches"], 100,
    )
    bowler_consistency["wicket_taking_spell_percentage"] = _safe_ratio(
        bowler_consistency["wicket_taking_qualified_matches"],
        bowler_consistency["qualified_bowling_matches"], 100,
    )
    bowler_consistency["bowling_consistency_minimum_met"] = (
        bowler_consistency["bowling_matches"].ge(MIN_MATCHES_FOR_CONSISTENCY)
        & bowler_consistency["qualified_bowling_matches"].ge(MIN_MATCHES_FOR_CONSISTENCY)
    )

    result = batter_consistency.merge(
        bowler_consistency, on=["competition_group", "player_id"], how="outer", validate="one_to_one"
    )
    for column in ("batting_consistency_minimum_met", "bowling_consistency_minimum_met"):
        result[column] = result[column].fillna(False).astype(bool)
    result = _attach_dimension(result, dimension)
    result["minimum_matches_for_consistency"] = MIN_MATCHES_FOR_CONSISTENCY
    result["minimum_balls_per_bowling_match"] = MIN_BOWLER_SPELL_BALLS_FOR_CONSISTENCY
    return result.sort_values(["competition_group", "player_name"], na_position="last").reset_index(drop=True)


def _season_player_analysis(deliveries: pd.DataFrame, dimension: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group = ["competition_group", "season", "striker_id"]
    batting = _batting_aggregate(deliveries, group).rename(columns={"striker_id": "player_id"})
    batting.insert(1, "analysis_scope", "season")
    batting = _attach_dimension(batting, dimension)
    batting = batting.sort_values(["competition_group", "player_name", "season"], na_position="last")

    group = ["competition_group", "season", "bowler_id"]
    bowling = _bowling_aggregate(deliveries, group).rename(columns={"bowler_id": "player_id"})
    bowling.insert(1, "analysis_scope", "season")
    bowling = _attach_dimension(bowling, dimension)
    bowling = bowling.sort_values(["competition_group", "player_name", "season"], na_position="last")
    return batting, bowling


def _player_overall_analysis(deliveries: pd.DataFrame, dimension: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    batting = _batting_aggregate(deliveries, ["competition_group", "striker_id"]).rename(columns={"striker_id": "player_id"})
    batting.insert(1, "analysis_scope", "all_seasons")
    batting = _attach_dimension(batting, dimension)
    batting = batting.sort_values(["competition_group", "player_name"], na_position="last")

    bowling = _bowling_aggregate(deliveries, ["competition_group", "bowler_id"]).rename(columns={"bowler_id": "player_id"})
    bowling.insert(1, "analysis_scope", "all_seasons")
    bowling = _attach_dimension(bowling, dimension)
    bowling = bowling.sort_values(["competition_group", "player_name"], na_position="last")
    return batting, bowling


def _phase_analysis(deliveries: pd.DataFrame) -> pd.DataFrame:
    frame = deliveries.copy()
    frame["_known_legal"] = frame["legal_delivery_calculated"].eq(True).fillna(False).astype("int8")
    frame["_unknown_legal"] = frame["legal_delivery_calculated"].isna().astype("int8")
    frame["_wickets"] = frame["is_wicket"].fillna(False).astype("int8")
    frame["_bowler_wickets"] = frame["bowler_wicket_calculated"].astype("int8")
    frame["_dot"] = frame["known_total_run_dot"].astype("int8")
    frame["_known_bowler_runs"] = frame["bowler_runs_conceded_known"].fillna(0)
    frame["_boundary_known_legal"] = (
        frame["boundary_event"] & frame["legal_delivery_calculated"].eq(True).fillna(False)
    ).astype("int8")
    keys = ["competition_group", "season", "innings_phase"]
    summary = (
        frame.groupby(keys, as_index=False, dropna=False)
        .agg(
            delivery_rows=("match_id", "size"),
            matches=("match_id", "nunique"),
            batter_runs=("runs", "sum"),
            total_runs=("total_runs_calculated", "sum"),
            known_legal_balls=("_known_legal", "sum"),
            legal_status_unknown_deliveries=("_unknown_legal", "sum"),
            total_wicket_events=("_wickets", "sum"),
            bowler_attributable_wickets=("_bowler_wickets", "sum"),
            total_run_dot_balls=("_dot", "sum"),
            boundary_events_known_legal=("_boundary_known_legal", "sum"),
            bowler_runs_conceded_known=("_known_bowler_runs", "sum"),
            bowler_extra_attribution_unknown=("bowler_runs_attribution_unknown", "sum"),
            phase_assignment_uncertain_rows=("phase_assignment_uncertain", "sum"),
        )
    )
    summary["run_rate_per_known_legal_ball"] = _safe_ratio(summary["total_runs"], summary["known_legal_balls"], 6)
    summary["bowling_economy_known"] = _safe_ratio(summary["bowler_runs_conceded_known"], summary["known_legal_balls"], 6)
    summary["dot_ball_percentage_known_legal"] = _safe_ratio(summary["total_run_dot_balls"], summary["known_legal_balls"], 100)
    summary["boundary_percentage_known_legal"] = _safe_ratio(summary["boundary_events_known_legal"], summary["known_legal_balls"], 100)
    summary["bowler_wickets_per_100_known_legal_balls"] = _safe_ratio(summary["bowler_attributable_wickets"], summary["known_legal_balls"], 100)
    summary["phase_definition"] = "legal balls 1-36, 37-90, 91+; uncertain when unknown deliveries could cross a phase boundary"
    return summary.sort_values(keys).reset_index(drop=True)


def _match_situation_outputs(
    states: pd.DataFrame, matches: pd.DataFrame, out_dir: Path
) -> tuple[pd.DataFrame, dict[str, object]]:
    summary = _match_context_summary(states)
    context_rows = len(states)
    chase = states.loc[states["innings"].eq(2) & states["chase_target"].notna()]
    match_context = {
        "delivery_state_rows_first_and_second_innings": int(context_rows),
        "second_innings_state_rows_with_target": int(len(chase)),
        "target_sources": chase["target_source"].value_counts(dropna=False).to_dict(),
        "median_second_innings_required_run_rate_lower": float(chase["required_run_rate_lower"].median()) if not chase.empty else None,
        "median_second_innings_required_run_rate_upper": float(chase["required_run_rate_upper"].median()) if not chase.empty else None,
        "median_second_innings_wickets_remaining": float(chase["wickets_remaining"].median()) if not chase.empty else None,
        "closing_outcome_margin_check_is_ex_post_only": True,
    }
    # Match results are summarized separately and are never joined to delivery states.
    result_text = matches["result"].fillna("").astype(str)
    run_margin = pd.to_numeric(result_text.str.extract(r"(?i)won\s+by\s+(\d+)\s+runs?", expand=False), errors="coerce")
    wicket_margin = pd.to_numeric(result_text.str.extract(r"(?i)won\s+by\s+(\d+)\s+wickets?", expand=False), errors="coerce")
    match_context["matches_won_by_5_runs_or_fewer_ex_post"] = int(run_margin.le(5).sum())
    match_context["matches_won_by_2_wickets_or_fewer_ex_post"] = int(wicket_margin.le(2).sum())
    match_context["outcome_records_no_result_or_abandoned"] = int(result_text.str.contains("no result|abandoned", case=False, regex=True).sum())
    match_context["outcome_results_not_used_in_delivery_state_features"] = True
    return summary, match_context


def _eda_summary(
    tables: dict[str, pd.DataFrame],
    deliveries: pd.DataFrame,
    competition: pd.DataFrame,
    batting: pd.DataFrame,
    bowling: pd.DataFrame,
    phase: pd.DataFrame,
    consistency: pd.DataFrame,
    states: pd.DataFrame,
    match_context_json: dict[str, object],
) -> tuple[str, dict[str, object]]:
    phase_primary = phase.loc[~phase.innings_phase.isin(["Uncertain phase", "Tie-break / super-over innings"])]
    weighted_competitions = competition.copy()
    weighted_competitions["regular_wickets_total"] = (
        weighted_competitions["average_regular_innings_wickets"]
        * weighted_competitions["regular_innings"]
    )
    comp_groups = weighted_competitions.groupby("competition_group").agg(
        competitions=("comp_id", "nunique"),
        matches=("matches", "sum"),
        players=("unique_players", "sum"),
        regular_innings=("regular_innings", "sum"),
        regular_runs=("regular_innings_runs", "sum"),
        regular_wickets=("regular_wickets_total", "sum"),
        legal_balls=("regular_legal_balls", "sum"),
    )
    comp_groups["avg_innings_score"] = comp_groups.regular_runs / comp_groups.regular_innings.replace(0, np.nan)
    comp_groups["avg_wickets"] = comp_groups.regular_wickets / comp_groups.regular_innings.replace(0, np.nan)
    comp_groups["avg_run_rate"] = comp_groups.regular_runs * 6 / comp_groups.legal_balls.replace(0, np.nan)
    phase_rollup = phase_primary.groupby(["competition_group", "innings_phase"]).agg(
        runs=("total_runs", "sum"),
        legal_balls=("known_legal_balls", "sum"),
        bowler_wickets=("bowler_attributable_wickets", "sum"),
        boundary_events=("boundary_events_known_legal", "sum"),
        dot_balls=("total_run_dot_balls", "sum"),
    )
    phase_rollup["run_rate"] = phase_rollup.runs * 6 / phase_rollup.legal_balls.replace(0, np.nan)
    phase_rollup["boundary_pct"] = phase_rollup.boundary_events * 100 / phase_rollup.legal_balls.replace(0, np.nan)
    phase_rollup["dot_pct"] = phase_rollup.dot_balls * 100 / phase_rollup.legal_balls.replace(0, np.nan)
    batting_context = _context_metrics(deliveries, "innings context")
    consistency_eligible = consistency.loc[
        consistency["batting_consistency_minimum_met"]
        | consistency["bowling_consistency_minimum_met"]
    ]
    consistency_rollup = consistency_eligible.groupby("competition_group").agg(
        players_with_minimum_matches=("player_id", "nunique"),
        median_batting_low_score_match_pct=("low_score_match_percentage", "median"),
        median_batting_high_impact_match_pct=("high_impact_match_percentage", "median"),
        median_bowling_expensive_match_pct=("expensive_match_percentage", "median"),
        median_wicket_taking_spell_pct=("wicket_taking_spell_percentage", "median"),
    )
    eligible = batting.loc[batting.minimum_60_known_balls]
    eligible_bowl = bowling.loc[bowling.minimum_24_known_legal_balls]
    unknown = int(deliveries["legal_delivery_calculated"].isna().sum())
    uncertain_bowler = int(deliveries["bowler_runs_attribution_unknown"].sum())
    super_over_innings = deliveries.loc[deliveries.innings.gt(2), ["match_id", "innings"]].drop_duplicates().shape[0]

    findings = {
        "scope": "descriptive EDA only; no pressure model, prediction, or final player ranking",
        "dataset_rows": {key: int(len(value)) for key, value in tables.items()},
        "match_date_range": [str(tables["matches"].match_date.min()), str(tables["matches"].match_date.max())],
        "competition_group_summary": comp_groups.reset_index().to_dict("records"),
        "uncertain_legal_status_deliveries": unknown,
        "bowler_extra_attribution_uncertain_deliveries": uncertain_bowler,
        "phase_assignment_uncertain_rows": int(deliveries.phase_assignment_uncertain.sum()),
        "tie_break_or_super_over_innings": int(super_over_innings),
        "minimum_display_opportunity": {
            "batting_known_balls": MIN_BATTER_BALLS_FOR_COMPARISON,
            "bowling_known_legal_balls": MIN_BOWLER_BALLS_FOR_COMPARISON,
            "consistency_matches": MIN_MATCHES_FOR_CONSISTENCY,
            "qualified_bowling_match_legal_balls": MIN_BOWLER_SPELL_BALLS_FOR_CONSISTENCY,
        },
        "eligible_batters_descriptive_only": int(len(eligible)),
        "eligible_bowlers_descriptive_only": int(len(eligible_bowl)),
        "innings_context_comparison": batting_context.to_dict("records"),
        "player_consistency_summary": consistency_rollup.reset_index().to_dict("records"),
        "revised_innings_ball_limit_interpretation": "first_inn_revised_overs and second_inn_revised_overs values are multiples of six (30-114) and are treated as legal-ball limits; field names are ambiguous",
        "match_context": match_context_json,
        "phase_patterns": phase_rollup.reset_index().to_dict("records"),
    }

    lines = [
        "SUPERR SELECTOR — EXPLORATORY DATA ANALYSIS",
        "=" * 78,
        "",
        "Scope: descriptive batting, bowling, innings, competition, and match-context analysis.",
        "No pressure classification, pressure score, predictive model, or final player ranking was created.",
        "All results are computed from the supplied data pack. Source rows are preserved.",
        "",
        "DATASET AND FORMAT COVERAGE",
        f"- Ball-by-ball rows: {len(deliveries):,}; matches: {tables['matches'].match_id.nunique():,}; date range: {findings['match_date_range'][0]} to {findings['match_date_range'][1]}.",
        f"- Competition groups (competition entries, matches, summed mapped-player records, regular innings): {comp_groups.to_dict('index')}.",
        "- IPL means competition name IPL YYYY with format T20. T20I is kept distinct. Two remaining T20 events are grouped as Other T20.",
        "- Competition-level values in competition_analysis.csv are per competition. The group-level mean score, wickets, and run rate below are weighted by regular innings/legal-ball volume.",
        "- Super-over/tie-break innings (innings 3/4) are reported separately from standard first/second innings in rate summaries.",
        "",
        "BATTING PATTERNS",
        f"- Player rows: {len(batting):,}; {len(eligible):,} meet the descriptive 60-known-balls display filter.",
        "- Strike rate uses runs divided by known balls faced; a separate sensitivity column assumes all uncertain extras deliveries were faced. Wide is excluded; no-balls are included as faced balls.",
        "- Batting dot balls mean zero runs off the bat on a known faced delivery. Boundary ball and boundary run percentages are reported separately.",
        "- Match consistency uses match-wise aggregate runs and strike rates; low-score games are defined as <=10 runs and high-impact games as >=30 runs, descriptive thresholds only.",
        "- First versus second innings and season-level output are included in batting_analysis.csv; repeated team changes remain attached as team-name lists.",
        f"- First/second innings context metrics (strike rate, economy, wickets, dot balls, and boundaries): {batting_context.to_dict('records')}.",
        f"- Match-level consistency summaries for players meeting the {MIN_MATCHES_FOR_CONSISTENCY}-match minimum: {consistency_rollup.reset_index().to_dict('records')}.",
        "",
        "BOWLING PATTERNS",
        f"- Player rows: {len(bowling):,}; {len(eligible_bowl):,} meet the descriptive 24-known-legal-ball display filter.",
        "- Bowling economy and strike rate use known legal balls only. Wickets use the Phase 2 bowler-attributable dismissal mapping; total wicket events are also retained.",
        "- Bowling dot balls mean zero total runs on a known legal delivery. Boundary concession percentage is boundary events over deliveries with known legality.",
        "- Consistency spell summaries use match spells of at least 12 known legal balls; expensive spell is economy >=10, descriptive only.",
        "- First versus second innings and season-level output are included in bowling_analysis.csv.",
        "",
        "INNINGS PHASE PATTERNS",
        "- Phases use legal-ball positions: balls 1-36, 37-90, and 91 onward. A delivery receives a phase only when uncertain earlier extras cannot move it across a phase boundary.",
        f"- Deliveries with uncertain legal status: {unknown:,}; uncertain bowler-extra attribution: {uncertain_bowler:,}; rows whose phase can vary under unknown legality: {int(deliveries.phase_assignment_uncertain.sum()):,}.",
        "- Phase totals include an explicit Uncertain phase row and tie-break innings row; comparisons in this report exclude those two categories.",
        f"- Phase summaries (combined across seasons within each group; runs/legal balls, bowler wickets, boundary%, dot%): {phase_rollup.reset_index().to_dict('records')}.",
        "",
        "MATCH SITUATION DISCOVERIES",
        f"- State rows use only information available before each delivery. Current score, wickets lost, legal balls used, balls remaining, and chase requirement are reconstructed from delivery records.",
        "- For second innings, revised_target is used where supplied; otherwise target is first-innings total + 1. Final result fields are never joined to delivery-state features.",
        "- Revised innings-limit columns are named as overs but their observed values are multiples of six (30-114); they are treated as legal-ball limits. If absent, the default cap is 120 legal balls.",
        f"- Reconstructed context summary: {match_context_json}.",
        "- Outcome-margin counts in the JSON summary are ex-post descriptive checks only and are not used to construct any earlier match state.",
        "",
        "SAMPLE SIZE AND LIMITATIONS",
        f"- Display-only minimums: {MIN_BATTER_BALLS_FOR_COMPARISON} known balls faced for batting; {MIN_BOWLER_BALLS_FOR_COMPARISON} known legal balls for bowling; {MIN_MATCHES_FOR_CONSISTENCY} matches for consistency comparisons; {MIN_BOWLER_SPELL_BALLS_FOR_CONSISTENCY} known legal balls per bowling match spell.",
        "- Filters do not remove players from CSV outputs and do not establish statistical reliability; they only flag more interpretable descriptive comparisons.",
        "- Carry-forward data issues: 267 delivery legality classifications uncertain; 264 'other' extras labels plus two untyped nonzero-extras rows leave 266 deliveries with uncertain bowler-extra attribution; one leg-byes row has runs=1 with zero extras; 16 abandoned matches have no deliveries; repeated playing-eleven keys; no independent innings score totals.",
        "- No source rows were deleted. Unknown legal deliveries and bowler extras remain flagged and are not silently reallocated.",
        "",
        "OUTPUTS",
        "- competition_analysis.csv, batting_analysis.csv, bowling_analysis.csv, innings_phase_analysis.csv, innings_context_analysis.csv, player_consistency.csv, match_context_analysis.csv, and eda_summary.json are generated under outputs/reports/.",
        "- EDA charts are generated under outputs/figures/eda/.",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n", findings


def run_eda(data_source: str | Path | dict[str, pd.DataFrame], report_dir: str | Path, figure_dir: str | Path) -> dict[str, Any]:
    """Run EDA summaries, match-state reconstruction, and chart generation."""
    tables = data_source if isinstance(data_source, dict) else load_data_pack(data_source)
    deliveries = add_delivery_context(tables)
    dimension = _player_dimension(tables)
    innings = _build_innings_summary(deliveries)
    competition = _competition_analysis(tables, innings)
    batting_overall, bowling_overall = _player_overall_analysis(deliveries, dimension)
    batting_season, bowling_season = _season_player_analysis(deliveries, dimension)
    batting = pd.concat([batting_overall, batting_season], ignore_index=True, sort=False)
    bowling = pd.concat([bowling_overall, bowling_season], ignore_index=True, sort=False)
    phases = _phase_analysis(deliveries)
    context_comparison = _context_metrics(deliveries, "first-versus-second innings")
    consistency = _player_match_consistency(deliveries, dimension)
    states = _build_match_states(deliveries, tables)
    match_context, match_context_json = _match_situation_outputs(states, tables["matches"], Path(figure_dir))

    reports = Path(report_dir)
    reports.mkdir(parents=True, exist_ok=True)
    competition.to_csv(reports / "competition_analysis.csv", index=False)
    batting.to_csv(reports / "batting_analysis.csv", index=False)
    bowling.to_csv(reports / "bowling_analysis.csv", index=False)
    phases.to_csv(reports / "innings_phase_analysis.csv", index=False)
    consistency.to_csv(reports / "player_consistency.csv", index=False)
    match_context.to_csv(reports / "match_context_analysis.csv", index=False)
    context_comparison.to_csv(reports / "innings_context_analysis.csv", index=False)

    from src.analysis.eda_visualizations import generate_eda_figures

    figures = generate_eda_figures(
        tables=tables,
        deliveries=deliveries,
        competition=competition,
        batting=batting_overall,
        bowling=bowling_overall,
        phases=phases,
        consistency=consistency,
        context=match_context,
        innings_context=context_comparison,
        states=states,
        out_dir=figure_dir,
    )
    summary_text, summary_json = _eda_summary(
        tables, deliveries, competition, batting_overall, bowling_overall, phases,
        consistency, states, match_context_json,
    )
    (reports / "eda_summary.txt").write_text(summary_text, encoding="utf-8")
    import json

    with (reports / "eda_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_json, handle, indent=2, ensure_ascii=False, default=str, allow_nan=False)
    LOGGER.info(
        "EDA complete: %s competition rows, %s batter rows, %s bowler rows, %s figures",
        len(competition), len(batting), len(bowling), len(figures),
    )
    return {
        "tables": tables,
        "deliveries": deliveries,
        "competition": competition,
        "batting": batting,
        "bowling": bowling,
        "phase": phases,
        "consistency": consistency,
        "states": states,
        "match_context": match_context,
        "figures": figures,
        "summary": summary_json,
    }
