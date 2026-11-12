"""Relational, identifier, and cricket-specific data validation."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import pandas as pd

from src.data.cricket_calculations import calculate_delivery_metrics

LOGGER = logging.getLogger(__name__)


def duplicate_key_summary(frame: pd.DataFrame, key: Iterable[str]) -> dict[str, int]:
    """Count duplicate rows for a proposed identifier without dropping them."""
    columns = list(key)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing identifier columns: {', '.join(sorted(missing))}")
    duplicated = frame.duplicated(columns, keep=False)
    return {
        "duplicate_rows": int(duplicated.sum()),
        "duplicate_keys": int(frame.loc[duplicated, columns].drop_duplicates().shape[0]),
    }


def validate_identifiers(
    child: pd.DataFrame,
    parent: pd.DataFrame,
    child_columns: Iterable[str],
    parent_columns: Iterable[str],
) -> dict[str, object]:
    """Report foreign keys not found in the parent, preserving all source rows."""
    child_cols, parent_cols = list(child_columns), list(parent_columns)
    if len(child_cols) != len(parent_cols):
        raise ValueError("Child and parent keys must have the same number of columns")
    for column in child_cols:
        if column not in child:
            raise ValueError(f"Missing child identifier column: {column}")
    for column in parent_cols:
        if column not in parent:
            raise ValueError(f"Missing parent identifier column: {column}")

    left = child[child_cols].drop_duplicates().copy()
    left.columns = [f"key_{i}" for i in range(len(child_cols))]
    right = parent[parent_cols].drop_duplicates().copy()
    right.columns = [f"key_{i}" for i in range(len(parent_cols))]
    unmatched = left.merge(right, how="left", indicator=True)
    unmatched = unmatched.loc[unmatched["_merge"].eq("left_only")]
    return {
        "child_distinct_keys": int(len(left)),
        "unmatched_distinct_keys": int(len(unmatched)),
        "unmatched_key_sample": unmatched.drop(columns="_merge").head(20).to_dict("records"),
    }


def _check(rows: list[dict[str, object]], name: str, result: dict[str, object]) -> None:
    missing = int(result.get("unmatched_distinct_keys", 0))
    rows.append(
        {
            "check": name,
            "status": "PASS" if missing == 0 else "REVIEW",
            "records_checked": result.get("child_distinct_keys", 0),
            "unmatched_count": missing,
            "details": str(result.get("unmatched_key_sample", [])),
        }
    )


def relational_validation(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Run foreign-key and key-uniqueness checks for this inspected data pack."""
    balls = tables["ball_by_ball"]
    matches = tables["matches"]
    competitions = tables["competitions"]
    teams = tables["teams"]
    playing = tables["playing_eleven"]
    player_map = tables["competition_team_player_mapping"]
    checks: list[dict[str, object]] = []

    _check(checks, "ball_by_ball.match_id -> matches.match_id", validate_identifiers(
        balls, matches, ["match_id"], ["match_id"]
    ))
    match_coverage = validate_identifiers(matches, balls, ["match_id"], ["match_id"])
    missing_ball_ids = matches.loc[~matches.match_id.isin(balls.match_id), ["match_id", "result"]]
    checks.append({
        "check": "matches.match_id -> ball_by_ball.match_id (all match coverage)",
        "status": "REVIEW" if match_coverage["unmatched_distinct_keys"] else "PASS",
        "records_checked": match_coverage["child_distinct_keys"],
        "unmatched_count": match_coverage["unmatched_distinct_keys"],
        "details": f"matches without deliveries by result={missing_ball_ids.result.value_counts(dropna=False).to_dict()}; unmatched={match_coverage['unmatched_key_sample']}",
    })
    _check(checks, "matches.comp_id -> competitions.comp_id", validate_identifiers(
        matches, competitions, ["comp_id"], ["comp_id"]
    ))
    _check(checks, "competitions.comp_id -> matches.comp_id (coverage)", validate_identifiers(
        competitions, matches, ["comp_id"], ["comp_id"]
    ))

    home_names = matches.merge(
        teams.rename(columns={"team_id": "home_team_id", "team_name": "catalog_team_name"}),
        on=["comp_id", "home_team_id"], how="left", validate="many_to_one"
    )
    away_names = matches.merge(
        teams.rename(columns={"team_id": "away_team_id", "team_name": "catalog_team_name"}),
        on=["comp_id", "away_team_id"], how="left", validate="many_to_one"
    )
    bad_home = home_names.loc[home_names.home_team.ne(home_names.catalog_team_name), ["comp_id", "home_team_id", "home_team", "catalog_team_name"]]
    bad_away = away_names.loc[away_names.away_team.ne(away_names.catalog_team_name), ["comp_id", "away_team_id", "away_team", "catalog_team_name"]]
    name_mismatches = pd.concat([bad_home, bad_away], ignore_index=True)
    checks.append({
        "check": "match home/away team names agree with team IDs",
        "status": "PASS" if name_mismatches.empty else "REVIEW",
        "records_checked": 2 * len(matches),
        "unmatched_count": len(name_mismatches),
        "details": name_mismatches.head(20).to_dict("records"),
    })

    team_lookup = teams[["comp_id", "team_id"]].drop_duplicates()
    match_teams = pd.concat(
        [
            matches[["comp_id", "home_team_id"]].rename(columns={"home_team_id": "team_id"}),
            matches[["comp_id", "away_team_id"]].rename(columns={"away_team_id": "team_id"}),
        ],
        ignore_index=True,
    )
    _check(checks, "home/away match teams -> teams by competition", validate_identifiers(
        match_teams, team_lookup, ["comp_id", "team_id"], ["comp_id", "team_id"]
    ))
    match_comp = matches[["match_id", "comp_id"]].drop_duplicates()
    ball_comp = balls[["match_id", "batting_team_id", "bowling_team_id"]].merge(
        match_comp, on="match_id", how="left", validate="many_to_one"
    )
    for column, role in (("batting_team_id", "batting"), ("bowling_team_id", "bowling")):
        role_lookup = ball_comp[["comp_id", column]].rename(columns={column: "team_id"})
        _check(checks, f"ball {role} teams -> teams by competition", validate_identifiers(
            role_lookup, team_lookup, ["comp_id", "team_id"], ["comp_id", "team_id"]
        ))

    ball_context = balls[["match_id", "batting_team_id", "bowling_team_id", "striker_id", "non_striker_id", "bowler_id"]].merge(
        match_comp, on="match_id", how="left", validate="many_to_one"
    )
    mapping_keys = player_map[["comp_id", "team_id", "player_id"]]
    for team_col, player_col, role in (
        ("batting_team_id", "striker_id", "striker"),
        ("batting_team_id", "non_striker_id", "non-striker"),
        ("bowling_team_id", "bowler_id", "bowler"),
    ):
        deliveries = ball_context[["comp_id", team_col, player_col]].rename(
            columns={team_col: "team_id", player_col: "player_id"}
        )
        _check(checks, f"delivery {role} -> competition player mapping", validate_identifiers(
            deliveries, mapping_keys, ["comp_id", "team_id", "player_id"], ["comp_id", "team_id", "player_id"]
        ))

    captains = pd.concat(
        [
            matches[["comp_id", "home_team_id", "home_team_captain_id"]].rename(
                columns={"home_team_id": "team_id", "home_team_captain_id": "player_id"}
            ),
            matches[["comp_id", "away_team_id", "away_team_captain_id"]].rename(
                columns={"away_team_id": "team_id", "away_team_captain_id": "player_id"}
            ),
        ], ignore_index=True,
    ).dropna(subset=["player_id"])
    _check(checks, "nonmissing match captains -> competition team player mapping", validate_identifiers(
        captains, mapping_keys, ["comp_id", "team_id", "player_id"], ["comp_id", "team_id", "player_id"]
    ))
    awards = matches.loc[matches.man_of_the_match.ne(0) & matches.man_of_the_match.notna(), ["comp_id", "man_of_the_match"]].rename(
        columns={"man_of_the_match": "player_id"}
    )
    award_players = player_map[["comp_id", "player_id"]].drop_duplicates()
    _check(checks, "nonzero man-of-the-match IDs -> competition player mapping", validate_identifiers(
        awards, award_players, ["comp_id", "player_id"], ["comp_id", "player_id"]
    ))

    for team_column in ("toss_win_team_id", "win_team_id"):
        present = matches.loc[matches[team_column].notna()]
        valid_side = present[team_column].eq(present.home_team_id) | present[team_column].eq(present.away_team_id)
        invalid = present.loc[~valid_side, ["match_id", "comp_id", team_column, "home_team_id", "away_team_id"]]
        checks.append({
            "check": f"{team_column} is one of the match teams",
            "status": "PASS" if invalid.empty else "REVIEW",
            "records_checked": len(present),
            "unmatched_count": len(invalid),
            "details": invalid.head(20).to_dict("records"),
        })

    roster = playing[["match_id", "team_id", "player_id"]].merge(
        match_comp, on="match_id", how="left", validate="many_to_one"
    )
    _check(checks, "playing_eleven -> competition player mapping", validate_identifiers(
        roster, mapping_keys, ["comp_id", "team_id", "player_id"], ["comp_id", "team_id", "player_id"]
    ))

    key_specs = (
        ("matches", matches, ["match_id"]),
        ("competitions", competitions, ["comp_id"]),
        ("teams", teams, ["comp_id", "team_id"]),
        ("competition_team_player_mapping", player_map, ["comp_id", "team_id", "player_id"]),
        ("ball_by_ball delivery", balls, ["match_id", "innings", "overid", "over_ball_num"]),
    )
    for name, frame, key in key_specs:
        summary = duplicate_key_summary(frame, key)
        checks.append({
            "check": f"{name} unique key {tuple(key)}",
            "status": "PASS" if summary["duplicate_rows"] == 0 else "REVIEW",
            "records_checked": len(frame),
            "unmatched_count": summary["duplicate_keys"],
            "details": f"duplicate rows={summary['duplicate_rows']}; duplicate keys={summary['duplicate_keys']}",
        })
    innings_sequence = balls.groupby(["match_id", "innings"], dropna=False).agg(
        rows=("innings_ball_num", "size"),
        minimum=("innings_ball_num", "min"),
        maximum=("innings_ball_num", "max"),
        unique=("innings_ball_num", "nunique"),
    )
    nonsequential = innings_sequence.loc[
        innings_sequence.minimum.ne(1)
        | innings_sequence.maximum.ne(innings_sequence.rows)
        | innings_sequence.unique.ne(innings_sequence.rows)
    ]
    checks.append({
        "check": "innings_ball_num is a contiguous delivery-row sequence",
        "status": "PASS" if nonsequential.empty else "REVIEW",
        "records_checked": len(innings_sequence),
        "unmatched_count": len(nonsequential),
        "details": "This ordinal includes recorded wide/no-ball events; do not use it as legal-ball count.",
    })
    roster_dups = duplicate_key_summary(playing, ["match_id", "team_id", "player_id"])
    checks.append({
        "check": "playing_eleven repeated match/team/player keys",
        "status": "REVIEW" if roster_dups["duplicate_keys"] else "PASS",
        "records_checked": len(playing),
        "unmatched_count": roster_dups["duplicate_keys"],
        "details": (
            f"{roster_dups['duplicate_rows']} rows across {roster_dups['duplicate_keys']} keys; "
            "source has no innings column, so repeated player rows are preserved"
        ),
    })

    date_values = pd.to_datetime(matches["match_date"], errors="coerce")
    checks.append({
        "check": "matches.match_date parses as a date",
        "status": "PASS" if date_values.notna().all() else "REVIEW",
        "records_checked": len(matches),
        "unmatched_count": int(date_values.isna().sum()),
        "details": f"range={date_values.min()} to {date_values.max()}",
    })
    return pd.DataFrame(checks)


def cricket_validation(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Produce scoring checks and a row for each innings total."""
    deliveries = calculate_delivery_metrics(tables["ball_by_ball"])
    checks: list[dict[str, object]] = []

    def add(name: str, value: int | float, status: str, details: str) -> None:
        checks.append({"check": name, "value": value, "status": status, "details": details})

    total_runs = int(deliveries["total_runs_calculated"].sum())
    batter_runs = int(deliveries["batter_runs_calculated"].sum())
    extras = int(deliveries["extras_calculated"].sum())
    add("Delivery total runs = runs + extras1_runs + extras_2_runs", total_runs, "CALCULATED",
        f"batter-runs field sum={batter_runs}; extras sum={extras}; no innings-total field exists in matches.csv for external reconciliation")
    add("Batter runs (sum of runs field)", batter_runs, "CALCULATED",
        "The data separates runs from two extras fields; one leg-byes row has runs=1 and no extras, flagged for source review")
    add("Extras (extras1_runs + extras_2_runs)", extras, "CALCULATED",
        "Rows are summed as stored; two nonzero-extras rows lack extras_type")

    legal_counts = deliveries["legal_delivery_calculated"].value_counts(dropna=False)
    known_illegal = int(legal_counts.get(False, 0))
    known_legal = int(legal_counts.get(True, 0))
    unknown_legal = int(legal_counts.get(pd.NA, 0))
    add("Legal deliveries (tri-state classification)", known_legal, "REVIEW" if unknown_legal else "CALCULATED",
        f"known legal={known_legal}; known illegal={known_illegal}; legality unknown={unknown_legal}; wide and Overstep classified illegal; ambiguous extras left unknown")

    bowler_runs = int(deliveries["bowler_runs_conceded_known"].sum())
    bowler_uncertain = int(deliveries["bowler_runs_attribution_unknown"].sum())
    add("Bowler runs conceded (known attribution only)", bowler_runs, "REVIEW" if bowler_uncertain else "CALCULATED",
        f"Includes batter runs and wide/no-ball extras; excludes byes, leg-byes and penalty; {bowler_uncertain} deliveries have extras with unknown attribution")

    wickets = int(deliveries["is_wicket"].fillna(False).astype(bool).sum())
    bowler_wickets = int(deliveries["bowler_wicket_calculated"].sum())
    add("Bowler-attributable wickets", bowler_wickets, "CALCULATED",
        f"All wicket events={wickets}; uses caught, bowled, LBW, stumped, and hit wicket; excludes run-out, retire out, and obstruction")
    unknown_wickets = int(deliveries["wicket_attribution_unknown"].sum())
    add("Wicket attribution unknown", unknown_wickets, "REVIEW" if unknown_wickets else "CALCULATED",
        "Wicket rows with a missing dismissal type are not credited to a bowler")

    innings = (
        deliveries.groupby(["match_id", "innings"], dropna=False)
        .agg(
            deliveries=("runs", "size"),
            batter_runs=("batter_runs_calculated", "sum"),
            extras=("extras_calculated", "sum"),
            total_runs=("total_runs_calculated", "sum"),
            known_legal_balls=("legal_delivery_calculated", lambda s: int(s.eq(True).sum())),
            legality_unknown=("legal_delivery_calculated", lambda s: int(s.isna().sum())),
            wickets=("is_wicket", "sum"),
            bowler_runs_known=("bowler_runs_conceded_known", "sum"),
            bowler_wickets=("bowler_wicket_calculated", "sum"),
        )
        .reset_index()
    )
    add("Innings run reconciliation", int(innings.total_runs.sum()), "NOT AVAILABLE",
        "No innings scorecard or innings-total column is supplied; per-innings totals are derived from delivery fields and cannot be compared to an independent total")
    return pd.DataFrame(checks), innings
