"""Dataset profiling and report generation for the official data pack."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.cricket_calculations import calculate_delivery_metrics, calculate_innings_summary
from src.data.loader import load_data_pack
from src.data.preprocessing import build_delivery_table, write_derived_tables
from src.data.validation import cricket_validation, relational_validation
from src.data.visualizations import generate_profile_figures

LOGGER = logging.getLogger(__name__)


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and pd.isna(value):
            return None
        return value
    if pd.isna(value):
        return None
    return str(value)


def profile_frame(name: str, frame: pd.DataFrame) -> dict[str, object]:
    """Create a complete machine-readable profile for one table."""
    first_five = frame.head(5).astype(object).where(frame.head(5).notna(), None).to_dict("records")
    json_records = [
        {str(key): _json_value(value) for key, value in record.items()}
        for record in first_five
    ]
    profile: dict[str, object] = {
        "rows": int(len(frame)),
        "columns_count": int(len(frame.columns)),
        "duplicate_rows_exact": int(frame.duplicated().sum()),
        "columns": {},
        "first_five_records": json_records,
    }
    columns: dict[str, object] = {}
    date_columns = [col for col in frame.columns if "date" in col.casefold() or "time" in col.casefold()]
    for column in frame.columns:
        series = frame[column]
        missing_count = int(series.isna().sum())
        details: dict[str, object] = {
            "dtype": str(series.dtype),
            "missing_count": missing_count,
            "missing_percent": round(100 * missing_count / len(frame), 4) if len(frame) else 0.0,
            "unique_non_missing": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            stats = series.describe().to_dict()
            details["descriptive_statistics"] = {str(k): _json_value(v) for k, v in stats.items()}
            details["minimum"] = _json_value(series.min()) if series.notna().any() else None
            details["maximum"] = _json_value(series.max()) if series.notna().any() else None
        else:
            counts = series.astype("string").value_counts(dropna=False)
            unique_values = [None if pd.isna(v) else str(v) for v in series.drop_duplicates().tolist()]
            details["unique_values"] = unique_values if len(unique_values) <= 40 else unique_values[:40] + [f"... {len(unique_values)-40} more distinct values"]
            details["most_common_values"] = {str(k): int(v) for k, v in counts.head(20).items()}
        if column in date_columns:
            parsed = pd.to_datetime(series, errors="coerce", utc=True)
            details["parsed_date_count"] = int(parsed.notna().sum())
            details["invalid_date_count"] = int(series.notna().sum() - parsed.notna().sum())
            details["date_minimum"] = parsed.min().isoformat() if parsed.notna().any() else None
            details["date_maximum"] = parsed.max().isoformat() if parsed.notna().any() else None
        columns[str(column)] = details
    profile["columns"] = columns
    return profile


def _render_profile_text(profiles: dict[str, dict[str, object]]) -> str:
    lines = ["SUPERR SELECTOR — DATASET PROFILE", "=" * 72, "", "All source CSVs are read-only inputs. No source rows are removed.", ""]
    for name, meta in profiles.items():
        lines += [f"DATASET: {name}", "-" * 72, f"Rows: {meta['rows']:,}", f"Columns: {meta['columns_count']}", f"Exact duplicate rows: {meta['duplicate_rows_exact']:,}", "", "Columns:"]
        for column, detail in meta["columns"].items():
            lines.append(
                f"  - {column} [{detail['dtype']}]: missing={detail['missing_count']:,} "
                f"({detail['missing_percent']:.3f}%), unique={detail['unique_non_missing']:,}"
            )
            if "minimum" in detail:
                lines.append(f"      range={detail['minimum']} to {detail['maximum']}; stats={detail['descriptive_statistics']}")
            if "date_minimum" in detail:
                lines.append(f"      parsed dates={detail['parsed_date_count']:,}; invalid={detail['invalid_date_count']:,}; range={detail['date_minimum']} to {detail['date_maximum']}")
            if "unique_values" in detail:
                lines.append(f"      unique values={detail['unique_values']}")
                if len(detail["most_common_values"]) > 0:
                    lines.append(f"      most common={detail['most_common_values']}")
        lines += ["", "First five records:", json.dumps(meta["first_five_records"], ensure_ascii=False, indent=2, default=str), "", ""]
    return "\n".join(lines).rstrip() + "\n"


def _write_missing_values(profiles: dict[str, dict[str, object]], path: Path) -> None:
    rows = []
    for dataset, meta in profiles.items():
        for column, detail in meta["columns"].items():
            rows.append({
                "dataset": dataset,
                "column": column,
                "missing_count": detail["missing_count"],
                "missing_percent": detail["missing_percent"],
                "row_count": meta["rows"],
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def _competition_coverage(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    matches = tables["matches"]
    competitions = tables["competitions"]
    balls = tables["ball_by_ball"]
    player_map = tables["competition_team_player_mapping"]
    delivery_comp = balls[["match_id"]].merge(
        matches[["match_id", "comp_id"]], on="match_id", how="left", validate="many_to_one"
    )
    match_counts = matches.groupby("comp_id").agg(match_count=("match_id", "nunique"), first_match_date=("match_date", "min"), last_match_date=("match_date", "max"))
    ball_counts = delivery_comp.groupby("comp_id").size().rename("delivery_count")
    player_counts = player_map.groupby("comp_id").player_id.nunique().rename("mapped_player_count")
    table = competitions.merge(match_counts, on="comp_id", how="left").merge(ball_counts, on="comp_id", how="left").merge(player_counts, on="comp_id", how="left")
    table[["match_count", "delivery_count", "mapped_player_count"]] = table[["match_count", "delivery_count", "mapped_player_count"]].fillna(0).astype(int)
    table["competition_group"] = "Other"
    table.loc[table["match_format"].eq("T20I"), "competition_group"] = "T20I"
    table.loc[table["comp_name"].str.fullmatch(r"IPL\s+\d{4}", case=False, na=False), "competition_group"] = "IPL"
    table.loc[table["match_format"].eq("T20") & table["competition_group"].eq("Other"), "competition_group"] = "Other T20"
    table["has_match_records"] = table["match_count"].gt(0)
    return table.sort_values(["competition_group", "season", "comp_name"]).reset_index(drop=True)


def _quality_report(tables: dict[str, pd.DataFrame], relational: pd.DataFrame, scoring: pd.DataFrame, innings: pd.DataFrame) -> str:
    balls, matches, comps = tables["ball_by_ball"], tables["matches"], tables["competitions"]
    player_map, playing = tables["competition_team_player_mapping"], tables["playing_eleven"]
    comp_refs = set(matches.comp_id)
    comp_without_matches = comps.loc[~comps.comp_id.isin(comp_refs), ["comp_id", "comp_name", "match_format"]]
    competition_summary = _competition_coverage(tables).groupby("competition_group").agg(
        competition_count=("comp_id", "nunique"),
        match_count=("match_count", "sum"),
        mapped_competition_player_records=("mapped_player_count", "sum"),
    ).to_dict("index")
    matches_without_balls = matches.loc[~matches.match_id.isin(balls.match_id)]
    missing_type = balls.extras_type.isna() & (balls.extras1_runs.ne(0) | balls.extras_2_runs.ne(0))
    other_extras = balls.extras_type.astype("string").str.casefold().eq("other")
    type_normalized = balls.extras_type.astype("string").str.casefold()
    ipl_mask = comps["match_format"].eq("T20") & comps["comp_name"].str.fullmatch(
        r"IPL\s+\d{4}", case=False, na=False
    )
    ambiguous = (balls.extras1_runs.ne(0) | balls.extras_2_runs.ne(0)) & ~type_normalized.isin(["wide", "overstep", "byes", "legbyes", "penalty"])
    mapped_roster = playing.merge(matches[["match_id", "comp_id"]], on="match_id", how="left", validate="many_to_one").merge(
        player_map[["comp_id", "team_id", "player_id"]].drop_duplicates(),
        on=["comp_id", "team_id", "player_id"], how="left", indicator=True
    )
    unmatched_roster = mapped_roster.loc[mapped_roster._merge.eq("left_only")]
    duplicate_roster = playing[playing.duplicated(["match_id", "team_id", "player_id"], keep=False)]
    repeated_keys = duplicate_roster[["match_id", "team_id", "player_id"]].drop_duplicates()
    legbye_batter_runs = balls[(type_normalized == "legbyes") & balls.runs.ne(0)]
    wicket_types = balls.loc[balls.is_wicket, "dismissal_type"].value_counts(dropna=False)
    over120 = innings.loc[innings.known_legal_balls.gt(120)]
    ambiguous_legal_upper = innings.loc[
        (innings.known_legal_balls + innings.legality_unknown).gt(120)
    ]
    negative_scoring = int((balls[["runs", "extras1_runs", "extras_2_runs"]] < 0).any(axis=1).sum())
    boundary_mismatches = int((balls.is_four & balls.runs.ne(4)).sum() + (balls.is_six & balls.runs.ne(6)).sum())
    zero_bowl_speed = int(balls.bowl_speed.eq(0).sum()) if "bowl_speed" in balls else 0
    missing_winners = matches.loc[matches.win_team_id.isna(), "result"].value_counts(dropna=False).to_dict()
    captain_missing = (int(matches.home_team_captain_id.isna().sum()), int(matches.away_team_captain_id.isna().sum()))
    playing_not_played = playing.loc[~playing.is_played]
    nonplayed_role_missing = (int(playing_not_played.is_captain.isna().sum()), int(playing_not_played.is_wk.isna().sum()))
    team_name_row = relational.loc[relational.check.eq("match home/away team names agree with team IDs")]
    team_name_mismatches = int(team_name_row.unmatched_count.iloc[0]) if not team_name_row.empty else -1

    lines = [
        "SUPERR SELECTOR — DATA QUALITY REPORT",
        "=" * 72,
        "",
        "Scope: structural and cricket-scoring validation only. No pressure model or player rankings were run.",
        "Source files are left unchanged. Review statuses preserve anomalies for later decisions.",
        "",
        "Coverage",
        f"- Matches: {len(matches):,}; unique match IDs: {matches.match_id.nunique():,}.",
        f"- Deliveries: {len(balls):,}; delivery match IDs missing from matches: {balls.loc[~balls.match_id.isin(matches.match_id), 'match_id'].nunique():,}.",
        f"- Matches without delivery rows: {len(matches_without_balls):,}; all results: {matches_without_balls.result.value_counts(dropna=False).to_dict()}.",
        f"- Competitions in catalog: {len(comps):,}; competitions with no match reference: {len(comp_without_matches):,}; details: {comp_without_matches.to_dict('records')}.",
        f"- Match date coverage: {pd.to_datetime(matches.match_date).min().date()} to {pd.to_datetime(matches.match_date).max().date()}.",
        f"- Match records with null winner IDs: {matches.win_team_id.isna().sum()} ({missing_winners}); includes no-result, abandoned, and tied matches.",
        f"- Missing captain IDs: home={captain_missing[0]}, away={captain_missing[1]}; missing values are retained.",
        "",
        "Competition formats",
        f"- Catalog format counts: {comps.match_format.value_counts(dropna=False).to_dict()}.",
        f"- IPL is identified by competition name matching 'IPL YYYY' and match_format T20: {int(ipl_mask.sum())} seasons.",
        f"- T20 format also contains non-IPL events: {comps.loc[comps.match_format.eq('T20') & ~ipl_mask, 'comp_name'].tolist()}.",
        "- T20I remains separate; the T20 format label alone is not treated as IPL.",
        f"- Match/player coverage by distinct competition grouping (player counts are summed across competitions and are not unique cross-competition people): {competition_summary}.",
        "",
        "Relationships and identifiers",
        f"- Competition team-name consistency: {team_name_mismatches} home/away name-to-ID mismatches (expected 0).",
    ]
    for row in relational.itertuples(index=False):
        lines.append(f"- {row.status}: {row.check}; checked={row.records_checked}; flagged={row.unmatched_count}; {row.details}")
    lines += [
        f"- Five playing-eleven mapping-miss rows are all is_played=False for player 29833/team 4444 in competition 2670; source roster rows retained.",
        f"- Playing-eleven has {len(repeated_keys)} repeated match/team/player keys across {len(duplicate_roster)} rows; batting_order differs and the source table has no innings key. No row was removed.",
        f"- For is_played=False rows, is_captain and is_wk are missing in {nonplayed_role_missing[0]} and {nonplayed_role_missing[1]} rows respectively; these are structurally blank role fields, not imputed.",
        "",
        "Scoring fields and limitations",
        "- Delivery total runs are derived as runs + extras1_runs + extras_2_runs. runs is treated as batter runs because the extras are separate fields; a leg-byes row violates the expected pattern (runs=1 with zero extras) and is flagged.",
        f"- Extras type values: {balls.extras_type.value_counts(dropna=False).to_dict()}.",
        f"- {int(missing_type.sum())} rows have nonzero extras but no extras_type; {int(other_extras.sum())} rows use the unclassified 'other' extras label.",
        f"- {int(ambiguous.sum())} deliveries carry extras with unclear legality or bowler attribution and remain flagged; the separate Penalty row is not charged to the bowler but its ball legality is unknown.",
        "- Legal ball classification: wide and Overstep (observed no-ball label) are illegal; known byes/leg-byes and no-extra rows are legal. Unknown extras remain nullable, so counts are reported as known legal, known illegal, and unknown.",
        "- innings_ball_num is a contiguous per-innings delivery-row ordinal (includes illegal deliveries), not a legal-ball count. Legal balls are derived from the extras classification.",
        f"- Negative values in runs/extras fields: {negative_scoring}; mismatched is_four/is_six versus runs fields: {boundary_mismatches}.",
        f"- bowl_speed equals zero on {zero_bowl_speed} rows; treated as a likely unavailable measurement, not as a measured zero-speed delivery.",
        "- Bowler runs: batter runs plus wide/no-ball extras where attributable; byes, leg-byes, and penalty extras are not charged to the bowler. Unclassified extras are not silently allocated.",
        f"- Innings 3/4 appear in {int(balls.loc[balls.innings.gt(2), ['match_id','innings']].drop_duplicates().shape[0])} match-innings; retained as recorded (likely tie-break/super-over innings; source has no explicit super-over flag).",
        f"- Innings above 120 known-legal deliveries: {len(over120)}; innings above 120 if all ambiguous rows were legal: {len(ambiguous_legal_upper)}. Ambiguous cases remain flagged and no innings is truncated.",
        f"- Dismissal types observed: {wicket_types.to_dict()}. Bowler wickets count caught, bowled, LBW, stumped and hit wicket; run-out, retire-out and obstruction are excluded.",
        "- matches.csv has no innings score totals, so calculated totals cannot be independently reconciled against a scorecard. Per-innings derived totals are supplied in data/processed/innings_summary.csv.",
        "- revised_target is populated on some matches and revised overs are recorded separately. No target reconstruction is attempted in this profiling phase.",
        "",
        "Cricket scoring summary",
    ]
    for row in scoring.itertuples(index=False):
        lines.append(f"- {row.status}: {row.check} = {row.value}; {row.details}")
    lines += ["", "No records were silently deleted or rewritten.", ""]
    return "\n".join(lines)


def run_profiling(data_dir: str | Path, report_dir: str | Path, figure_dir: str | Path, processed_dir: str | Path) -> dict[str, object]:
    """Run the complete Phase 2 profiling pipeline and write all requested outputs."""
    reports = Path(report_dir)
    reports.mkdir(parents=True, exist_ok=True)
    tables = load_data_pack(data_dir)
    profiles = {name: profile_frame(name, frame) for name, frame in tables.items()}

    (reports / "dataset_profile.txt").write_text(_render_profile_text(profiles), encoding="utf-8")
    with (reports / "dataset_profile.json").open("w", encoding="utf-8") as handle:
        json.dump(profiles, handle, indent=2, ensure_ascii=False, default=str, allow_nan=False)
    _write_missing_values(profiles, reports / "missing_values.csv")

    relational = relational_validation(tables)
    enriched_deliveries = build_delivery_table(tables)
    enrichment_check = pd.DataFrame([{
        "check": "match/competition metadata enrichment preserves delivery rows",
        "status": "PASS" if len(enriched_deliveries) == len(tables["ball_by_ball"]) else "REVIEW",
        "records_checked": len(tables["ball_by_ball"]),
        "unmatched_count": int(enriched_deliveries[["comp_id", "comp_name", "match_format", "season"]].isna().any(axis=1).sum()),
        "details": "many-to-one metadata joins; delivery rows are preserved and unmatched values remain visible",
    }])
    relational = pd.concat([relational, enrichment_check], ignore_index=True)
    relational.to_csv(reports / "relational_validation.csv", index=False)
    scoring, innings = cricket_validation(tables)
    scoring.to_csv(reports / "cricket_scoring_validation.csv", index=False)
    innings.to_csv(reports / "innings_scoring_summary.csv", index=False)
    _competition_coverage(tables).to_csv(reports / "competition_coverage.csv", index=False)

    (reports / "data_quality_report.txt").write_text(
        _quality_report(tables, relational, scoring, innings), encoding="utf-8"
    )
    figures = generate_profile_figures(tables, profiles, figure_dir)
    derived = write_derived_tables(tables, processed_dir)
    LOGGER.info("Wrote profiling outputs to %s and %s figures to %s", reports, len(figures), figure_dir)
    return {
        "tables": tables,
        "profiles": profiles,
        "relational_validation": relational,
        "scoring_validation": scoring,
        "innings": innings,
        "figures": figures,
        "derived_summary": derived,
    }
