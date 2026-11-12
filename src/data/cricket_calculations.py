"""Cricket scoring calculations based on the inspected data-pack schema."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

REQUIRED_SCORING_COLUMNS = {
    "runs",
    "extras1_runs",
    "extras_2_runs",
    "extras_type",
    "is_wicket",
    "dismissal_type",
}
NON_LEGAL_EXTRAS = {"wide", "overstep"}
KNOWN_LEGAL_EXTRAS = {"byes", "legbyes"}
BOWLER_WICKET_DISMISSALS = {
    "bowled",
    "caught",
    "lbw",
    "stump",
    "hit wicket",
}
NON_BOWLER_WICKET_DISMISSALS = {
    "runout",
    "retire out",
    "retired hurt",
    "obstracting the field",
    "obstructing the field",
}


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required scoring columns: {', '.join(missing)}")


def _normalized_type(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.casefold()


def calculate_delivery_metrics(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Return row-level scoring metrics while preserving every source row.

    ``runs`` is treated as batter runs and the two extras fields are additive,
    matching the inspected data structure. Legal status is tri-state: wides
    and ``Overstep`` (the observed no-ball label) are illegal; byes, leg-byes,
    and no-extra rows are classifiable; unfamiliar/missing nonzero extras are
    left unknown for review.
    """
    _require_columns(deliveries, REQUIRED_SCORING_COLUMNS)
    result = deliveries.copy()
    runs = result["runs"].fillna(0)
    extra_one = result["extras1_runs"].fillna(0)
    extra_two = result["extras_2_runs"].fillna(0)
    extras_type = _normalized_type(result["extras_type"])
    extras_present = extra_one.ne(0) | extra_two.ne(0)

    result["batter_runs_calculated"] = runs
    result["extras_calculated"] = extra_one + extra_two
    result["total_runs_calculated"] = runs + extra_one + extra_two

    illegal = extras_type.isin(NON_LEGAL_EXTRAS)
    known_legal = (
        extras_type.isin(KNOWN_LEGAL_EXTRAS)
        | (extras_type.isna() & ~extras_present)
    )
    legal = pd.Series(pd.NA, index=result.index, dtype="boolean")
    legal.loc[illegal] = False
    legal.loc[known_legal] = True
    result["legal_delivery_calculated"] = legal

    known_bowler_extra = extras_type.isin(NON_LEGAL_EXTRAS)
    result["bowler_runs_conceded_known"] = runs + (extra_one + extra_two).where(
        known_bowler_extra, 0
    )
    uncertain_bowler_extras = extras_present & ~(
        known_bowler_extra | extras_type.isin(KNOWN_LEGAL_EXTRAS) | extras_type.eq("penalty")
    )
    result["bowler_runs_attribution_unknown"] = uncertain_bowler_extras

    dismissal = _normalized_type(result["dismissal_type"])
    wicket_flag = result["is_wicket"].fillna(False).astype(bool)
    result["bowler_wicket_calculated"] = wicket_flag & dismissal.isin(
        BOWLER_WICKET_DISMISSALS
    )
    known_dismissals = BOWLER_WICKET_DISMISSALS | NON_BOWLER_WICKET_DISMISSALS
    result["wicket_attribution_unknown"] = wicket_flag & (
        dismissal.isna() | ~dismissal.isin(known_dismissals)
    )
    return result


def calculate_innings_summary(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Aggregate total runs, extras, balls, and wickets for each innings."""
    _require_columns(deliveries, {"match_id", "innings", *REQUIRED_SCORING_COLUMNS})
    metrics = calculate_delivery_metrics(deliveries)
    metrics["_legal_yes"] = metrics["legal_delivery_calculated"].eq(True).fillna(False)
    metrics["_legal_unknown"] = metrics["legal_delivery_calculated"].isna()
    metrics["_wicket"] = metrics["is_wicket"].fillna(False).astype(bool)
    return (
        metrics.groupby(["match_id", "innings"], dropna=False)
        .agg(
            deliveries=("runs", "size"),
            batter_runs=("batter_runs_calculated", "sum"),
            extras=("extras_calculated", "sum"),
            total_runs=("total_runs_calculated", "sum"),
            known_legal_balls=("_legal_yes", "sum"),
            legality_unknown_deliveries=("_legal_unknown", "sum"),
            wickets=("_wicket", "sum"),
            bowler_attributable_wickets=("bowler_wicket_calculated", "sum"),
            bowler_extra_attribution_unknown=("bowler_runs_attribution_unknown", "sum"),
        )
        .reset_index()
    )
