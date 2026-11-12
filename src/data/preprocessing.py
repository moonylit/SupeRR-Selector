"""Non-destructive, reproducible derived-table builders."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data.cricket_calculations import calculate_innings_summary

LOGGER = logging.getLogger(__name__)


def build_delivery_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Attach match and competition metadata to all deliveries, without row loss."""
    balls = tables["ball_by_ball"].copy()
    matches = tables["matches"][
        ["match_id", "comp_id", "match_date", "home_team_id", "away_team_id"]
    ]
    competitions = tables["competitions"][
        ["comp_id", "comp_name", "match_format", "season"]
    ]
    result = balls.merge(matches, on="match_id", how="left", validate="many_to_one", indicator="_match_merge")
    result = result.merge(
        competitions, on="comp_id", how="left", validate="many_to_one", indicator="_competition_merge"
    )
    if len(result) != len(balls):
        raise ValueError("Metadata enrichment changed the delivery row count")
    missing_matches = int(result["_match_merge"].ne("both").sum())
    missing_competitions = int(result["_competition_merge"].ne("both").sum())
    if missing_matches or missing_competitions:
        LOGGER.warning(
            "Preserving %s deliveries with unmatched matches and %s with unmatched competitions",
            missing_matches,
            missing_competitions,
        )
    return result.drop(columns=["_match_merge", "_competition_merge"])


def write_derived_tables(tables: dict[str, pd.DataFrame], output_dir: str | Path) -> Path:
    """Write a reproducible innings summary to the ignored processed-data area."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary = calculate_innings_summary(tables["ball_by_ball"])
    target = output_path / "innings_summary.csv"
    summary.to_csv(target, index=False)
    LOGGER.info("Wrote %s innings summaries to %s", len(summary), target)
    return target
