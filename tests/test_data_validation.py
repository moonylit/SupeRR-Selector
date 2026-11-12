"""Synthetic unit tests for data loading and cricket scoring calculations."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.cricket_calculations import calculate_delivery_metrics, calculate_innings_summary
from src.data.loader import load_dataset, load_data_pack
from src.data.preprocessing import build_delivery_table
from src.data.validation import duplicate_key_summary, validate_identifiers


def _deliveries() -> pd.DataFrame:
    """Return a compact synthetic fixture covering observed scoring types."""
    return pd.DataFrame(
        {
            "match_id": [1] * 16,
            "innings": [1] * 16,
            "runs": [0, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "extras1_runs": [1, 1, 2, 1, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "extras_2_runs": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "extras_type": [
                "wide", "Overstep", "byes", "legbyes", "Penalty", "other",
                None, None, None, None, None, None, "Retire Out", "Retired Hurt",
                "Obstracting the field", "Obstructing the field",
            ],
            "is_wicket": [False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True],
            "dismissal_type": [
                None, None, None, None, None, None,
                "Bowled", "Caught", "Stump", "LBW", "Hit Wicket", "RunOut",
                "Retire Out", "Retired Hurt", "Obstracting the field", "Obstructing the field",
            ],
        }
    )


def test_load_dataset_reads_csv(tmp_path):
    source = tmp_path / "sample.csv"
    source.write_text("id,value\n1,a\n2,b\n", encoding="utf-8")
    frame = load_dataset(source)
    assert frame.to_dict("records") == [{"id": 1, "value": "a"}, {"id": 2, "value": "b"}]


def test_missing_dataset_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_dataset(tmp_path / "missing.csv")
    with pytest.raises(FileNotFoundError, match="ball_by_ball.csv"):
        load_data_pack(tmp_path)


def test_duplicate_identifier_detection_does_not_drop_rows():
    frame = pd.DataFrame({"match_id": [1, 1, 2], "team_id": [7, 7, 8]})
    assert duplicate_key_summary(frame, ["match_id", "team_id"]) == {
        "duplicate_rows": 2,
        "duplicate_keys": 1,
    }
    assert len(frame) == 3


def test_foreign_key_validation_reports_unmatched_ids():
    result = validate_identifiers(
        pd.DataFrame({"comp_id": [1, 2, 2]}),
        pd.DataFrame({"competition_id": [1]}),
        ["comp_id"],
        ["competition_id"],
    )
    assert result["child_distinct_keys"] == 2
    assert result["unmatched_distinct_keys"] == 1
    assert result["unmatched_key_sample"] == [{"key_0": 2}]


def test_metadata_enrichment_preserves_unmatched_delivery_rows():
    tables = {
        "ball_by_ball": pd.DataFrame({"match_id": [1, 2], "runs": [0, 1]}),
        "matches": pd.DataFrame({
            "match_id": [1], "comp_id": [10], "match_date": ["2024-01-01"],
            "home_team_id": [3], "away_team_id": [4],
        }),
        "competitions": pd.DataFrame({
            "comp_id": [10], "comp_name": ["IPL 2024"], "match_format": ["T20"], "season": ["2024"],
        }),
    }
    enriched = build_delivery_table(tables)
    assert len(enriched) == 2
    assert enriched.loc[enriched.match_id.eq(2), "comp_id"].isna().all()


def test_legal_delivery_classification_and_extras():
    metrics = calculate_delivery_metrics(_deliveries())
    legal = metrics["legal_delivery_calculated"]
    assert legal.iloc[:4].tolist() == [False, False, True, True]
    assert legal.iloc[4:6].isna().all()
    assert legal.iloc[6:].all()
    assert metrics.loc[0, "extras_calculated"] == 1
    assert metrics.loc[2, "total_runs_calculated"] == 2
    assert metrics.loc[4, "total_runs_calculated"] == 5


def test_bowler_runs_exclude_byes_legbyes_and_penalty():
    metrics = calculate_delivery_metrics(_deliveries())
    assert metrics["bowler_runs_conceded_known"].tolist()[:6] == [1, 5, 0, 0, 0, 1]
    assert metrics.loc[5, "bowler_runs_attribution_unknown"]
    assert not metrics.loc[4, "bowler_runs_attribution_unknown"]


def test_only_bowler_attributable_dismissals_are_credited():
    metrics = calculate_delivery_metrics(_deliveries())
    assert metrics["bowler_wicket_calculated"].tolist() == [
        False, False, False, False, False, False, True, True, True, True, True, False,
        False, False, False, False,
    ]


def test_innings_run_reconciliation_uses_batter_runs_plus_extras():
    summary = calculate_innings_summary(_deliveries()).iloc[0]
    assert summary["match_id"] == 1
    assert summary["innings"] == 1
    assert summary["deliveries"] == 16
    assert summary["batter_runs"] == 5
    assert summary["extras"] == 11
    assert summary["total_runs"] == 16
    assert summary["wickets"] == 10
    assert summary["bowler_attributable_wickets"] == 5


def test_calculation_fails_clearly_when_schema_is_incomplete():
    with pytest.raises(ValueError, match="Missing required scoring columns"):
        calculate_delivery_metrics(pd.DataFrame({"runs": [1]}))
