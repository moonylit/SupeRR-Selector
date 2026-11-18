import pandas as pd
import pytest
import numpy as np

from src.models.final_strategy import _correct_batting_match_exposure, _validate_candidates, run_final_strategy


def test_final_strategy_writes_separate_eligible_top_tens(tmp_path):
    ids = list(range(1, 13))
    batting = pd.DataFrame({
        "competition_group": "IPL", "striker_id": ids,
        "player_name": [f"Batter {i}" for i in ids],
        "eligible_analytical_candidate": True,
        "overall_batting_pressure_score": [79, 79] + [80 - i for i in ids[2:]],
        "score_model_a_equal_domains": [79 - i for i in ids],
        "score_model_b_correlation_reduced": [78 - i for i in ids],
        "pressure_runs_per_ball": [1.4] * 12, "pressure_balls": [80] * 12,
        "pressure_matches": [6] * 12, "reliability_score": [70] * 12,
        "effective_pressure_exposure": [80.0] * 12,
        "rpb_personal_baseline_difference": [.1] * 12,
        "pressure_minus_baseline_rpb": [.1] * 12,
        "pressure_minus_baseline_ci_low": [-.05] * 12,
        "pressure_minus_baseline_ci_high": [.25] * 12,
        "baseline_history_balls": [100] * 12,
    })
    bowling = pd.DataFrame({
        "competition_group": "IPL", "bowler_id": ids,
        "player_name": [f"Bowler {i}" for i in ids],
        "eligible_analytical_candidate": True,
        "overall_bowling_pressure_score": [81, 81] + [82 - i for i in ids[2:]],
        "score_model_a_equal_weight": [81 - i for i in ids],
        "score_model_b_correlation_adjusted": [80 - i for i in ids],
        "score_model_c_shrinkage_adjusted": [82 - i for i in ids],
        "score_model_d_reliability_adjusted": [79 - i for i in ids],
        "pressure_legal_balls": [60] * 12, "pressure_matches": [5] * 12,
        "pressure_economy": [7.5] * 12, "relative_economy_improvement": [.2] * 12,
        "pressure_wickets_per_over": [1.0] * 12, "top10_selection_frequency": [.8] * 12,
        "statistical_reliability_score": [75] * 12, "baseline_coverage_fraction": [.7] * 12,
    })
    sensitivity_rows = []
    for definition in ["primary", "alternate"]:
        for pid in ids:
            sensitivity_rows.append({"analysis": "pressure definition", "definition": definition,
                                     "competition_group": "IPL", "striker_id": pid,
                                     "sensitivity_runs_per_ball": 1.5 - pid / 100})
    batting_delivery = pd.DataFrame([
        {"competition_group": "IPL", "striker_id": pid, "batting_order": (pid % 8) + 1,
         "match_id": match, "pressure_high_p90_walk_forward": True, "balls_faced_known": True}
        for pid in ids for match in range(1, 7)
    ])
    bowling_delivery = pd.DataFrame([
        {"competition_group": "IPL", "bowler_id": pid, "innings_phase": "death" if pid % 2 else "middle"}
        for pid in ids for _ in range(20)
    ])

    result = run_final_strategy(
        batting, bowling, pd.DataFrame(sensitivity_rows), batting_delivery, bowling_delivery,
        tmp_path / "reports", tmp_path / "figures", tmp_path / "processed",
    )

    assert result["batters"]["striker_id"].is_unique
    assert result["bowlers"]["bowler_id"].is_unique
    assert len(result["batters"]) == len(result["bowlers"]) == 10
    assert result["batters"]["final_rank"].tolist() == list(range(1, 11))
    assert result["bowlers"]["final_rank"].tolist() == list(range(1, 11))
    assert result["batters"]["striker_id"].iloc[:2].tolist() == [1, 2]
    assert result["bowlers"]["bowler_id"].iloc[:2].tolist() == [1, 2]
    assert "robustness" not in result["batters"]
    assert "robustness" not in result["bowlers"]
    assert (tmp_path / "reports" / "final_strategy_report.md").exists()
    assert (tmp_path / "figures" / "final_strategy" / "ipl_top10_shortlists.png").exists()
    assert (tmp_path / "processed" / "final_strategy_audit.json").exists()


def test_candidate_validation_rejects_missing_primary_sort_values():
    invalid = pd.DataFrame({
        "striker_id": [1], "overall_batting_pressure_score": [float("nan")],
        "reliability_score": [70.0], "pressure_balls": [80],
    })
    with pytest.raises(ValueError, match="missing/non-finite"):
        _validate_candidates(
            invalid, "striker_id", "overall_batting_pressure_score",
            "reliability_score", "pressure_balls", "test batting",
        )


def test_batting_match_audit_replaces_read_only_phase5_columns():
    def read_only(values, dtype):
        array = np.asarray(values, dtype=dtype)
        array.flags.writeable = False
        return pd.Series(array, copy=False)

    scores = pd.DataFrame({
        "competition_group": ["IPL"], "striker_id": [1],
        "pressure_matches": read_only([5], "int64"),
        "pressure_balls": [80], "baseline_history_balls": [100],
        "score_model_a_equal_domains": [60.0], "effective_pressure_exposure": [80.0],
        "reliability_score": read_only([70.0], "float64"),
        "score_model_c_shrinkage_adjusted": read_only([57.0], "float64"),
        "overall_batting_pressure_score": read_only([57.0], "float64"),
        "eligible_analytical_candidate": [True], "eligibility_status": ["eligible"],
    }, copy=False)
    deliveries = pd.DataFrame({
        "competition_group": ["IPL", "IPL"], "striker_id": [1, 1],
        "match_id": [10, 10], "pressure_high_p90_walk_forward": [True, True],
        "balls_faced_known": [True, False],
    })

    audited = _correct_batting_match_exposure(scores, deliveries)

    assert audited.loc[0, "phase5_pressure_matches_reported"] == 5
    assert audited.loc[0, "pressure_matches"] == 1
    assert audited.loc[0, "known_faced_pressure_matches"] == 1
    assert audited.loc[0, "reliability_score"] < 70
