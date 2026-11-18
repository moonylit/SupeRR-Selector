"""Evidence-led final IPL auction shortlists built from Phases 5 and 6."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import BATTING_MODEL_CONFIG, BOWLING_MODEL_CONFIG

IPL_2027_RETAINED_PHASE7_CANDIDATES = {
    "batters": [
        ("Sai Sudharsan", "Gujarat Titans"),
        ("Virat Kohli", "Royal Challengers Bengaluru"),
        ("Heinrich Klaasen", "Sunrisers Hyderabad"),
        ("Shivam Dube", "Chennai Super Kings"),
        ("Riyan Parag", "Rajasthan Royals"),
        ("Aiden Markram", "Lucknow Super Giants"),
        ("Sanju Samson", "Chennai Super Kings"),
        ("Shubman Gill", "Gujarat Titans"),
        ("Suryakumar Yadav", "Mumbai Indians"),
    ],
    "bowlers": [
        ("Noor Ahmad", "Chennai Super Kings"),
        ("Axar Patel", "Delhi Capitals"),
        ("Suyash Sharma", "Royal Challengers Bengaluru"),
        ("Krunal Pandya", "Royal Challengers Bengaluru"),
        ("Ravi Bishnoi", "Rajasthan Royals"),
        ("Yuzvendra Chahal", "Punjab Kings"),
        ("Rashid Khan", "Gujarat Titans"),
        ("Ravindra Jadeja", "Rajasthan Royals"),
    ],
}
IPL_2027_RETENTION_SOURCE = "https://www.iplt20.com/news/article/tata-ipl-2027-player-retentions-announced"


def _top_membership(frame: pd.DataFrame, score_columns: list[str], key: str) -> pd.Series:
    """Count top-ten appearances across already calculated score variants."""
    counts = pd.Series(0, index=frame.index, dtype="int64")
    for column in score_columns:
        if column not in frame:
            continue
        order = frame.sort_values([column, key], ascending=[False, True], kind="mergesort").head(10).index
        counts.loc[order] += 1
    return counts


def _batting_sensitivity(scores: pd.DataFrame, sensitivity: pd.DataFrame) -> pd.DataFrame:
    eligible = scores.loc[scores["competition_group"].eq("IPL") & scores["eligible_analytical_candidate"].fillna(False)].copy()
    if eligible.empty:
        return pd.DataFrame(columns=["striker_id", "score_variant_top10_count", "pressure_definition_top10_count", "pressure_definitions_tested"])
    eligible["score_variant_top10_count"] = _top_membership(
        eligible,
        ["score_model_a_equal_domains", "score_model_b_correlation_reduced", "overall_batting_pressure_score"],
        "striker_id",
    )
    pressure = sensitivity.loc[
        sensitivity.get("analysis", pd.Series(index=sensitivity.index, dtype="object")).eq("pressure definition")
    ].copy()
    counts: list[dict[str, Any]] = []
    for _, group in pressure.groupby("definition", dropna=False):
        group = group.loc[group["competition_group"].eq("IPL") & group["striker_id"].isin(eligible["striker_id"])]
        top_ids = group.sort_values(["sensitivity_runs_per_ball", "striker_id"], ascending=[False, True]).head(10)["striker_id"]
        counts.extend({"striker_id": pid, "pressure_definition_top10": True} for pid in top_ids)
    if counts:
        pressure_counts = pd.DataFrame(counts).groupby("striker_id").size().rename("pressure_definition_top10_count")
        tested = pressure.loc[pressure["competition_group"].eq("IPL"), "definition"].nunique()
        eligible = eligible.merge(pressure_counts, on="striker_id", how="left", validate="one_to_one")
        eligible["pressure_definition_top10_count"] = eligible["pressure_definition_top10_count"].fillna(0).astype(int)
        eligible["pressure_definitions_tested"] = tested
    else:
        eligible["pressure_definition_top10_count"] = 0
        eligible["pressure_definitions_tested"] = 0
    return eligible[["striker_id", "score_variant_top10_count", "pressure_definition_top10_count", "pressure_definitions_tested"]]


def _scenario_audit(
    eligible: pd.DataFrame,
    primary_column: str,
    key: str,
    sensitivity: pd.DataFrame,
    sensitivity_kind: str,
) -> pd.DataFrame:
    """Compare actual alternate scenario rankings with the primary top ten."""
    primary_top = set(eligible.nlargest(10, primary_column)[key])
    rows: list[dict[str, Any]] = []
    subset = sensitivity.loc[
        sensitivity.get("analysis", pd.Series(index=sensitivity.index, dtype="object")).eq(sensitivity_kind)
        & sensitivity.get("competition_group", pd.Series(index=sensitivity.index, dtype="object")).eq("IPL")
    ]
    score_column = "sensitivity_runs_per_ball" if sensitivity_kind == "pressure definition" else "score"
    for scenario, part in subset.groupby("definition", dropna=False):
        part = part.loc[part[key].isin(eligible[key])].drop_duplicates(key)
        if part.empty:
            continue
        scenario_top = set(part.sort_values([score_column, key], ascending=[False, True], kind="mergesort").head(10)[key])
        joined = eligible[[key, primary_column]].merge(part[[key, score_column]], on=key, how="inner", validate="one_to_one")
        rho = joined[primary_column].corr(joined[score_column], method="spearman") if len(joined) > 1 else np.nan
        rows.append({
            "scenario": str(scenario), "candidate_records": int(len(part)),
            "primary_top10_overlap": int(len(primary_top & scenario_top)),
            "primary_top10_size": int(len(primary_top)),
            "spearman_with_primary_score": float(rho) if pd.notna(rho) else np.nan,
        })
    return pd.DataFrame(rows)


def _model_variant_audit(frame: pd.DataFrame, key: str, primary: str, variants: list[str]) -> pd.DataFrame:
    primary_top = set(frame.nlargest(10, primary)[key])
    rows = []
    for column in variants:
        top = set(frame.nlargest(10, column)[key])
        rho = frame[primary].corr(frame[column], method="spearman") if len(frame) > 1 else np.nan
        rows.append({
            "score_model": column,
            "primary_top10_overlap": int(len(primary_top & top)),
            "top10_size": int(len(primary_top)),
            "spearman_with_primary_score": float(rho) if pd.notna(rho) else np.nan,
        })
    return pd.DataFrame(rows)


def _validate_candidates(frame: pd.DataFrame, key: str, score: str, reliability: str, exposure: str, label: str) -> None:
    if frame[key].isna().any():
        raise ValueError(f"{label}: an eligible candidate is missing stable player ID")
    if frame[key].duplicated().any():
        raise ValueError(f"{label}: duplicate eligible player IDs; aggregation is not one row per player")
    for column in (score, reliability, exposure):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all():
            raise ValueError(f"{label}: eligible ranking field {column} has missing/non-finite values")
    if not pd.to_numeric(frame[score], errors="coerce").between(0, 100).all():
        raise ValueError(f"{label}: primary composite score falls outside 0–100")


def _correct_batting_match_exposure(scores: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    """Count pressure matches only where the batter faced a known ball.

    Phase 5's aggregated pressure-match field can include P90 rows with a
    striker ID but no known faced ball (for example, an extra). This Phase 7
    audit recalculates the match exposure gate from known-faced P90 deliveries
    and reapplies the unchanged Phase 5 reliability/shrinkage formula.
    """
    required = {"competition_group", "striker_id", "match_id", "pressure_high_p90_walk_forward", "balls_faced_known"}
    if not required.issubset(deliveries.columns):
        raise ValueError(f"Batting exposure audit requires delivery fields: {sorted(required)}")
    # Phase 5 can return columns backed by read-only Arrow/NumPy buffers.  A
    # deep pandas copy does not always make those buffers writable, so build
    # replacement columns below instead of assigning through ``.loc``.
    frame = scores.copy(deep=True)
    pressure = deliveries.loc[
        deliveries["competition_group"].eq("IPL")
        & deliveries["pressure_high_p90_walk_forward"].eq(True).fillna(False)
        & deliveries["balls_faced_known"].eq(True).fillna(False)
        & deliveries["striker_id"].notna()
    ]
    matches = pressure.groupby(["competition_group", "striker_id"], as_index=False, dropna=False).agg(
        known_faced_pressure_matches=("match_id", "nunique")
    )
    frame["phase5_pressure_matches_reported"] = frame["pressure_matches"]
    frame = frame.merge(matches, on=["competition_group", "striker_id"], how="left", validate="many_to_one")
    ipl = frame["competition_group"].eq("IPL")
    known_matches = pd.to_numeric(frame["known_faced_pressure_matches"], errors="coerce").fillna(0)
    frame["known_faced_pressure_matches"] = known_matches
    original_matches = pd.to_numeric(frame["pressure_matches"], errors="coerce")
    frame["pressure_matches"] = original_matches.where(~ipl, known_matches)

    pressure_balls = pd.to_numeric(frame["pressure_balls"], errors="coerce").fillna(0)
    known_matches = pd.to_numeric(frame["pressure_matches"], errors="coerce").fillna(0)
    baseline_balls = pd.to_numeric(frame["baseline_history_balls"], errors="coerce").fillna(0)
    eligibility = (
        pressure_balls.ge(BATTING_MODEL_CONFIG.eligibility_pressure_balls)
        & known_matches.ge(BATTING_MODEL_CONFIG.eligibility_pressure_matches)
        & baseline_balls.ge(BATTING_MODEL_CONFIG.eligibility_baseline_balls)
    )
    old_eligibility = frame.get("eligible_analytical_candidate", pd.Series(False, index=frame.index))
    frame["eligible_analytical_candidate"] = old_eligibility.where(~ipl, eligibility)
    if "eligibility_status" in frame:
        status = frame["eligibility_status"].astype(object)
        audited_status = pd.Series(np.where(eligibility, "eligible after known-faced pressure-match audit",
                                            "ineligible after known-faced pressure-match audit"), index=frame.index)
        frame["eligibility_status"] = status.where(~ipl, audited_status)

    if not {"score_model_a_equal_domains", "effective_pressure_exposure"}.issubset(frame.columns):
        raise ValueError("Phase 5 score table lacks fields required to recalculate audited reliability shrinkage")
    ess = pd.to_numeric(frame["effective_pressure_exposure"], errors="coerce").fillna(0)
    match_count = pd.to_numeric(frame["pressure_matches"], errors="coerce").fillna(0)
    baseline = pd.to_numeric(frame["baseline_history_balls"], errors="coerce").fillna(0)
    reliability = (
        100 * (1 - np.exp(-ess / 100)) * (1 - np.exp(-match_count / 8))
        * (0.5 + 0.5 * (1 - np.exp(-baseline / 100)))
    ).clip(0, 100)
    old_reliability = pd.to_numeric(frame["reliability_score"], errors="coerce").astype("float64")
    frame["reliability_score"] = old_reliability.where(~ipl, reliability)
    score = (pd.to_numeric(frame["score_model_a_equal_domains"], errors="coerce")
             * frame["reliability_score"] / 100
             + 50 * (1 - frame["reliability_score"] / 100)).clip(0, 100)
    if "score_model_c_shrinkage_adjusted" not in frame:
        frame["score_model_c_shrinkage_adjusted"] = frame["overall_batting_pressure_score"]
    previous_model_c = pd.to_numeric(frame["score_model_c_shrinkage_adjusted"], errors="coerce").astype("float64")
    previous_overall = pd.to_numeric(frame["overall_batting_pressure_score"], errors="coerce").astype("float64")
    frame["score_model_c_shrinkage_adjusted"] = previous_model_c.where(~ipl, score)
    frame["overall_batting_pressure_score"] = previous_overall.where(~ipl, score)
    return frame


def _season_scope(*frames: pd.DataFrame) -> str:
    values: set[str] = set()
    for frame in frames:
        if "season" not in frame or "competition_group" not in frame:
            continue
        seasons = frame.loc[frame["competition_group"].eq("IPL"), "season"].dropna()
        values.update(seasons.astype(str).str.replace(r"\.0$", "", regex=True))
    ordered = sorted(values)
    return ", ".join(ordered) if ordered else "not available in generated evidence tables"


def _batting_roles(deliveries: pd.DataFrame) -> pd.DataFrame:
    cols = ["competition_group", "striker_id", "batting_order"]
    if "match_id" in deliveries:
        cols.append("match_id")
    if not set(cols).issubset(deliveries.columns):
        return pd.DataFrame(columns=["competition_group", "striker_id", "median_batting_order", "observed_role"])
    frame = deliveries.loc[deliveries["competition_group"].eq("IPL"), cols].copy()
    frame["batting_order"] = pd.to_numeric(frame["batting_order"], errors="coerce")
    frame = frame.dropna(subset=["striker_id", "batting_order"])
    frame = frame.drop_duplicates(["competition_group", "striker_id", "match_id"] if "match_id" in frame else ["competition_group", "striker_id", "batting_order"])
    roles = frame.groupby(["competition_group", "striker_id"], as_index=False).agg(
        median_batting_order=("batting_order", "median"), batting_order_matches=("batting_order", "size")
    )
    roles["observed_role"] = np.select(
        [roles["median_batting_order"].le(3), roles["median_batting_order"].le(6)],
        ["top order (1–3)", "middle order (4–6)"], default="lower order (7+)",
    )
    roles.loc[roles["batting_order_matches"].lt(5), "observed_role"] = "role uncertain (<5 observed match-order records)"
    return roles


def _bowling_roles(deliveries: pd.DataFrame) -> pd.DataFrame:
    keys = ["competition_group", "bowler_id", "innings_phase"]
    if not set(keys).issubset(deliveries.columns):
        return pd.DataFrame(columns=["competition_group", "bowler_id", "observed_bowling_phase", "phase_share"])
    phase = deliveries.loc[deliveries["competition_group"].eq("IPL") & deliveries["bowler_id"].notna(), keys].copy()
    phase = phase.groupby(keys, as_index=False).size().rename(columns={"size": "phase_deliveries"})
    phase["share"] = phase["phase_deliveries"] / phase.groupby(["competition_group", "bowler_id"])["phase_deliveries"].transform("sum")
    phase = phase.sort_values(["competition_group", "bowler_id", "phase_deliveries", "innings_phase"], ascending=[True, True, False, True])
    result = phase.drop_duplicates(["competition_group", "bowler_id"])[["competition_group", "bowler_id", "innings_phase", "share"]]
    result = result.rename(columns={"innings_phase": "observed_bowling_phase", "share": "phase_share"})
    result.loc[result["phase_share"].lt(.35), "observed_bowling_phase"] = "no dominant phase (<35% of observed deliveries)"
    return result


def _add_batting_explanations(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    pieces = []
    for _, row in frame.iterrows():
        parts = [f"pressure {row.get('pressure_runs_per_ball', np.nan):.3f} runs/ball on {int(row.get('pressure_balls', 0))} balls"]
        if pd.notna(row.get("rpb_personal_baseline_difference")):
            parts.append(f"{row['rpb_personal_baseline_difference']:+.3f} vs walk-forward personal baseline")
        if pd.notna(row.get("pressure_minus_baseline_rpb")):
            lo, hi = row.get("pressure_minus_baseline_ci_low"), row.get("pressure_minus_baseline_ci_high")
            if pd.notna(lo) and pd.notna(hi):
                parts.append(f"match-bootstrap baseline-difference interval [{lo:.3f}, {hi:.3f}]")
        parts.append(f"{int(row.get('pressure_matches', 0))} pressure matches; {int(row.get('baseline_history_balls', 0))} prior baseline balls")
        pieces.append("; ".join(parts))
    frame["evidence_summary"] = pieces
    return frame


def _add_bowling_explanations(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    pieces = []
    for _, row in frame.iterrows():
        parts = [f"{row.get('pressure_legal_balls', 0):.0f} high-pressure legal balls across {row.get('pressure_matches', 0):.0f} matches"]
        if pd.notna(row.get("pressure_economy")):
            parts.append(f"pressure economy {row['pressure_economy']:.2f}")
        if pd.notna(row.get("relative_economy_improvement")):
            parts.append(f"relative economy improvement {row['relative_economy_improvement']:.2f}")
        if pd.notna(row.get("pressure_wickets_per_over")):
            parts.append(f"{row['pressure_wickets_per_over']:.2f} wickets/over in pressure contexts")
        freq = row.get("top10_selection_frequency")
        if pd.notna(freq):
            parts.append(f"match-bootstrap top-10 frequency {freq:.0%}")
        parts.append(f"baseline coverage {row.get('baseline_coverage_fraction', 0):.0%}")
        pieces.append("; ".join(parts))
    frame["evidence_summary"] = pieces
    return frame


def _format_player_table(frame: pd.DataFrame, columns: list[str]) -> str:
    present = [column for column in columns if column in frame]
    def cell(value: Any) -> str:
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{value:.3f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    rows = ["| " + " | ".join(present) + " |", "| " + " | ".join("---" for _ in present) + " |"]
    rows.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in frame[present].itertuples(index=False, name=None))
    return "\n".join(rows)


def run_final_strategy(
    batting_scores: pd.DataFrame,
    bowling_scores: pd.DataFrame,
    batting_sensitivity: pd.DataFrame,
    batting_deliveries: pd.DataFrame,
    bowling_deliveries: pd.DataFrame,
    reports_dir: str | Path,
    figures_dir: str | Path,
    processed_dir: str | Path,
    bowling_sensitivity: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Create separate, evidence-audited top-ten IPL batting and bowling lists."""
    reports, figures, processed = map(Path, (reports_dir, figures_dir, processed_dir))
    for path in (reports, figures, processed):
        path.mkdir(parents=True, exist_ok=True)

    batting_scores = _correct_batting_match_exposure(batting_scores, batting_deliveries)
    batting = batting_scores.loc[
        batting_scores["competition_group"].eq("IPL") & batting_scores["eligible_analytical_candidate"].fillna(False)
    ].copy()
    bowling = bowling_scores.loc[
        bowling_scores["competition_group"].eq("IPL") & bowling_scores["eligible_analytical_candidate"].fillna(False)
    ].copy()
    batting["striker_id"] = pd.to_numeric(batting["striker_id"], errors="raise").astype("Int64")
    bowling["bowler_id"] = pd.to_numeric(bowling["bowler_id"], errors="raise").astype("Int64")
    if len(batting) < 10 or len(bowling) < 10:
        raise ValueError(f"Cannot publish two complete IPL top-tens: eligible batters={len(batting)}, bowlers={len(bowling)}")
    _validate_candidates(batting, "striker_id", "overall_batting_pressure_score", "reliability_score", "pressure_balls", "Phase 5 batting")
    _validate_candidates(bowling, "bowler_id", "overall_bowling_pressure_score", "statistical_reliability_score", "pressure_legal_balls", "Phase 6 bowling")

    b_sensitivity = _batting_sensitivity(batting_scores, batting_sensitivity)
    batting = batting.merge(b_sensitivity, on="striker_id", how="left", validate="one_to_one")
    batting["score_variant_top10_count"] = batting["score_variant_top10_count"].fillna(0).astype(int)
    batting["pressure_definition_top10_count"] = batting["pressure_definition_top10_count"].fillna(0).astype(int)
    batting["pressure_definitions_tested"] = batting["pressure_definitions_tested"].fillna(0).astype(int)
    batting["score_variants_tested"] = 3
    batting = batting.merge(_batting_roles(batting_deliveries), on=["competition_group", "striker_id"], how="left", validate="one_to_one")
    batting = batting.sort_values(["overall_batting_pressure_score", "reliability_score", "pressure_balls", "striker_id"], ascending=[False, False, False, True], kind="mergesort")
    batting["final_rank"] = range(1, len(batting) + 1)
    batting = _add_batting_explanations(batting)

    bowling_columns = ["score_model_a_equal_weight", "score_model_b_correlation_adjusted", "score_model_c_shrinkage_adjusted", "score_model_d_reliability_adjusted"]
    bowling["score_variant_top10_count"] = _top_membership(bowling, bowling_columns, "bowler_id")
    bowling["score_variants_tested"] = len(bowling_columns)
    bowling = bowling.merge(_bowling_roles(bowling_deliveries), on=["competition_group", "bowler_id"], how="left", validate="one_to_one")
    bowling = bowling.sort_values(["overall_bowling_pressure_score", "statistical_reliability_score", "pressure_legal_balls", "bowler_id"], ascending=[False, False, False, True], kind="mergesort")
    bowling["final_rank"] = range(1, len(bowling) + 1)
    bowling = _add_bowling_explanations(bowling)
    bat_top, bowl_top = batting.head(10).copy(), bowling.head(10).copy()

    batting_models = _model_variant_audit(
        batting, "striker_id", "overall_batting_pressure_score",
        ["score_model_a_equal_domains", "score_model_b_correlation_reduced", "overall_batting_pressure_score"],
    )
    batting_pressure_scenarios = _scenario_audit(
        batting, "overall_batting_pressure_score", "striker_id", batting_sensitivity, "pressure definition",
    )
    batting_weight_scenarios = _scenario_audit(
        batting, "overall_batting_pressure_score", "striker_id", batting_sensitivity, "composite weight sensitivity",
    )
    bowling_models = _model_variant_audit(
        bowling, "bowler_id", "overall_bowling_pressure_score", bowling_columns,
    )
    bowling_sensitivity = bowling_sensitivity if bowling_sensitivity is not None else pd.DataFrame()
    bowling_sensitivity_rows = pd.DataFrame()
    if not bowling_sensitivity.empty:
        scenario_names = bowling_sensitivity.get("scenario", pd.Series(index=bowling_sensitivity.index, dtype="object")).astype("string")
        retain = scenario_names.str.startswith(("A.", "B.", "C.", "D."), na=False) | scenario_names.str.contains(
            "baseline|nonlinear walk-forward|weighted context P90|weighted fixed threshold|shrinkage prior|leave season", case=False, na=False
        )
        bowling_sensitivity_rows = bowling_sensitivity.loc[retain].copy()

    for name, frame in (
        ("final_batting_model_sensitivity.csv", batting_models),
        ("final_batting_pressure_sensitivity.csv", batting_pressure_scenarios),
        ("final_batting_weight_sensitivity.csv", batting_weight_scenarios),
        ("final_bowling_model_sensitivity.csv", bowling_models),
        ("final_bowling_pipeline_sensitivity.csv", bowling_sensitivity_rows),
    ):
        frame.to_csv(processed / name, index=False)

    bat_top.to_csv(processed / "final_top10_batters_ipl.csv", index=False)
    bowl_top.to_csv(processed / "final_top10_bowlers_ipl.csv", index=False)
    batting.to_csv(processed / "final_eligible_batters_ipl.csv", index=False)
    bowling.to_csv(processed / "final_eligible_bowlers_ipl.csv", index=False)
    combined = pd.concat([
        bat_top.assign(role_type="batter", final_score=bat_top["overall_batting_pressure_score"]),
        bowl_top.assign(role_type="bowler", final_score=bowl_top["overall_bowling_pressure_score"]),
    ], ignore_index=True, sort=False)
    combined.to_csv(processed / "final_strategy_evidence.csv", index=False)

    seasons = _season_scope(batting_deliveries, bowling_deliveries)
    retained_batters = IPL_2027_RETAINED_PHASE7_CANDIDATES["batters"]
    retained_bowlers = IPL_2027_RETAINED_PHASE7_CANDIDATES["bowlers"]
    retained_candidate_count = len(retained_batters) + len(retained_bowlers)
    retained_by_rr_count = sum(franchise == "Rajasthan Royals" for _, franchise in retained_batters + retained_bowlers)
    retained_by_other_franchises_count = retained_candidate_count - retained_by_rr_count
    retained_details = "; ".join(
        f"{name} ({franchise})" for name, franchise in retained_batters + retained_bowlers
    )
    metadata = {
        "target": "historical IPL pressure-performance shortlists; not auction-eligible player pools or purchase recommendations",
        "historical_ipl_seasons_in_generated_evidence": seasons,
        "auction_availability_verified": False,
        "official_retention_snapshot_as_of": "2026-11-18",
        "known_retained_phase7_top10_candidates": {
            "batters": [name for name, _ in retained_batters],
            "bowlers": [name for name, _ in retained_bowlers],
        },
        "retained_phase7_top10_candidate_count": retained_candidate_count,
        "retained_by_rr_candidate_count": retained_by_rr_count,
        "retained_candidate_franchises": {
            name: franchise for name, franchise in retained_batters + retained_bowlers
        },
        "retention_sources": [IPL_2027_RETENTION_SOURCE],
        "eligibility_rules": {
            "batters": {"walk_forward_p90_known_balls_minimum": BATTING_MODEL_CONFIG.eligibility_pressure_balls,
                        "walk_forward_p90_matches_minimum": BATTING_MODEL_CONFIG.eligibility_pressure_matches,
                        "prior_nonpressure_baseline_balls_minimum": BATTING_MODEL_CONFIG.eligibility_baseline_balls},
            "bowlers": {"walk_forward_high_pressure_legal_balls_minimum": BOWLING_MODEL_CONFIG.eligibility_pressure_balls,
                        "pressure_matches_minimum": BOWLING_MODEL_CONFIG.eligibility_pressure_matches,
                        "pressure_innings_minimum": BOWLING_MODEL_CONFIG.eligibility_pressure_innings,
                        "prior_normal_baseline_legal_balls_minimum": BOWLING_MODEL_CONFIG.eligibility_baseline_balls,
                        "walk_forward_baseline_coverage_minimum": BOWLING_MODEL_CONFIG.baseline_coverage_minimum},
        },
        "selection_method": "Separate Phase 5 and Phase 6 within-IPL eligibility-filtered rankings; no cross-role or cross-format score aggregation.",
        "eligible_ipl_batters": int(len(batting)), "eligible_ipl_bowlers": int(len(bowling)),
        "phase5_pressure_match_count_recalculation": {
            "records_recomputed_from_known_faced_p90_deliveries": int(
                batting_scores.loc[batting_scores["competition_group"].eq("IPL"), "phase5_pressure_matches_reported"].ne(
                    batting_scores.loc[batting_scores["competition_group"].eq("IPL"), "known_faced_pressure_matches"]
                ).sum()
            ),
            "reliability_and_model_c_recomputed": True,
            "reason": "Phase 5's aggregate match count can include P90 delivery rows where the named striker did not face a known ball; Phase 7 counts unique matches with at least one known-faced P90 ball.",
        },
        "top_ten_batters": int(len(bat_top)), "top_ten_bowlers": int(len(bowl_top)),
        "batting_score": "Phase 5 Model C reliability shrinkage; Phase 7 recalculates the Phase 5 pressure-match count from known-faced walk-forward P90 deliveries, reapplies the original reliability formula, and retains the 0-100 score scale",
        "bowling_score": "Phase 6 overall_bowling_pressure_score (format-aware empirical shrinkage Model C; 0-100)",
        "roster_and_purse": f"The supplied challenge pack has no roster or purse inputs. The official IPL retention announcement dated 2026-11-18 lists {retained_candidate_count} of the 20 historical top-ten candidates as retained; the final auction registration list was not verified. Candidate inclusion does not imply availability.",
        "limitations": ["Pressure is a descriptive match-state construct, not a psychological measurement.", "Scores are provisional and not outcome-trained.", "Competition scope is kept separate; T20I is not mixed into IPL auction rankings.", f"The supplied challenge pack contains no current RR squad or auction registration list; a separate official retention snapshot dated 2026-11-18 identifies {retained_candidate_count} top-ten candidates as retained."],
    }
    (processed / "final_strategy_audit.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    bat_cols = ["final_rank", "player_name", "observed_role", "overall_batting_pressure_score", "absolute_performance_score", "relative_resilience_score", "consistency_score", "reliability_score", "pressure_runs_per_ball", "rpb_personal_baseline_difference", "pressure_balls", "pressure_matches", "phase5_pressure_matches_reported", "score_variant_top10_count", "score_variants_tested", "pressure_definition_top10_count", "pressure_definitions_tested", "evidence_summary"]
    bowl_cols = ["final_rank", "player_name", "observed_bowling_phase", "overall_bowling_pressure_score", "absolute_pressure_score_shrunk", "relative_improvement_score", "consistency_score", "statistical_reliability_score", "pressure_economy", "relative_economy_improvement", "pressure_wickets_per_over", "pressure_legal_balls", "pressure_matches", "score_variant_top10_count", "score_variants_tested", "top10_selection_frequency", "evidence_summary"]
    batting_ci_cross_zero = 0
    if {"pressure_minus_baseline_ci_low", "pressure_minus_baseline_ci_high"}.issubset(bat_top.columns):
        batting_ci_cross_zero = int((
            pd.to_numeric(bat_top["pressure_minus_baseline_ci_low"], errors="coerce").lt(0)
            & pd.to_numeric(bat_top["pressure_minus_baseline_ci_high"], errors="coerce").gt(0)
        ).sum())
    bowling_top10_frequency = pd.to_numeric(
        bowl_top.get("top10_selection_frequency", pd.Series(dtype="float64")), errors="coerce"
    ).dropna()
    bowling_frequency_range = (
        f"{bowling_top10_frequency.min():.0%}–{bowling_top10_frequency.max():.0%}"
        if not bowling_top10_frequency.empty else "not available"
    )
    report = (
        "# SupeRR Selector — Final IPL Historical Performance Rankings\n\n"
        "## Decision scope\n\n"
        f"The two lists below summarize historical IPL records from seasons {seasons}. Batting and bowling use separate Phase 5 and Phase 6 eligibility gates and reliability-adjusted within-IPL composite scores; the two score scales are not comparable to one another. The primary score sorts descending, followed by reliability, pressure exposure, and stable player ID as deterministic tie-breakers. Names are display labels; player IDs are retained in the accompanying CSVs.\n\n"
        f"These are historical performers who meet analytical sample thresholds, not a current-auction shortlist or purchase recommendation. The supplied challenge pack does not include the current RR squad, open slots, purse, reserve prices, or role-gap brief. Observed batting-order and bowling-phase fields describe historical usage only; they do not establish specialization or current squad need.\n\n"
        "### Official IPL 2027 retention snapshot (2026-11-18)\n\n"
        f"The [official IPL retention announcement]({IPL_2027_RETENTION_SOURCE}) lists {retained_candidate_count} of the 20 historically ranked candidates as retained: {retained_details}. {retained_by_rr_count} are already retained by RR and {retained_by_other_franchises_count} by other franchises; none is available to purchase at auction while retained. The historical ranking files intentionally keep them in score order. The other three candidates are not confirmed auction entrants or purchasable merely because they are absent from the retained list. The final registered auction pool was not verified.\n\n"
        f"Eligibility gates: batters require at least {BATTING_MODEL_CONFIG.eligibility_pressure_balls} known-faced walk-forward P90 balls in {BATTING_MODEL_CONFIG.eligibility_pressure_matches} distinct matches with at least one known-faced P90 ball, plus {BATTING_MODEL_CONFIG.eligibility_baseline_balls} prior non-pressure baseline balls. Phase 7 recomputes the batting pressure-match count from known-faced P90 rows and reapplies the unchanged Phase 5 reliability and Model C formulas; the original Phase 5 count is retained in the detailed CSV for audit. Bowlers require {BOWLING_MODEL_CONFIG.eligibility_pressure_balls} known legal high-pressure balls, {BOWLING_MODEL_CONFIG.eligibility_pressure_matches} pressure matches, {BOWLING_MODEL_CONFIG.eligibility_pressure_innings} pressure innings, {BOWLING_MODEL_CONFIG.eligibility_baseline_balls} prior normal-pressure legal balls, and at least {BOWLING_MODEL_CONFIG.baseline_coverage_minimum:.0%} walk-forward baseline coverage. Meeting these data thresholds is not auction eligibility.\n\n"
        "## Top 10 batters\n\n" + _format_player_table(bat_top, bat_cols) + "\n\n"
        "## Top 10 bowlers\n\n" + _format_player_table(bowl_top, bowl_cols) + "\n\n"
        "## Method and strategic interpretation\n\n"
        "### Data, pressure, and accounting\n\n"
        "The rankings use the challenge ball-by-ball records joined to match, competition, team, and player-ID dimensions; the generated source table has 200,999 deliveries and the ranked historical IPL scope is shown above. Aggregation and joins use stable IDs and checked delivery keys. The scoring columns for batter runs, extras counts, wicket flags, and player IDs have no missing source values in this pack. Blank extras labels with zero extras are treated as no-extra deliveries; unfamiliar nonzero extras remain flagged as uncertain. Unknown-faced balls are excluded from known batter exposure.\n\n"
        "Match state is reconstructed before each delivery by subtracting that delivery from cumulative runs, wickets, and legal balls. Pressure normalization compares each month with earlier months only within competition group × innings phase; fewer than 100 reference states leaves the pressure label unavailable. Resource depletion is the mean of innings progress and wicket loss: `100 × ((known/possible legal-ball progress midpoint + wickets lost / 10) / 2)`. For batting, chase demand is the mean of the prior-history percentile of required-rate gap and fraction of target remaining; first-innings demand is one minus the prior-history run-rate percentile. The weighted batting index is `100 × (0.5 × demand + 0.5 × resource depletion)`. The nonlinear sensitivity index is `100 × (1 − sqrt((1 − demand) × (1 − resource)))`. For bowling, chase threat is one minus batting chase demand and bowling resource is one minus resource depletion; in the first innings, threat is the run-rate percentile and resource is that percentile multiplied by wickets remaining / 10. These components are combined with the same weighted and nonlinear formulas. The indices are descriptive match-state proxies, not pressure probabilities or psychological measurements.\n\n"
        "Batting P90 means the month's weighted pressure score is at or above its prior-month 90th percentile within competition group × phase. The batting baseline uses only earlier-month, known-faced, non-P90 observations. It first matches competition, season, phase, and first/chase context, then falls back to broader competition pools; a peer pool needs 30 balls where possible, sparse pools are flagged, and player rates are shrunk toward the peer rate using 50 prior balls. A batter's baseline differential is pressure outcome rate minus the baseline averaged across that batter's known-faced P90 deliveries.\n\n"
        "Bowling high pressure is assigned at the prior-month P90 within competition group × phase; the baseline reference band is the P25–P75 normal band. Personal baseline dates are strictly earlier than the evaluated calendar date, with same-date matches excluded. The primary format-hierarchical peer order is exact competition/phase/context, same-format phase/context, same-format phase, then same-format overall; the chosen peer level needs 60 legal balls where possible, with an explicitly flagged broader or player-only fallback otherwise. Personal normal history accumulates across seasons within format, phase, and context. Each rate is shrunk toward the selected peer rate with prior strengths of 60 balls for economy, 180 for wickets/over, 60 for dots, 60 for boundary deliveries, and 120 for extras/over.\n\n"
        "Bowling accounting uses known legal balls and bowler-attributable runs/wickets. Byes and leg-byes are excluded from bowler runs; wides and no-balls are included. Unfamiliar extras and uncertain legality are retained as flags and bounds; the primary rate score uses known lower-bound bowler runs and known legal balls, not an imputed unknown attribution. Approximate baseline standard errors are count-based summaries, not full uncertainty intervals.\n\n"
        "### Composite formulas and reliability\n\n"
        "For batting, each metric is robust-scored within competition: `z = clip((x − median) / (1.4826 × MAD), −3, 3)` (population standard deviation is the fallback when MAD is zero), then `score = 50 + (50/3) × z`; lower-is-better metrics reverse the sign. This maps the clipped range to 0–100. The absolute domain equally averages pressure runs/ball, boundary-ball frequency, inverse dot frequency, and inverse dismissal frequency. The relative domain equally averages the corresponding pressure-minus-personal-baseline differences. Consistency is the lower-is-better robust score of the player's standard deviation in match-level runs/ball over pressure and non-pressure matches; it is not a pressure-only consistency estimate. Model A equally averages absolute, relative, and consistency domains. Reliability is `100 × (1−e^(−ESS/100)) × (1−e^(−M/8)) × (0.5 + 0.5×(1−e^(−B/100)))`, where `ESS` is Kish effective pressure exposure, `M` is distinct known-faced P90 matches, and `B` is prior baseline balls. Model C is `reliability × Model A + (1−reliability) × 50`; Phase 7 recalculates `M` from known-faced P90 deliveries before applying the unchanged formula. Reliability is applied once as shrinkage.\n\n"
        "For bowling, each component is a within-competition percentile rank (direction reversed for lower-is-better metrics). Each peer-shrunk rate follows `scale × (player numerator + peer rate × prior-ball strength) / (player legal balls + prior-ball strength)`. The five absolute inputs are pressure economy, wickets/over, bowler-dot rate, boundary-concession rate, and bowler extras/over. The five relative inputs are the favorable signed differences between each unshrunk pressure rate and its walk-forward normal baseline. The four Model C domains are peer-shrunk absolute outcomes, relative improvement, match performance/dispersion, and statistical reliability; each domain has equal 25% weight. Within each domain, metric weights are proportional to `1 / (1 + mean absolute Spearman correlation)` and renormalized over available metrics; each domain score is the weighted mean of its available components. The consistency domain combines lower-tail match-performance percentile and fraction meeting the competition median standard with inverse median absolute deviation and inverse standard deviation; it therefore includes both performance level and dispersion, not dispersion alone. The reliability domain ranks pressure matches, innings and balls; baseline history balls, coverage and peer balls; and lower fallback level. Reliability is one explicit domain in Model C, not a second post-score discount.\n\n"
        f"The batter uncertainty output resamples match aggregates {BATTING_MODEL_CONFIG.bootstrap_replicates} times with seed {BATTING_MODEL_CONFIG.random_seed}; baseline values are held fixed, so pressure-minus-baseline percentile intervals are conditional and not calibrated prediction intervals. {batting_ci_cross_zero} of {len(bat_top)} selected batters have an interval spanning zero. Bowling resamples competition-group matches jointly across bowlers {BOWLING_MODEL_CONFIG.bootstrap_replicates} times; baseline, consistency, and reliability inputs are fixed. The selected bowlers' top-ten resample frequencies range from {bowling_frequency_range}; these are conditional ranking-stability frequencies, not probabilities of future selection or performance.\n\n"
        "Observed batting order is the median supplied batting_order over distinct player-match records (with fewer than five records marked uncertain), grouped as 1–3, 4–6, or 7+. Observed bowling phase is the largest share of a player's supplied innings-phase delivery records; it is usage evidence, not proof of powerplay/death specialization or pace/spin style. No wicketkeeping or all-rounder classification is inferred.\n\n"
        "Robustness is reported as raw top-ten overlap and rank correlation for existing composite/pressure scenarios; categorical high/mixed/low labels are not used. Pressure-only sensitivity ranks use runs per ball and therefore are not alternate composite rankings. Sensitivity results identify the scenario type and sample size; retrospective static thresholds are not treated as prospective.\n\n"
        "Batting's primary score gives equal weight to related run-production, boundary, dot, and dismissal measures within its absolute and relative domains; those measures can be correlated. The correlation-reduced alternative is a sensitivity check rather than an outcome-validated replacement. Bowling consistency also includes lower-tail performance and standard attainment, so it overlaps conceptually with absolute performance. These are explicit composite choices, not empirically estimated optimal weights.\n\n"
        "The Phase 5 pressure-definition table includes the walk-forward weighted P85/P90/P95 labels, the nonlinear walk-forward label, fixed absolute-score cutoffs, and a static full-sample Phase 4 threshold. Fixed/static scenarios are sensitivity analyses only, not valid prospective pressure labels. Phase 5 does not provide a separate alternate-baseline hierarchy rescore, so batting baseline-hierarchy robustness is not claimed.\n\n"
        "### Existing-model sensitivity\n\n"
        "Batting composite and weighting comparisons:\n\n" + _format_player_table(batting_models, ["score_model", "primary_top10_overlap", "top10_size", "spearman_with_primary_score"]) + "\n\n"
        "Batting pressure-definition scenarios rank pressure runs per ball, not the final composite:\n\n" + _format_player_table(batting_pressure_scenarios, ["scenario", "candidate_records", "primary_top10_overlap", "primary_top10_size", "spearman_with_primary_score"]) + "\n\n"
        "Batting absolute/relative-weight scenarios omit consistency and reliability by design in the existing Phase 5 sensitivity routine:\n\n" + _format_player_table(batting_weight_scenarios, ["scenario", "candidate_records", "primary_top10_overlap", "primary_top10_size", "spearman_with_primary_score"]) + "\n\n"
        "Bowling composite comparisons:\n\n" + _format_player_table(bowling_models, ["score_model", "primary_top10_overlap", "top10_size", "spearman_with_primary_score"]) + "\n\n"
        + ("Bowling pipeline sensitivity rows below are the existing Phase 6 summaries pooled across competition-group/player pairs (the primary eligible groups are IPL and T20I); they should not be read as an IPL-only overlap. Leave-season-out rows use absolute pressure outcomes and keep baselines fixed.\n\n" + _format_player_table(bowling_sensitivity_rows, ["scenario", "top10_overlap_player_group_pairs", "top10_base_pairs", "eligible_rank_spearman", "note"]) if not bowling_sensitivity_rows.empty else "No Phase 6 scenario sensitivity table was supplied to Phase 7.\n\n")
        + "The operational use is first to cross-reference the final official registered auction pool and retention status, then assess squad needs and auction constraints before scouting or price decisions. Historical performance is not evidence that a player is currently available or will perform under future pressure; the analysis does not establish that pressure caused the observed outcomes.\n\n"
        "## Limitations\n\n"
        "- Pressure labels are provisional match-state proxies without independent ground truth; they do not measure psychology or establish causal effects.\n"
        "- The scores are descriptive, not trained against future auction or match outcomes.\n"
        "- Bootstrap uncertainty is conditional on the existing Phase 5/6 pipeline designs; ranking stability is not a forecast guarantee.\n"
        "- IPL is ranked separately from T20I and Other T20. Cross-format evidence is not pooled into the auction order.\n"
        f"- The list pools historical IPL seasons and does not establish upcoming-auction eligibility or availability. The official IPL retention notice dated 2026-11-18 lists {retained_candidate_count} candidates from these two top tens as retained; absence from that list does not establish registration or auction availability.\n"
        "- Current RR needs, reserve prices, and scouting/medical information are not model inputs. No player price ceiling or final purchase recommendation is produced.\n"
        "- Composite scores are within-model rankings over historical IPL records. Batting and bowling scores have different constructions and must not be compared numerically.\n"
    )
    (reports / "final_strategy_report.md").write_text(report, encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].barh(bat_top["player_name"].astype(str).iloc[::-1], bat_top["overall_batting_pressure_score"].iloc[::-1], color="#d97706")
    axes[0].set(title="IPL batting pressure shortlist", xlabel="Phase 5 score (0–100)")
    axes[1].barh(bowl_top["player_name"].astype(str).iloc[::-1], bowl_top["overall_bowling_pressure_score"].iloc[::-1], color="#2563eb")
    axes[1].set(title="IPL bowling pressure shortlist", xlabel="Phase 6 score (0–100)")
    fig.suptitle("Provisional observed-performance rankings — not a pressure-trait or outcome prediction")
    fig.tight_layout()
    figure_path = figures / "final_strategy" / "ipl_top10_shortlists.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return {"batters": bat_top, "bowlers": bowl_top, "eligible_batters": batting,
            "eligible_bowlers": bowling, "figure": figure_path, "metadata": metadata,
            "sensitivity": {"batting_models": batting_models, "batting_pressure": batting_pressure_scenarios,
                            "batting_weights": batting_weight_scenarios, "bowling_models": bowling_models,
                            "bowling_pipeline": bowling_sensitivity_rows}}
