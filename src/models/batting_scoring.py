"""Robust, bounded, separated player evaluation components."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def robust_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Competition-group median/MAD score mapped to 0–100, clipped at ±3 MAD."""
    x = pd.to_numeric(series, errors="coerce")
    med = x.median()
    mad = (x - med).abs().median()
    scale = 1.4826 * mad if pd.notna(mad) and mad > 0 else x.std(ddof=0)
    if not np.isfinite(scale) or scale == 0:
        z = pd.Series(0.0, index=x.index)
    else:
        z = ((x - med) / scale).clip(-3, 3)
    if not higher_is_better:
        z = -z
    return (50 + z / 6 * 50).fillna(50)


def build_composite_scores(statistics: pd.DataFrame, eligibility: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create equal-domain, correlation-reduced and reliability-shrunk scores."""
    frame = statistics.copy()
    if frame.empty:
        return frame, pd.DataFrame()
    metrics = {
        "pressure_rpb": ("pressure_runs_per_ball", True),
        "pressure_boundary": ("pressure_boundary_frequency", True),
        "pressure_dot": ("pressure_dot_ball_frequency", False),
        "pressure_dismissal": ("pressure_dismissal_frequency", False),
        "relative_rpb": ("rpb_personal_baseline_difference", True),
        "relative_boundary": ("boundary_frequency_personal_baseline_difference", True),
        "relative_dot": ("dot_frequency_personal_baseline_difference", False),
        "relative_dismissal": ("dismissal_frequency_personal_baseline_difference", False),
    }
    for alias, (source, _) in metrics.items():
        frame[alias] = _numeric(frame, source, np.nan)
    score_cols = []
    for group, idx in frame.groupby("competition_group", dropna=False).groups.items():
        for alias, (_, better) in metrics.items():
            frame.loc[idx, f"z_{alias}"] = robust_score(frame.loc[idx, alias], better)
            score_cols.append(f"z_{alias}")
    # Balanced domains reduce duplicate credit for correlated scoring metrics.
    frame["absolute_performance_score"] = frame[["z_pressure_rpb", "z_pressure_boundary", "z_pressure_dot", "z_pressure_dismissal"]].mean(axis=1)
    frame["relative_resilience_score"] = frame[["z_relative_rpb", "z_relative_boundary", "z_relative_dot", "z_relative_dismissal"]].mean(axis=1)
    if "player_match_rpb_std" not in frame:
        frame["player_match_rpb_std"] = np.nan
    frame["consistency_score"] = frame.groupby("competition_group", dropna=False)["player_match_rpb_std"].transform(lambda x: robust_score(x, False))
    balls = _numeric(frame, "pressure_balls").fillna(0)
    effective_balls = _numeric(frame, "effective_pressure_exposure").fillna(0)
    matches = _numeric(frame, "pressure_matches").fillna(0)
    baseline_balls = _numeric(frame, "baseline_history_balls").fillna(0)
    frame["reliability_score"] = (100 * (1 - np.exp(-effective_balls / 100)) * (1 - np.exp(-matches / 8)) * (0.5 + 0.5 * (1 - np.exp(-baseline_balls / 100)))).clip(0, 100)
    frame["score_model_a_equal_domains"] = frame[["absolute_performance_score", "relative_resilience_score", "consistency_score"]].mean(axis=1)
    # Inverse absolute-correlation weights over the eight robust components.
    component_cols = [f"z_{k}" for k in metrics]
    corr = frame[component_cols].corr().abs().fillna(0)
    weights = 1 / corr.sum(axis=1).replace(0, 1)
    weights = weights / weights.sum()
    frame["score_model_b_correlation_reduced"] = frame[component_cols].mul(weights, axis=1).sum(axis=1)
    reliability = frame["reliability_score"] / 100
    frame["score_model_c_shrinkage_adjusted"] = (frame["score_model_a_equal_domains"] * reliability + 50 * (1 - reliability)).clip(0, 100)
    frame["overall_batting_pressure_score"] = frame["score_model_c_shrinkage_adjusted"].clip(0, 100)
    frame["score_interpretation"] = "exploratory observed performance score; not a psychological trait or final selection"
    comparison = pd.DataFrame({"component": component_cols, "correlation_reduced_weight": [weights.get(c, np.nan) for c in component_cols]})
    return frame, comparison


def assign_eligibility(scores: pd.DataFrame, config: object) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retain every observed player and report threshold sensitivity counts."""
    frame = scores.copy()
    balls = _numeric(frame, "pressure_balls").fillna(0)
    matches = _numeric(frame, "pressure_matches").fillna(0)
    base = _numeric(frame, "baseline_history_balls").fillna(0)
    frame["eligible_pressure_sample"] = balls.ge(config.eligibility_pressure_balls) & matches.ge(config.eligibility_pressure_matches)
    frame["baseline_reliable"] = base.ge(config.eligibility_baseline_balls)
    frame["eligible_analytical_candidate"] = frame["eligible_pressure_sample"] & frame["baseline_reliable"]
    frame["uncertain_data_flag"] = pd.to_numeric(frame.get("uncertain_faced_deliveries", 0), errors="coerce").fillna(0).gt(0)
    frame["eligibility_status"] = np.select(
        [~frame["eligible_pressure_sample"], ~frame["baseline_reliable"], frame["uncertain_data_flag"]],
        ["insufficient pressure sample", "eligible pressure sample; unreliable baseline", "eligible with uncertain deliveries"],
        default="eligible analytical candidate")
    rows = []
    for b in config.pressure_ball_sensitivity:
        for m in config.pressure_match_sensitivity:
            for group, part in frame.groupby("competition_group", dropna=False):
                pressure_ok = balls.loc[part.index].ge(b) & matches.loc[part.index].ge(m)
                rows.append({"competition_group": group, "minimum_pressure_balls": b, "minimum_pressure_matches": m,
                             "players_meeting_pressure_sample": int(pressure_ok.sum()),
                             "also_meeting_baseline_requirement": int((pressure_ok & frame.loc[part.index,"baseline_reliable"]).sum()),
                             "total_player_competition_records": len(part)})
    return frame, pd.DataFrame(rows)
