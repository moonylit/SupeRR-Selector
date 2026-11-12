"""Walk-forward pressure labels, personal baselines, and context regression."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import BattingModelConfig

LOGGER = logging.getLogger(__name__)
CONTEXT_KEYS = ["competition_group", "comp_id", "innings_phase", "innings_context"]
PLAYER_KEYS = CONTEXT_KEYS + ["striker_id"]


def assign_walk_forward_pressure_labels(deliveries: pd.DataFrame, config: BattingModelConfig) -> pd.DataFrame:
    """Label each month using prior-month pressure distributions only."""
    frame = deliveries.copy()
    month = pd.to_datetime(frame["match_date"], errors="coerce").dt.to_period("M")
    frame["pressure_label_available"] = False
    frame["pressure_high_p90_walk_forward"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    frame["pressure_high_nonlinear_p90_walk_forward"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    frame["pressure_band_walk_forward"] = "reference unavailable"
    frame["pressure_reference_state_count"] = 0
    groups = frame.groupby(["competition_group", "innings_phase"], dropna=False, sort=False).groups
    for keys in ( (config.pressure_score_column, "pressure_high_p90_walk_forward"),
                  (config.alternate_pressure_score_column, "pressure_high_nonlinear_p90_walk_forward") ):
        score_col, label_col = keys
        if score_col not in frame:
            continue
        for indices in groups.values():
            idx = pd.Index(indices)
            group_months = month.loc[idx]
            history: list[float] = []
            for current_month in sorted(group_months.dropna().unique()):
                month_idx = idx[group_months.eq(current_month).to_numpy()]
                values = pd.to_numeric(frame.loc[month_idx, score_col], errors="coerce").dropna()
                if history and len(history) >= max(1, config.pressure_reference_minimum):
                    p50, p75, p90 = np.quantile(history, config.pressure_band_percentiles)
                    frame.loc[values.index, label_col] = values.ge(p90).to_numpy()
                    frame.loc[values.index, "pressure_label_available"] = True
                    frame.loc[values.index, "pressure_reference_state_count"] = len(history)
                    if label_col == "pressure_high_p90_walk_forward":
                        frame.loc[values.index, "pressure_band_walk_forward"] = np.select(
                            [values.lt(p50), values.lt(p75), values.lt(p90)],
                            ["Low", "Moderate", "High"], default="Extreme"
                        )
                history.extend(values.astype(float).tolist())
    frame["pressure_high_context_p90_static_sensitivity"] = frame.get(
        "batting_high_weighted_context_p90", pd.Series(pd.NA, index=frame.index)
    )
    return frame


def _summarize_history(history: pd.DataFrame) -> pd.DataFrame:
    """Aggregate normal-pressure, known-faced history by player and context."""
    normal = history.loc[
        history["pressure_label_available"]
        & history["pressure_high_p90_walk_forward"].eq(False).fillna(False)
        & history["balls_faced_known"] & history["striker_id"].notna()
    ].copy()
    if normal.empty:
        return pd.DataFrame(columns=PLAYER_KEYS + ["runs", "balls", "fours", "sixes", "boundary_balls", "dots", "matches", "innings", "dismissals"])
    normal["_four"] = normal["fours_known"].astype(int)
    normal["_six"] = normal["sixes_known"].astype(int)
    normal["_boundary"] = normal["boundary_count_known"].gt(0).astype(int)
    normal["_dot"] = normal["dot_ball_batter_run_known"].astype(int)
    normal["_innings_key"] = normal["match_id"].astype("string") + ":" + normal["innings"].astype("string")
    stats = normal.groupby(PLAYER_KEYS, as_index=False, dropna=False).agg(
        runs=("batter_runs", "sum"), balls=("balls_faced_known", "sum"), fours=("_four", "sum"),
        sixes=("_six", "sum"), boundary_balls=("_boundary", "sum"), dots=("_dot", "sum"),
        matches=("match_id", "nunique"), innings=("_innings_key", "nunique"),
    )
    if "dismissal_for_striker" in history:
        events = history.loc[
            history["pressure_label_available"]
            & history["pressure_high_p90_walk_forward"].eq(False).fillna(False)
            & history["dismissal_event_valid"] & history["dismissed_player_id"].notna()
        ].copy()
        events["striker_id"] = events["dismissed_player_id"]
        dismissals = events.groupby(PLAYER_KEYS, as_index=False, dropna=False).size().rename(columns={"size": "dismissals"})
        stats = stats.merge(dismissals, on=PLAYER_KEYS, how="outer", validate="one_to_one")
    if "dismissals" not in stats:
        stats["dismissals"] = 0
    return stats.fillna(0)


def _rate(num: float, den: float) -> float:
    return float(num / den) if den > 0 else np.nan


def _baseline_record(player: pd.Series, peers: pd.DataFrame, config: BattingModelConfig, source: str) -> dict[str, Any]:
    other = peers.loc[peers["striker_id"].ne(player["striker_id"])]
    peer_balls = float(other["balls"].sum())
    own_balls = float(player["balls"])
    strength = float(config.baseline_prior_strength_balls)
    metrics = {
        "rpb": ("runs", "balls"), "boundary_frequency": ("boundary_balls", "balls"),
        "four_frequency": ("fours", "balls"), "six_frequency": ("sixes", "balls"),
        "dot_ball_frequency": ("dots", "balls"), "dismissal_frequency": ("dismissals", "balls"),
    }
    out: dict[str, Any] = {"baseline_pool_source": source, "baseline_pool_balls": peer_balls,
                            "baseline_player_history_balls": own_balls,
                            "baseline_player_history_matches": float(player["matches"]),
                            "baseline_player_history_innings": float(player["innings"])}
    for label, (numerator, denominator) in metrics.items():
        peer_rate = _rate(float(other[numerator].sum()), peer_balls) if peer_balls else np.nan
        own_rate = _rate(float(player[numerator]), own_balls)
        if np.isfinite(peer_rate):
            estimate = (float(player[numerator]) + peer_rate * strength) / (own_balls + strength)
        else:
            estimate = own_rate
        out[f"baseline_{label}_hierarchical"] = estimate
        if label == "rpb":
            out["baseline_rpb_raw"] = own_rate
            out["baseline_pool_runs_per_ball"] = peer_rate
    out["baseline_shrinkage_fraction"] = own_balls / (own_balls + strength) if own_balls + strength else np.nan
    out["baseline_pool_reliability"] = peer_balls >= config.minimum_context_pool_balls
    out["baseline_status"] = (
        "no historical baseline" if not np.isfinite(out["baseline_rpb_hierarchical"])
        else "player history available" if own_balls >= strength
        else "partially pooled baseline" if out["baseline_pool_reliability"]
        else "low-history partially pooled baseline"
    )
    return out


def add_historical_personal_baselines(deliveries: pd.DataFrame, config: BattingModelConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach prior-month matched-context baseline with peer shrinkage and fallback."""
    frame = deliveries.copy()
    months = pd.to_datetime(frame["match_date"], errors="coerce").dt.to_period("M")
    baseline_fields = ["baseline_rpb_hierarchical", "baseline_boundary_frequency_hierarchical",
        "baseline_four_frequency_hierarchical", "baseline_six_frequency_hierarchical",
        "baseline_dot_ball_frequency_hierarchical", "baseline_dismissal_frequency_hierarchical",
        "baseline_rpb_raw", "baseline_pool_runs_per_ball", "baseline_player_history_balls",
        "baseline_player_history_matches", "baseline_player_history_innings", "baseline_pool_balls",
        "baseline_shrinkage_fraction", "baseline_pool_reliability", "baseline_pool_source", "baseline_status"]
    for col in baseline_fields:
        frame[col] = np.nan if col not in ("baseline_pool_reliability", "baseline_pool_source", "baseline_status") else False if col == "baseline_pool_reliability" else "none" if col == "baseline_pool_source" else "no prior matches"
    history_parts: list[pd.DataFrame] = []
    count_columns = ["runs", "balls", "fours", "sixes", "boundary_balls", "dots", "matches", "innings", "dismissals"]
    for month in sorted(months.dropna().unique()):
        current_mask = months.eq(month)
        history = pd.concat(history_parts, ignore_index=True) if history_parts else frame.iloc[:0]
        summary = _summarize_history(history)
        targets = frame.loc[current_mask, PLAYER_KEYS].drop_duplicates()
        lookup = targets.merge(summary, on=PLAYER_KEYS, how="left", validate="one_to_one")
        for col in count_columns:
            lookup[col] = pd.to_numeric(lookup[col], errors="coerce").fillna(0)
        lookup = lookup.rename(columns={col: f"own_{col}" for col in count_columns})
        lookup["baseline_pool_source"] = "none"
        for col in count_columns:
            lookup[col] = 0.0
        candidates = [
            (CONTEXT_KEYS, "matched context"),
            (["competition_group", "innings_context"], "competition context fallback"),
            (["competition_group"], "competition fallback"),
        ]
        for grouping, source in candidates:
            if summary.empty:
                continue
            pool = summary.groupby(grouping, as_index=False, dropna=False)[count_columns].sum()
            pool = pool.rename(columns={col: f"pool_{col}" for col in count_columns})
            candidate = lookup.merge(pool, on=grouping, how="left", validate="many_to_one")
            peer_balls = candidate["pool_balls"].fillna(0) - candidate["own_balls"]
            choose = lookup["baseline_pool_source"].eq("none") & peer_balls.ge(config.minimum_context_pool_balls)
            for col in count_columns:
                peer_count = (candidate[f"pool_{col}"].fillna(0) - candidate[f"own_{col}"]).clip(lower=0)
                lookup.loc[choose, col] = peer_count.loc[choose].to_numpy()
            lookup.loc[choose, "baseline_pool_source"] = source
        # If no pool passes the minimum, retain the broadest available peer pool as a low-confidence estimate.
        no_pool = lookup["baseline_pool_source"].eq("none")
        if no_pool.any() and not summary.empty:
            pool = summary.groupby(["competition_group"], as_index=False, dropna=False)[count_columns].sum().rename(columns={col: f"pool_{col}" for col in count_columns})
            candidate = lookup.loc[no_pool].merge(pool, on="competition_group", how="left", validate="many_to_one")
            for col in count_columns:
                lookup.loc[no_pool, col] = (candidate[f"pool_{col}"].fillna(0) - candidate[f"own_{col}"]).clip(lower=0).to_numpy()
            lookup.loc[no_pool, "baseline_pool_source"] = "competition fallback (sparse)"
        own_balls = lookup["own_balls"]
        peer_balls = lookup["balls"]
        for label, numerator in (("rpb", "runs"), ("boundary_frequency", "boundary_balls"),
            ("four_frequency", "fours"), ("six_frequency", "sixes"), ("dot_ball_frequency", "dots"),
            ("dismissal_frequency", "dismissals")):
            own_rate = lookup[f"own_{numerator}"].div(own_balls.replace(0, np.nan))
            peer_rate = lookup[numerator].div(peer_balls.replace(0, np.nan))
            shrunk = (lookup[f"own_{numerator}"] + peer_rate * config.baseline_prior_strength_balls).div(own_balls + config.baseline_prior_strength_balls)
            lookup[f"baseline_{label}_hierarchical"] = shrunk.where(peer_rate.notna(), own_rate)
        lookup["baseline_rpb_raw"] = lookup["own_runs"].div(own_balls.replace(0, np.nan))
        lookup["baseline_pool_runs_per_ball"] = lookup["runs"].div(peer_balls.replace(0, np.nan))
        lookup["baseline_player_history_balls"] = own_balls
        lookup["baseline_player_history_matches"] = lookup["own_matches"]
        lookup["baseline_player_history_innings"] = lookup["own_innings"]
        lookup["baseline_pool_balls"] = peer_balls
        lookup["baseline_shrinkage_fraction"] = own_balls.div(own_balls + config.baseline_prior_strength_balls)
        lookup["baseline_pool_reliability"] = peer_balls.ge(config.minimum_context_pool_balls)
        lookup["baseline_status"] = np.select(
            [lookup["baseline_rpb_hierarchical"].isna(), own_balls.ge(config.baseline_prior_strength_balls), lookup["baseline_pool_reliability"]],
            ["no historical baseline", "player history available", "partially pooled baseline"],
            default="low-history partially pooled baseline")
        idx = frame.index[current_mask]
        indexed = frame.loc[current_mask, PLAYER_KEYS].merge(lookup[PLAYER_KEYS + baseline_fields], on=PLAYER_KEYS, how="left", validate="many_to_one")
        for col in baseline_fields:
            frame.loc[idx, col] = indexed[col].to_numpy()
        this_month_normal = frame.loc[current_mask & frame["pressure_label_available"]
            & frame["pressure_high_p90_walk_forward"].eq(False).fillna(False) & frame["balls_faced_known"]]
        history_parts.append(this_month_normal)
    frame["baseline_pool_reliability"] = frame["baseline_pool_reliability"].fillna(False).astype(bool)
    audit = frame.groupby(CONTEXT_KEYS, as_index=False, dropna=False).agg(
        delivery_rows=("match_id", "size"), rows_with_personal_baseline=("baseline_rpb_hierarchical", "count"),
        median_player_historical_balls=("baseline_player_history_balls", "median"),
        median_peer_pool_balls=("baseline_pool_balls", "median"), pooled_baseline_rows=("baseline_pool_reliability", "sum"))
    return frame, audit


def add_walk_forward_context_regression(deliveries: pd.DataFrame, config: BattingModelConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare a prior-year ridge model with a personal baseline on future years."""
    frame = deliveries.copy()
    # Each row represents one faced delivery; the per-delivery outcome is runs/1.
    frame["batter_runs_per_ball_known"] = pd.to_numeric(frame["batter_runs"], errors="coerce").where(frame["balls_faced_known"])
    frame["context_regression_expected_runs_per_ball"] = np.nan
    frame["context_regression_training_rows"] = 0
    frame["context_regression_training_through_year"] = pd.NA
    frame["context_regression_available"] = False
    year = pd.to_datetime(frame["match_date"], errors="coerce").dt.year
    frame["player_id_model"] = frame["striker_id"].astype("string")
    cats = ["competition_group", "comp_id", "innings_phase", "innings_context", "player_id_model"]
    nums = ["batting_pressure_weighted", "current_run_rate", "required_run_rate", "wickets_remaining", "legal_balls_remaining", "innings_progress_fraction", "remaining_target_fraction"]
    for col in nums:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for col, bounds in {"batting_pressure_weighted": (0,100), "current_run_rate": (0,36),
                        "required_run_rate": (0,36), "wickets_remaining": (0,10),
                        "legal_balls_remaining": (0,120), "innings_progress_fraction": (0,1),
                        "remaining_target_fraction": (0,1)}.items():
        frame[col] = frame[col].clip(lower=bounds[0], upper=bounds[1])
    for col in cats:
        frame[col] = frame[col].astype("string").fillna("Unknown")
    fold_rows: list[dict[str, Any]] = []
    for test_year in sorted(year.dropna().unique()):
        train_mask = year.lt(test_year) & frame["balls_faced_known"] & frame["striker_id"].notna()
        test_mask = year.eq(test_year) & frame["balls_faced_known"] & frame["striker_id"].notna()
        train, test = frame.loc[train_mask], frame.loc[test_mask]
        if len(train) < config.minimum_regression_training_rows or test.empty:
            fold_rows.append({"test_year": int(test_year), "training_rows": len(train), "test_rows": len(test), "model_available": False, "reason": "insufficient prior-year training rows or empty test fold"})
            continue
        transformer = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), cats),
            ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler(with_mean=False))]), nums),
        ], sparse_threshold=1.0)
        model = Pipeline([("features", transformer), ("ridge", Ridge(alpha=config.ridge_alpha, solver="lsqr"))])
        model.fit(train[cats + nums], train["batter_runs_per_ball_known"])
        pred = np.clip(model.predict(test[cats + nums]), 0, 6)
        frame.loc[test.index, "context_regression_expected_runs_per_ball"] = pred
        frame.loc[test.index, "context_regression_training_rows"] = len(train)
        frame.loc[test.index, "context_regression_training_through_year"] = int(test_year) - 1
        frame.loc[test.index, "context_regression_available"] = True
        actual = test["batter_runs_per_ball_known"].to_numpy(float)
        base = test["baseline_rpb_hierarchical"].to_numpy(float)
        valid = np.isfinite(base)
        mae = float(np.mean(np.abs(actual - pred)))
        rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
        base_mae = float(np.mean(np.abs(actual[valid] - base[valid]))) if valid.any() else np.nan
        fold_rows.append({"test_year": int(test_year), "training_rows": len(train), "test_rows": len(test), "model_available": True,
            "mae_runs_per_ball": mae, "rmse_runs_per_ball": rmse, "hierarchical_baseline_mae_runs_per_ball_on_same_rows": base_mae,
            "mae_improvement_vs_hierarchical_baseline": base_mae - mae if np.isfinite(base_mae) else np.nan,
            "walk_forward_no_future_training": True, "predictors_use_pre_delivery_context_only": True})
    return frame, pd.DataFrame(fold_rows)
