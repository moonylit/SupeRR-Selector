"""Explainable, role-specific pressure features and candidate indices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PressureConfig:
    """Parameters are explicit so alternative thresholds are reproducible."""

    demand_weight: float = 0.5
    nonlinear_interaction_power: float = 0.5
    minimum_reference_states: int = 100
    fixed_high_threshold: float = 80.0
    competition_percentiles: tuple[int, ...] = (90, 95)
    context_percentiles: tuple[int, ...] = (90, 95)

    def __post_init__(self) -> None:
        if not 0 <= self.demand_weight <= 1:
            raise ValueError("demand_weight must be between 0 and 1")
        if self.nonlinear_interaction_power <= 0:
            raise ValueError("nonlinear_interaction_power must be greater than 0")
        if self.minimum_reference_states < 1:
            raise ValueError("minimum_reference_states must be at least 1")
        if not 0 <= self.fixed_high_threshold <= 100:
            raise ValueError("fixed_high_threshold must be between 0 and 100")
        percentiles = self.competition_percentiles + self.context_percentiles
        if any(not 0 <= percentile <= 100 for percentile in percentiles):
            raise ValueError("percentile thresholds must be between 0 and 100")


def _rank_by_context(
    values: pd.Series,
    frame: pd.DataFrame,
    reference_values: pd.Series | None = None,
    minimum_reference_states: int = 100,
) -> pd.Series:
    """Walk-forward empirical CDF using earlier match dates only.

    All deliveries in a calendar month share the reference set through the
    prior month, preventing same-month, same-match, or future outcomes from
    setting a delivery's normalization. Fewer than the configured number of
    prior states use 0.5.
    """
    result = pd.Series(0.5, index=frame.index, dtype="float64")
    reference_values = values if reference_values is None else reference_values
    periods = pd.to_datetime(frame["match_date"], errors="coerce").dt.to_period("M")
    grouped = frame.groupby(["competition_group", "innings_phase"], dropna=False, sort=False).groups
    for indices in grouped.values():
        group_periods = periods.loc[indices]
        by_period = group_periods.dropna().groupby(group_periods.dropna(), sort=True).groups
        history = np.asarray([], dtype=float)
        for _, on_period in by_period.items():
            valid_rows = pd.to_numeric(values.loc[on_period], errors="coerce").dropna()
            reference_current = pd.to_numeric(reference_values.loc[on_period], errors="coerce").dropna()
            current = valid_rows.to_numpy(dtype=float)
            if len(history) >= minimum_reference_states and len(current):
                lower = np.searchsorted(history, current, side="left")
                upper = np.searchsorted(history, current, side="right")
                ranks = (lower + upper) / (2 * len(history))
                result.loc[valid_rows.index] = ranks
            if len(reference_current):
                history = np.sort(np.concatenate((history, reference_current.to_numpy(dtype=float))))
    return result.clip(0, 1)


def _combine_index(
    demand: pd.Series,
    resource: pd.Series,
    demand_weight: float = 0.5,
    nonlinear_power: float = 0.5,
) -> tuple[pd.Series, pd.Series]:
    """Return additive and nonlinear indices from two equal conceptual domains.

    The additive index weights demand and resource pressure symmetrically by
    default (50% each); the demand share is configurable.
    The nonlinear alternative is a symmetric geometric interaction over the
    remaining comfort in each domain. No match-result label is fit to either.
    """
    d = demand.clip(0, 1)
    r = resource.clip(0, 1)
    if not 0 <= demand_weight <= 1:
        raise ValueError("demand_weight must be between 0 and 1")
    if nonlinear_power <= 0:
        raise ValueError("nonlinear_power must be greater than 0")
    additive = 100 * (demand_weight * d + (1 - demand_weight) * r)
    nonlinear = 100 * (1 - ((1 - d) * (1 - r)) ** nonlinear_power)
    return additive.clip(0, 100), nonlinear.clip(0, 100)


def add_pressure_features(
    states: pd.DataFrame, config: PressureConfig | None = None
) -> pd.DataFrame:
    """Add normalized role-specific strain components and two 0–100 indices.

    Normalization uses empirical delivery-state ranks within competition group
    and innings phase. This makes IPL, T20I, and Other T20 distributions visible
    separately while retaining meaningful comparisons within each group.
    """
    config = config or PressureConfig()
    frame = states.copy()
    def rank(values: pd.Series, source: pd.DataFrame, reference: pd.Series | None = None) -> pd.Series:
        return _rank_by_context(
            values,
            source,
            reference_values=reference,
            minimum_reference_states=config.minimum_reference_states,
        )

    def combine(demand: pd.Series, resource: pd.Series) -> tuple[pd.Series, pd.Series]:
        return _combine_index(
            demand,
            resource,
            demand_weight=config.demand_weight,
            nonlinear_power=config.nonlinear_interaction_power,
        )
    if "boundary_event" in frame and "match_date" in frame:
        daily = frame.groupby(
            ["match_date", "competition_group", "innings_phase"], dropna=False, as_index=False
        ).agg(
            prior_day_boundary_events=("boundary_event", "sum"),
            prior_day_deliveries=("boundary_event", "size"),
        ).sort_values("match_date", kind="stable")
        historical_keys = ["competition_group", "innings_phase"]
        daily["historical_boundary_events_prior"] = daily.groupby(
            historical_keys, dropna=False
        )["prior_day_boundary_events"].cumsum() - daily["prior_day_boundary_events"]
        daily["historical_deliveries_prior"] = daily.groupby(
            historical_keys, dropna=False
        )["prior_day_deliveries"].cumsum() - daily["prior_day_deliveries"]
        daily["historical_boundary_event_rate_prior"] = (
            daily["historical_boundary_events_prior"]
            / daily["historical_deliveries_prior"].replace(0, np.nan)
        )
        frame = frame.merge(
            daily[["match_date", "competition_group", "innings_phase", "historical_boundary_event_rate_prior"]],
            on=["match_date", "competition_group", "innings_phase"],
            how="left",
            validate="many_to_one",
        )
    limit = pd.to_numeric(frame["innings_ball_limit"], errors="coerce").replace(0, np.nan)
    progress_low = (frame["legal_balls_bowled_known"] / limit).clip(0, 1).fillna(0)
    progress_high = (frame["legal_balls_bowled_possible"] / limit).clip(0, 1).fillna(0)
    wicket_loss = (frame["wickets_lost_before"] / 10).clip(0, 1)
    resource_low = (progress_low + wicket_loss) / 2
    resource_high = (progress_high + wicket_loss) / 2
    resource_mid = (resource_low + resource_high) / 2
    frame["resource_depletion_score"] = 100 * resource_mid
    frame["resource_depletion_lower"] = 100 * resource_low
    frame["resource_depletion_upper"] = 100 * resource_high
    frame["innings_progress_fraction"] = (progress_low + progress_high) / 2
    frame["wicket_loss_fraction"] = wicket_loss

    is_chase = frame["is_chasing"] & frame["chase_target"].notna()
    gap_mid = frame["required_run_rate"] - frame["current_run_rate"]
    gap_low = frame["required_run_rate_lower"] - frame["current_run_rate_upper"]
    gap_high = frame["required_run_rate_upper"] - frame["current_run_rate_lower"]
    demand_rate_mid = rank(gap_mid.where(is_chase), frame)
    demand_rate_low = rank(gap_low.where(is_chase), frame, gap_mid.where(is_chase))
    demand_rate_high = rank(gap_high.where(is_chase), frame, gap_mid.where(is_chase))
    target_fraction = (frame["runs_required"] / frame["chase_target"].replace(0, np.nan)).clip(0, 1)
    demand_target = rank(target_fraction.where(is_chase), frame)
    chase_demand_mid = (demand_rate_mid + demand_target) / 2
    chase_demand_low = (demand_rate_low + demand_target) / 2
    chase_demand_high = (demand_rate_high + demand_target) / 2

    first_innings = ~frame["is_chasing"]
    pace_mid = rank(frame["current_run_rate"].where(first_innings), frame)
    pace_low = rank(
        frame["current_run_rate_lower"].where(first_innings),
        frame,
        reference=frame["current_run_rate"].where(first_innings),
    )
    pace_high = rank(
        frame["current_run_rate_upper"].where(first_innings),
        frame,
        reference=frame["current_run_rate"].where(first_innings),
    )
    # Without a first-innings target, use under-scoring versus the local pace
    # distribution; at the start of an innings the run-rate component is neutral.
    first_demand_mid = 1 - pace_mid
    first_demand_low = 1 - pace_high
    first_demand_high = 1 - pace_low

    batting_demand = chase_demand_mid.where(is_chase, first_demand_mid)
    batting_demand_low = chase_demand_low.where(is_chase, first_demand_low)
    batting_demand_high = chase_demand_high.where(is_chase, first_demand_high)
    batting_a, batting_b = combine(batting_demand, resource_mid)
    batting_a_low, batting_b_low = combine(batting_demand_low, resource_low)
    batting_a_high, batting_b_high = combine(batting_demand_high, resource_high)

    # Bowling pressure measures the opponent's scoring threat and resources.
    bowling_demand = (1 - chase_demand_mid).where(is_chase, pace_mid)
    bowling_demand_low = (1 - chase_demand_high).where(is_chase, pace_low)
    bowling_demand_high = (1 - chase_demand_low).where(is_chase, pace_high)
    first_innings_wickets_available = frame["wickets_remaining"] / 10
    bowling_resource = (1 - resource_mid).where(is_chase, pace_mid * first_innings_wickets_available)
    bowling_resource_low = (1 - resource_high).where(is_chase, pace_low * first_innings_wickets_available)
    bowling_resource_high = (1 - resource_low).where(is_chase, pace_high * first_innings_wickets_available)
    bowling_a, bowling_b = combine(bowling_demand, bowling_resource)
    bowling_a_low, bowling_b_low = combine(bowling_demand_low, bowling_resource_low)
    bowling_a_high, bowling_b_high = combine(bowling_demand_high, bowling_resource_high)

    frame["chase_rate_gap"] = gap_mid.where(is_chase)
    frame["chase_rate_gap_percentile"] = demand_rate_mid.where(is_chase)
    frame["remaining_target_fraction"] = target_fraction.where(is_chase)
    frame["batting_demand_component"] = batting_demand
    frame["bowling_threat_component"] = bowling_demand
    frame["bowling_resource_component"] = bowling_resource
    frame["batting_pressure_weighted"] = batting_a
    frame["batting_pressure_nonlinear"] = batting_b
    frame["bowling_pressure_weighted"] = bowling_a
    frame["bowling_pressure_nonlinear"] = bowling_b
    frame["batting_pressure_weighted_lower"] = batting_a_low
    frame["batting_pressure_weighted_upper"] = batting_a_high
    frame["batting_pressure_nonlinear_lower"] = batting_b_low
    frame["batting_pressure_nonlinear_upper"] = batting_b_high
    frame["bowling_pressure_weighted_lower"] = bowling_a_low
    frame["bowling_pressure_weighted_upper"] = bowling_a_high
    frame["bowling_pressure_nonlinear_lower"] = bowling_b_low
    frame["bowling_pressure_nonlinear_upper"] = bowling_b_high
    frame["pressure_index_reliability"] = np.where(
        frame["prior_legality_uncertain_deliveries"].gt(0),
        "bounded; prior delivery legality uncertain",
        np.where(
            frame["is_chasing"] & frame["chase_target"].isna(),
            "target unknown; neutral demand component",
            "point state from known prior legality",
        ),
    )
    return frame


def add_high_pressure_flags(
    scores: pd.DataFrame, config: PressureConfig | None = None
) -> pd.DataFrame:
    """Add candidate fixed and distribution-based high-pressure flags."""
    config = config or PressureConfig()
    frame = scores.copy()
    for role in ("batting", "bowling"):
        for formulation in ("weighted", "nonlinear"):
            column = f"{role}_pressure_{formulation}"
            fixed_thresholds = sorted(set((70.0, 80.0, 90.0, config.fixed_high_threshold)))
            for threshold in fixed_thresholds:
                frame[f"{role}_high_{formulation}_fixed_{int(threshold)}"] = frame[column].ge(threshold)
            frame[f"{role}_pressure_{formulation}_competition_percentile"] = frame.groupby(
                "competition_group", dropna=False
            )[column].rank(method="average", pct=True) * 100
            frame[f"{role}_pressure_{formulation}_context_percentile"] = frame.groupby(
                ["competition_group", "innings_phase"], dropna=False
            )[column].rank(method="average", pct=True) * 100
            for percentile in sorted(set((90, 95, *config.competition_percentiles))):
                frame[f"{role}_high_{formulation}_competition_p{percentile}"] = frame[
                    f"{role}_pressure_{formulation}_competition_percentile"
                ].ge(percentile)
            for percentile in sorted(set((90, 95, *config.context_percentiles))):
                frame[f"{role}_high_{formulation}_context_p{percentile}"] = frame[
                    f"{role}_pressure_{formulation}_context_percentile"
                ].ge(percentile)
    return frame
