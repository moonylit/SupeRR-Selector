"""Required EDA charts built with the shared SVG plotting helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.cricket_calculations import calculate_innings_summary
from src.data.visualizations import horizontal_bars, histogram, multi_line_chart


def _numeric_histogram(
    values: pd.Series, *, maximum: int, width: int, title: str, subtitle: str, path: Path
) -> None:
    clean = pd.to_numeric(values, errors="coerce").dropna().clip(lower=0, upper=maximum)
    edges = np.arange(0, maximum + width + 1, width)
    categories = pd.cut(clean, bins=edges, right=False, include_lowest=True)
    counts = categories.value_counts().sort_index()
    labels = [f"{int(interval.left)}–<{int(interval.right)}" for interval in counts.index]
    if labels:
        labels[-1] = f"{int(counts.index[-1].left)}+ (display cap)"
    horizontal_bars(labels, counts.astype(int).tolist(), title, subtitle, path)


def _group_series(
    frame: pd.DataFrame, *, group_columns: list[str], value: str, labels: list[str],
    index_range: range | None = None,
) -> tuple[list[str], dict[str, list[float]]]:
    pivot = frame.pivot_table(index=labels[0], columns=labels[1:], values=value, aggfunc="median")
    if index_range is not None:
        pivot = pivot.reindex(index_range)
    xlabels = [str(value) for value in pivot.index]
    result: dict[str, list[float]] = {}
    for column in pivot.columns:
        values = pivot[column].tolist()
        name = " · ".join(str(v) for v in (column if isinstance(column, tuple) else (column,)))
        result[name] = [float(value) if pd.notna(value) else np.nan for value in values]
    return xlabels, result


def generate_eda_figures(
    *,
    tables: dict[str, pd.DataFrame],
    deliveries: pd.DataFrame,
    competition: pd.DataFrame,
    batting: pd.DataFrame,
    bowling: pd.DataFrame,
    phases: pd.DataFrame,
    consistency: pd.DataFrame,
    context: pd.DataFrame,
    innings_context: pd.DataFrame,
    states: pd.DataFrame,
    out_dir: str | Path,
) -> list[Path]:
    """Generate format-aware distributions, comparisons, and state progression plots."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    groups = competition.groupby("competition_group").matches.sum().sort_values(ascending=False)
    target = root / "competition_match_distribution.svg"
    horizontal_bars(groups.index.astype(str).tolist(), groups.tolist(), "Matches by competition group", "Counts include abandoned matches with no delivery rows", target)
    paths.append(target)

    seasonal = tables["matches"].merge(
        tables["competitions"][["comp_id", "comp_name", "match_format", "season"]],
        on="comp_id", how="left", validate="many_to_one",
    )
    from src.analysis.exploratory_analysis import competition_group

    seasonal["competition_group"] = competition_group(seasonal)
    season_counts = seasonal.groupby(["competition_group", "season"]).match_id.nunique().sort_values(ascending=False)
    target = root / "season_match_distribution.svg"
    horizontal_bars([f"{group} · {season}" for group, season in season_counts.index], season_counts.tolist(), "Matches by season and competition group", "One bar per format-labeled season", target)
    paths.append(target)

    innings = calculate_innings_summary(tables["ball_by_ball"])
    target = root / "innings_total_distribution.svg"
    histogram(innings.total_runs, "Distribution of innings totals", "Runs + recorded extras; standard innings and tie-break innings retained", target, max_bin=300)
    paths.append(target)

    eligible_bat = batting.loc[batting.minimum_60_known_balls]
    target = root / "batting_strike_rate_distribution.svg"
    _numeric_histogram(eligible_bat.strike_rate_known_balls, maximum=300, width=20, title="Player batting strike-rate distribution", subtitle="Players with at least 60 known balls; known balls only, display capped at 300", path=target)
    paths.append(target)

    eligible_bowl = bowling.loc[bowling.minimum_24_known_legal_balls]
    target = root / "bowling_economy_distribution.svg"
    _numeric_histogram(eligible_bowl.economy_known, maximum=30, width=2, title="Player bowling-economy distribution", subtitle="Players with at least 24 known legal balls; known-ball economy, display capped at 30", path=target)
    paths.append(target)

    standard_phases = phases.loc[~phases.innings_phase.isin(["Uncertain phase", "Tie-break / super-over innings"])]
    plot_phases = standard_phases.groupby(["competition_group", "innings_phase"], as_index=False).agg(
        runs=("total_runs", "sum"), wickets=("bowler_attributable_wickets", "sum"),
        legal=("known_legal_balls", "sum"),
    )
    phase_labels = [f"{row.competition_group} · {row.innings_phase}" for row in plot_phases.itertuples()]
    target = root / "runs_by_innings_phase.svg"
    horizontal_bars(phase_labels, plot_phases.runs.tolist(), "Runs by innings phase", "Aggregate total runs; uncertain-phase and tie-break rows excluded", target)
    paths.append(target)
    target = root / "wickets_by_innings_phase.svg"
    horizontal_bars(phase_labels, plot_phases.wickets.tolist(), "Bowler-attributable wickets by phase", "Caught, bowled, LBW, stumped, and hit wicket only", target)
    paths.append(target)

    target = root / "batting_first_vs_second.svg"
    batting_context = innings_context.copy()
    first_second_batting = batting_context[["competition_group", "innings_context", "batter_strike_rate_known_balls"]].dropna()
    horizontal_bars(
        [f"{row.competition_group} · {row.innings_context}" for row in first_second_batting.itertuples()],
        first_second_batting.batter_strike_rate_known_balls.tolist(),
        "Batting strike rate: first versus second innings",
        "Aggregate batter runs per known ball faced; context refers to batting first or chasing",
        target,
    )
    paths.append(target)

    target = root / "bowling_first_vs_second.svg"
    first_second_bowling = batting_context[["competition_group", "innings_context", "bowling_economy_known_balls"]].dropna()
    horizontal_bars(
        [f"{row.competition_group} · {row.innings_context}" for row in first_second_bowling.itertuples()],
        first_second_bowling.bowling_economy_known_balls.tolist(),
        "Bowling economy: first versus second innings",
        "Aggregate runs conceded per six known legal balls; bowled in first versus chasing innings",
        target,
    )
    paths.append(target)

    eligible_bat_consistency = consistency.loc[consistency.batting_consistency_minimum_met]
    target = root / "batter_match_run_variability.svg"
    _numeric_histogram(eligible_bat_consistency.match_runs_std, maximum=60, width=5, title="Batter match-run standard deviation", subtitle="At least five batting matches; descriptive variability, not reliability ranking", path=target)
    paths.append(target)

    eligible_bowl_consistency = consistency.loc[consistency.bowling_consistency_minimum_met]
    target = root / "bowler_match_economy_variability.svg"
    _numeric_histogram(eligible_bowl_consistency.qualified_match_economy_std, maximum=15, width=1, title="Bowler match-economy standard deviation", subtitle="At least five matches and five spells of 12+ known legal balls", path=target)
    paths.append(target)

    chase_states = states.loc[states.innings.eq(2) & states.chase_target.notna()]
    target = root / "required_run_rate_distribution.svg"
    _numeric_histogram(chase_states.required_run_rate_lower, maximum=40, width=2, title="Chase required-run-rate distribution", subtitle="State before each delivery; lower bound assumes uncertain prior deliveries were legal", path=target)
    paths.append(target)

    target = root / "wickets_remaining_distribution.svg"
    wicket_counts = chase_states.wickets_remaining.value_counts().reindex(range(11), fill_value=0)
    horizontal_bars([str(v) for v in wicket_counts.index], wicket_counts.tolist(), "Wickets remaining during chases", "Delivery states before each ball; wicket events already recorded in the innings are deducted", target)
    paths.append(target)

    current_states = states.copy()
    current_states["innings_label"] = np.where(current_states.innings.eq(1), "First innings", "Chase")
    xlabels, current_series = _group_series(
        current_states,
        group_columns=["competition_group", "innings_label"],
        value="current_run_rate_lower",
        labels=["known_balls_used_over_bin", "competition_group", "innings_label"],
        index_range=range(21),
    )
    target = root / "current_run_rate_progression.svg"
    multi_line_chart(xlabels, current_series, "Run-rate progression", "Median current run rate before delivery by known legal over; uncertain balls lower the denominator bound", target)
    paths.append(target)

    chase_states["innings_label"] = chase_states["competition_group"]
    xlabels, required_series = _group_series(
        chase_states,
        group_columns=["competition_group"],
        value="required_run_rate_lower",
        labels=["known_balls_used_over_bin", "competition_group"],
        index_range=range(21),
    )
    target = root / "required_run_rate_progression.svg"
    multi_line_chart(xlabels, required_series, "Chase required-run-rate progression", "Median lower-bound required rate before delivery by competition group", target)
    paths.append(target)

    xlabels, wickets_series = _group_series(
        chase_states,
        group_columns=["competition_group"],
        value="wickets_remaining",
        labels=["known_balls_used_over_bin", "competition_group"],
        index_range=range(21),
    )
    target = root / "wickets_remaining_progression.svg"
    multi_line_chart(xlabels, wickets_series, "Wickets remaining during chases", "Median wickets remaining before delivery by known legal over and competition group", target)
    paths.append(target)
    return paths
