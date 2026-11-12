"""Dependency-light SVG charts for data profiling outputs."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Sequence

import pandas as pd

WIDTH = 1200
BG = "#f5f7fb"
INK = "#182230"
MUTED = "#667085"
BLUE = "#245b8f"
ORANGE = "#d9822b"
GRID = "#d8dee8"


def _base(title: str, subtitle: str = "") -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="720" viewBox="0 0 {WIDTH} 720">',
        f'<rect width="100%" height="100%" fill="{BG}"/>',
        f'<text x="70" y="60" font-family="Arial,sans-serif" font-size="28" font-weight="700" fill="{INK}">{escape(title)}</text>',
        f'<text x="70" y="92" font-family="Arial,sans-serif" font-size="15" fill="{MUTED}">{escape(subtitle)}</text>',
    ]


def _save(parts: list[str], path: Path) -> None:
    parts.extend(['<text x="70" y="690" font-family="Arial,sans-serif" font-size="12" fill="#98a2b3">SupeRR Selector · dataset profiling · source rows preserved</text>', '</svg>'])
    path.write_text("\n".join(parts), encoding="utf-8")


def horizontal_bars(
    labels: Sequence[str], values: Sequence[float], title: str, subtitle: str, path: Path,
    *, value_format: str = "{:,.0f}", color: str = BLUE,
) -> None:
    """Render a clear horizontal bar chart as a standalone SVG."""
    labels, values = list(labels), list(values)
    if not labels:
        labels, values = ["No data"], [0]
    parts = _base(title, subtitle)
    left, right, top, row_h = 330, 1090, 125, 20 if len(labels) > 18 else 34
    max_value = max(values) if values else 0
    plot_w = right - left
    ticks = 5
    for index in range(ticks + 1):
        x = left + plot_w * index / ticks
        parts.append(f'<line x1="{x:.1f}" y1="{top-10}" x2="{x:.1f}" y2="{top+row_h*len(labels)}" stroke="{GRID}"/>')
        tick_value = max_value * index / ticks
        parts.append(f'<text x="{x:.1f}" y="{top-18}" text-anchor="middle" font-family="Arial" font-size="11" fill="{MUTED}">{tick_value:,.0f}</text>')
    for index, (label, value) in enumerate(zip(labels, values)):
        y = top + index * row_h
        bar_w = 0 if max_value <= 0 else plot_w * float(value) / max_value
        parts.append(f'<text x="{left-14}" y="{y+15}" text-anchor="end" font-family="Arial" font-size="12" fill="{INK}">{escape(str(label)[:46])}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{bar_w:.2f}" height="19" rx="4" fill="{color}"/>')
        parts.append(f'<text x="{min(right+6,left+bar_w+7):.1f}" y="{y+15}" font-family="Arial" font-size="11" fill="{INK}">{value_format.format(value)}</text>')
    _save(parts, path)


def histogram(values: pd.Series, title: str, subtitle: str, path: Path, *, max_bin: int = 30) -> None:
    """Render an integer-valued histogram as SVG."""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if max_bin > 20:
        width = 10
        edges = list(range(0, max_bin + width, width))
        categorized = pd.cut(clean.clip(upper=max_bin), bins=edges, right=False, include_lowest=True)
        bins = categorized.value_counts().sort_index()
        labels = [f"{int(interval.left)}–{int(interval.right)-1}" for interval in bins.index]
        if labels:
            labels[-1] = f"{max_bin}+"
    else:
        bins = clean.clip(upper=max_bin).value_counts().sort_index()
        labels = [f"{int(i)}+" if int(i) == max_bin else str(int(i)) for i in bins.index]
    horizontal_bars(labels, bins.astype(int).tolist(), title, subtitle, path, value_format="{:,.0f}", color=ORANGE)


def line_chart(labels: Sequence[str], values: Sequence[float], title: str, subtitle: str, path: Path) -> None:
    """Render a line chart with visible labels and a light grid."""
    parts = _base(title, subtitle)
    left, right, top, bottom = 95, 1130, 150, 610
    vals = [float(v) for v in values]
    max_value = max(vals) if vals else 1
    max_value = max(max_value, 1)
    for i in range(6):
        y = bottom - (bottom - top) * i / 5
        value = max_value * i / 5
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{GRID}"/>')
        parts.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="12" fill="{MUTED}">{value:,.0f}</text>')
    points = []
    denom = max(1, len(vals)-1)
    for i, (label, value) in enumerate(zip(labels, vals)):
        x = left + (right-left) * i / denom
        y = bottom - (bottom-top) * value / max_value
        points.append((x, y))
        if len(labels) <= 30 or i % max(1, len(labels)//20) == 0:
            parts.append(f'<text x="{x:.1f}" y="{bottom+22}" text-anchor="middle" font-family="Arial" font-size="10" fill="{MUTED}">{escape(str(label))}</text>')
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    parts.append(f'<polyline points="{polyline}" fill="none" stroke="{BLUE}" stroke-width="3"/>')
    for x, y in points:
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{ORANGE}"/>')
    _save(parts, path)


def generate_profile_figures(
    tables: dict[str, pd.DataFrame], profiles: dict[str, dict[str, object]], out_dir: str | Path
) -> list[Path]:
    """Create the requested initial plots without lossy image dependencies."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    paths = [
        ("Records per dataset", [key.replace("_", " ") for key in profiles], [int(v["rows"]) for v in profiles.values()], "CSV row count"),
    ]
    for title, labels, values, subtitle in paths:
        target = root / "records_per_dataset.svg"
        horizontal_bars(labels, values, title, subtitle, target)
        created.append(target)

    missing = {name: sum(int(item["missing_count"]) for item in meta["columns"].values()) for name, meta in profiles.items()}
    target = root / "missing_values_by_dataset.svg"
    horizontal_bars(list(missing), list(missing.values()), "Missing values by dataset", "Total missing cells across columns", target)
    created.append(target)

    matches = tables["matches"].merge(
        tables["competitions"][["comp_id", "match_format", "season", "comp_name"]],
        on="comp_id", how="left", validate="many_to_one"
    )
    season_counts = matches.groupby(["match_format", "season"], dropna=False).size()
    labels = [f"{fmt} {season}" for fmt, season in season_counts.index]
    target = root / "matches_by_season.svg"
    horizontal_bars(labels, season_counts.tolist(), "Matches by competition season", "Competition format kept explicit", target)
    created.append(target)

    comp_counts = matches.groupby(["comp_name", "match_format"], dropna=False).size().sort_values(ascending=False).head(25)
    target = root / "matches_by_competition.svg"
    horizontal_bars([f"{name} · {fmt}" for (name, fmt) in comp_counts.index], comp_counts.tolist(), "Matches by competition", "Top 25 by match count; full coverage is in competition_coverage.csv", target)
    created.append(target)

    player_counts = tables["competition_team_player_mapping"].groupby(["comp_name", "comp_id"]).player_id.nunique().sort_values(ascending=False).head(25)
    names = tables["competitions"].set_index("comp_id").comp_name.to_dict()
    target = root / "players_by_competition.svg"
    horizontal_bars([names.get(int(cid), str(cid)) for _, cid in player_counts.index], player_counts.tolist(), "Mapped players by competition", "Top 25 competitions; counts use competition player mapping", target)
    created.append(target)

    dated = matches.copy()
    dated["month"] = pd.to_datetime(dated["match_date"], errors="coerce").dt.to_period("M").astype("string")
    monthly = dated.groupby("month", dropna=False).size().sort_index()
    target = root / "match_date_coverage.svg"
    line_chart(monthly.index.astype(str).tolist(), monthly.tolist(), "Match date coverage", "Matches per month; all competition formats", target)
    created.append(target)

    from src.data.cricket_calculations import calculate_innings_summary

    innings = calculate_innings_summary(tables["ball_by_ball"])
    target = root / "innings_total_distribution.svg"
    histogram(innings.total_runs, "Distribution of innings totals", "Total runs derived from runs + recorded extras; capped display bin at 30", target, max_bin=300)
    created.append(target)
    target = root / "wickets_per_innings_distribution.svg"
    histogram(innings.wickets, "Distribution of wickets per innings", "Wicket events in ball-by-ball data", target, max_bin=10)
    created.append(target)
    return created
