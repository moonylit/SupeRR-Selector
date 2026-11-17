"""Match-cluster bootstrap intervals for pressure performance."""

from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap_batter_pressure(deliveries: pd.DataFrame, replicates: int = 300, seed: int = 42) -> pd.DataFrame:
    """Bootstrap match aggregates (not individual balls) for P90 runs/ball."""
    rng = np.random.default_rng(seed)
    frame = deliveries.loc[deliveries["pressure_high_p90_walk_forward"].eq(True).fillna(False)
                          & deliveries["balls_faced_known"] & deliveries["striker_id"].notna()].copy()
    records = []
    for (competition, player), group in frame.groupby(["competition_group", "striker_id"], dropna=False):
        baseline = group.get("baseline_rpb_hierarchical", pd.Series(np.nan,index=group.index))
        group["_baseline"] = pd.to_numeric(baseline, errors="coerce")
        per_match = group.groupby("match_id").agg(runs=("batter_runs", "sum"), balls=("balls_faced_known", "sum"),
            baseline_sum=("_baseline", "sum"), baseline_n=("_baseline", "count"))
        if per_match.empty:
            continue
        vals = per_match.to_numpy(float)
        sampled = rng.integers(0, len(vals), size=(replicates, len(vals)))
        sums = vals[sampled].sum(axis=1)
        rpb = np.divide(sums[:, 0], sums[:, 1], out=np.full(replicates, np.nan), where=sums[:, 1] > 0)
        expected = np.divide(sums[:, 2], sums[:, 3], out=np.full(replicates, np.nan), where=sums[:, 3] > 0)
        difference = rpb - expected
        point = group["batter_runs"].sum() / group["balls_faced_known"].sum()
        baseline = group["_baseline"].mean()
        records.append({"competition_group": competition, "striker_id": player, "pressure_matches": len(per_match),
            "pressure_balls": int(group["balls_faced_known"].sum()), "pressure_rpb": point,
            "pressure_rpb_ci_low": float(np.nanquantile(rpb, .025)), "pressure_rpb_ci_high": float(np.nanquantile(rpb, .975)),
            "pressure_expected_baseline_rpb": baseline, "pressure_minus_baseline_rpb": point-baseline if np.isfinite(baseline) else np.nan,
            "pressure_minus_baseline_ci_low": float(np.nanquantile(difference, .025)) if np.isfinite(difference).any() else np.nan,
            "pressure_minus_baseline_ci_high": float(np.nanquantile(difference, .975)) if np.isfinite(difference).any() else np.nan,
            "bootstrap_replicates": replicates, "bootstrap_unit": "match"})
    return pd.DataFrame(records)
