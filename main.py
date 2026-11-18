"""Command-line entry point for reproducible SupeRR Selector analysis phases."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import DATA_PROCESSED_DIR, DATA_RAW_DIR, OUTPUTS_DIR
from src.data.loader import load_data_pack
from src.data.profiling import run_profiling


def main() -> None:
    """Run the selected profiling, EDA, and pressure-analysis phases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("all", "profile", "eda", "pressure", "batting", "bowling", "final"),
        default="all",
        help="Run one analytical phase or all phases, including bowling pressure (default: all).",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_RAW_DIR)
    parser.add_argument("--reports-dir", type=Path, default=OUTPUTS_DIR / "reports")
    parser.add_argument("--figures-dir", type=Path, default=OUTPUTS_DIR / "figures")
    parser.add_argument("--processed-dir", type=Path, default=DATA_PROCESSED_DIR)
    parser.add_argument("--pressure-processed-dir", type=Path, default=OUTPUTS_DIR / "processed")
    parser.add_argument("--bootstrap-replicates", type=int, default=None,
                        help="Override the Phase 6 match-bootstrap replicate count.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.phase in ("all", "profile"):
        result = run_profiling(args.data_dir, args.reports_dir, args.figures_dir, args.processed_dir)
        print(
            f"Profiled {len(result['tables'])} datasets; "
            f"wrote {len(result['figures'])} Phase 2 figures and {len(result['innings'])} innings summaries."
        )
        tables = result["tables"]
    else:
        tables = load_data_pack(args.data_dir)

    if args.phase in ("all", "eda"):
        from src.analysis.exploratory_analysis import run_eda

        result = run_eda(tables, args.reports_dir, args.figures_dir / "eda")
        print(
            f"EDA wrote {len(result['competition'])} competition rows, "
            f"{len(result['batting'])} batting rows, {len(result['bowling'])} bowling rows, "
            f"and {len(result['figures'])} figures."
        )

    if args.phase in ("all", "pressure", "batting", "bowling", "final"):
        from src.pressure.pressure_validation import run_pressure_engine

        result = run_pressure_engine(
            tables,
            args.reports_dir,
            args.figures_dir / "pressure",
            args.pressure_processed_dir,
        )
        print(
            f"Pressure engine wrote {len(result['scores'])} delivery scores, "
            f"{len(result['thresholds'])} threshold sensitivity rows, and "
            f"{len(result['figures'])} figures."
        )

        if args.phase in ("all", "batting", "final"):
            from src.models.batting_model import run_batting_model
            batting = run_batting_model(
                tables,
                result["scores"],
                result["scores"],
                result["scores"],
                args.reports_dir,
                args.figures_dir / "batting",
                args.pressure_processed_dir,
            )
            print(
                f"Batting model wrote {len(batting['deliveries'])} delivery rows, "
                f"{len(batting['scores'])} player/competition records, "
                f"and {len(batting['figures'])} figures."
            )

            if args.phase == "final":
                import gc

                batting_scores = batting["scores"]
                batting_sensitivity = batting["sensitivity"]
                batting_role_evidence = batting["deliveries"][
                    ["competition_group", "striker_id", "batting_order", "match_id", "season",
                     "pressure_high_p90_walk_forward", "balls_faced_known"]
                ].copy()
                del batting
                gc.collect()

        if args.phase in ("all", "bowling", "final"):
            import gc

            from src.models.bowling_model import run_bowling_model

            if "batting" in locals():
                del batting
                gc.collect()
            bowling = run_bowling_model(
                tables,
                result["scores"],
                result["scores"],
                result["scores"],
                args.reports_dir,
                args.figures_dir / "bowling",
                args.pressure_processed_dir,
                bootstrap_replicates=args.bootstrap_replicates,
            )
            print(
                f"Bowling model wrote {len(bowling['deliveries'])} delivery rows, "
                f"{len(bowling['scores'])} bowler/competition records, "
                f"and {len(bowling['figures'])} figures."
            )

            if args.phase == "final":
                from src.models.final_strategy import run_final_strategy

                bowling_role_evidence = bowling["deliveries"][
                    ["competition_group", "bowler_id", "innings_phase", "season"]
                ].copy()
                final = run_final_strategy(
                    batting_scores,
                    bowling["scores"],
                    batting_sensitivity,
                    batting_role_evidence,
                    bowling_role_evidence,
                    args.reports_dir,
                    args.figures_dir,
                    args.pressure_processed_dir,
                    bowling["sensitivity"],
                )
                print(
                    f"Final strategy wrote {len(final['batters'])} IPL batters and "
                    f"{len(final['bowlers'])} IPL bowlers; report: "
                    f"{args.reports_dir / 'final_strategy_report.md'}"
                )


if __name__ == "__main__":
    main()
