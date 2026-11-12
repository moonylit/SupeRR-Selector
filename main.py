"""Command-line entry point for reproducible dataset profiling."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import DATA_PROCESSED_DIR, DATA_RAW_DIR, OUTPUTS_DIR
from src.data.loader import load_data_pack
from src.data.profiling import run_profiling


def main() -> None:
    """Profile source CSVs and write Phase 2 reports and figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("all", "profile", "eda"),
        default="all",
        help="Run Phase 2 profiling, Phase 3 EDA, or both (default: all).",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_RAW_DIR)
    parser.add_argument("--reports-dir", type=Path, default=OUTPUTS_DIR / "reports")
    parser.add_argument("--figures-dir", type=Path, default=OUTPUTS_DIR / "figures")
    parser.add_argument("--processed-dir", type=Path, default=DATA_PROCESSED_DIR)
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


if __name__ == "__main__":
    main()
