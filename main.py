"""Command-line entry point for reproducible dataset profiling."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import DATA_PROCESSED_DIR, DATA_RAW_DIR, OUTPUTS_DIR
from src.data.profiling import run_profiling


def main() -> None:
    """Profile source CSVs and write Phase 2 reports and figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_RAW_DIR)
    parser.add_argument("--reports-dir", type=Path, default=OUTPUTS_DIR / "reports")
    parser.add_argument("--figures-dir", type=Path, default=OUTPUTS_DIR / "figures")
    parser.add_argument("--processed-dir", type=Path, default=DATA_PROCESSED_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = run_profiling(args.data_dir, args.reports_dir, args.figures_dir, args.processed_dir)
    print(
        f"Profiled {len(result['tables'])} datasets; "
        f"wrote {len(result['figures'])} figures and {len(result['innings'])} innings summaries."
    )


if __name__ == "__main__":
    main()
