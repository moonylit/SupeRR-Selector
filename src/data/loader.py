"""CSV loading helpers for the official SupeRR Selector data pack."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


def load_dataset(path: str | Path, **read_csv_kwargs: object) -> pd.DataFrame:
    """Load one CSV without altering its source file.

    Args:
        path: CSV path on disk.
        **read_csv_kwargs: Optional arguments forwarded to ``pandas.read_csv``.

    Raises:
        FileNotFoundError: If the requested CSV does not exist.
        ValueError: If the path is not a CSV or cannot be parsed as one.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV file, got: {csv_path}")
    try:
        frame = pd.read_csv(csv_path, **read_csv_kwargs)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        LOGGER.exception("Could not parse CSV %s", csv_path)
        raise ValueError(f"Could not parse CSV {csv_path}: {exc}") from exc
    LOGGER.info("Loaded %s rows and %s columns from %s", *frame.shape, csv_path)
    return frame


def load_data_pack(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load the six expected CSVs from a data directory; report missing files."""
    root = Path(data_dir)
    filenames = (
        "ball_by_ball.csv",
        "matches.csv",
        "competitions.csv",
        "playing_eleven.csv",
        "teams.csv",
        "competition_team_player_mapping.csv",
    )
    missing = [name for name in filenames if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing required dataset file(s) in {root}: {', '.join(missing)}"
        )
    return {Path(name).stem: load_dataset(root / name) for name in filenames}
