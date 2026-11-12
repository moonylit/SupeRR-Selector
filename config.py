"""Shared project paths and reproducibility settings."""
from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RANDOM_SEED = 42


@dataclass(frozen=True)
class BattingModelConfig:
    """Reproducible Phase 5 modeling and sensitivity settings."""

    pressure_score_column: str = "batting_pressure_weighted"
    alternate_pressure_score_column: str = "batting_pressure_nonlinear"
    pressure_reference_minimum: int = 100
    pressure_percentile: float = 0.90
    pressure_band_percentiles: tuple[float, float, float] = (0.50, 0.75, 0.90)
    baseline_prior_strength_balls: float = 50.0
    minimum_context_pool_balls: int = 30
    eligibility_pressure_balls: int = 50
    eligibility_pressure_matches: int = 5
    eligibility_baseline_balls: int = 30
    pressure_ball_sensitivity: tuple[int, ...] = (30, 50, 75, 100)
    pressure_match_sensitivity: tuple[int, ...] = (5, 10, 15)
    fixed_pressure_thresholds: tuple[float, ...] = (70.0, 80.0, 90.0)
    pressure_percentile_sensitivity: tuple[float, ...] = (0.85, 0.90, 0.95)
    composite_weight_sensitivity: tuple[float, ...] = (0.25, 1 / 3, 0.50)
    bootstrap_replicates: int = 300
    ridge_alpha: float = 20.0
    minimum_regression_training_rows: int = 5000
    random_seed: int = RANDOM_SEED


BATTING_MODEL_CONFIG = BattingModelConfig()
