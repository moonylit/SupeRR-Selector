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


@dataclass(frozen=True)
class BowlingModelConfig:
    """Reproducible Phase 6 pressure labels, baselines, and evaluation settings."""

    weighted_pressure_column: str = "bowling_pressure_weighted"
    alternate_pressure_column: str = "bowling_pressure_nonlinear"
    pressure_reference_minimum: int = 100
    pressure_percentiles: tuple[float, float, float, float] = (0.25, 0.50, 0.75, 0.90)
    minimum_peer_baseline_balls: int = 60
    baseline_strategy: str = "format_hierarchical"
    baseline_coverage_minimum: float = 0.25
    economy_prior_balls: float = 60.0
    wicket_prior_balls: float = 180.0
    event_rate_prior_balls: float = 60.0
    extras_prior_balls: float = 120.0
    eligibility_pressure_balls: int = 36
    eligibility_pressure_matches: int = 3
    eligibility_pressure_innings: int = 3
    eligibility_baseline_balls: int = 24
    eligibility_scenarios: tuple[tuple[int, int, int, int], ...] = (
        (24, 2, 2, 12), (36, 3, 3, 24), (60, 5, 5, 36), (90, 8, 8, 60)
    )
    shrinkage_sensitivity_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0)
    bootstrap_replicates: int = 300
    random_seed: int = RANDOM_SEED


BOWLING_MODEL_CONFIG = BowlingModelConfig()
