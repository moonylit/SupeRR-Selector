# SupeRR Selector

An explainable cricket analytics project for the Rajasthan Royals SupeRR Selector Season 2 hackathon. The official Round 1 challenge is to identify the most reliable batters and bowlers under pressure using the supplied historical IPL and T20 International ball-by-ball data.

The analysis will compare pressure performance with each player's own baseline, account for cricket scoring rules and opportunity, quantify uncertainty, and publish reproducible ranked lists with a documented methodology. No player rankings are produced until the supplied data and methodology have been validated.

## Project status

**Phase 2 — dataset profiling and validation.** The source data has been inspected and a reproducible profiling pipeline is implemented. Structural, relational, and cricket-scoring reports and figures are generated from the official CSVs. Pressure detection, modelling, player selection, dashboard views, and the final competition report remain later phases.

## Repository layout

```text
SupeRR-Selector/
├── app/                  # Streamlit dashboard
├── data/                 # Local-only source and processed data
├── notebooks/            # Reproducible exploration notebooks
├── outputs/              # Generated figures, reports, and rankings
├── src/
│   ├── analysis/          # Exploratory and player analysis
│   ├── data/              # Loading, validation, preprocessing
│   ├── evaluation/        # Validation and sensitivity analysis
│   ├── models/            # Batting and bowling models
│   ├── pressure/          # Match context and pressure features
│   └── reporting/         # Rankings and report generation
├── tests/                 # Calculation and integration tests
├── config.py
├── main.py
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.10 or newer and Git. From the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

## Data handling

Keep the supplied ZIP and extracted CSV files in `data/raw/` on your machine. Both `data/raw/` and `data/processed/` are ignored by Git; do not commit source data, credentials, or local environment files. Preserve the original files. The six CSV schemas and their relationships are profiled in Phase 2. See [data/README.md](data/README.md).

## Run

Run the Phase 2 profiling pipeline from the repository root:

```powershell
python main.py
```

Run the automated calculation and validation tests with:

```powershell
python -m pytest -q
```

The pipeline reads CSVs from `data/raw/`, writes reports and figures under `outputs/`, and writes the derived innings summary under the ignored `data/processed/` directory. It preserves every input row and does not modify the source CSVs.

Phase 2 creates `dataset_profile.txt`, `dataset_profile.json`, `data_quality_report.txt`, `missing_values.csv`, `relational_validation.csv`, `cricket_scoring_validation.csv`, `competition_coverage.csv`, and `innings_scoring_summary.csv` under `outputs/reports/`. Eight standalone SVG figures are written under `outputs/figures/`.

## Phase 2 findings

The data pack contains 883 match records, 200,999 ball-by-ball rows, 141 competition catalog entries, 28,381 playing-eleven rows, 353 competition-team records, and 6,413 competition-team-player mappings. Match dates span 2022-01-22 through 2026-11-01. Competition metadata identifies five IPL seasons, 134 T20I competitions, and two other T20 events.

All delivery match IDs and all delivery batter/bowler player keys resolve to the supplied match and player mapping tables. Sixteen abandoned matches have no delivery records; three catalog competitions have no matches. The playing-eleven data has five unmapped non-playing roster rows and 35 repeated match/team/player keys. Extras include 264 `other` labels and two rows with nonzero extras but no type, so legal-ball and bowler-extra calculations retain explicit uncertainty. `matches.csv` has no independent innings score totals for external reconciliation. See the generated data-quality reports for details; no pressure scoring or player rankings have been produced.

The dashboard will be implemented after subsequent analytical outputs are available:

```powershell
streamlit run app/streamlit_app.py
```

## Reproducibility and methodology

Analysis settings will be centralized in `config.py`, including the random seed. Each cricket calculation, pressure definition, sample-size rule, baseline, and ranking component will be documented and validated before final selections are produced. See the project branches and pull requests for phase-by-phase progress.

## License and data

The supplied hackathon dataset is not redistributed by this repository. Dataset terms and any project license will be documented after the source materials are reviewed.
