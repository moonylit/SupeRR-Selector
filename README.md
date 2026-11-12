# SupeRR Selector

An explainable cricket analytics project for the Rajasthan Royals SupeRR Selector Season 2 hackathon. The official Round 1 challenge is to identify the most reliable batters and bowlers under pressure using the supplied historical IPL and T20 International ball-by-ball data.

The analysis will compare pressure performance with each player's own baseline, account for cricket scoring rules and opportunity, quantify uncertainty, and publish reproducible ranked lists with a documented methodology. No player rankings are produced until the supplied data and methodology have been validated.

## Project status

**Phase 1 — project setup.** The repository scaffold is ready. Dataset profiling, pressure detection, modelling, validation, rankings, dashboard, and final report are planned as subsequent phases. No source data or analytical findings are included yet.

## Repository layout

```text
SupeRR-Selector/
├── app/                  # Streamlit dashboard
├── data/                 # Local-only source and processed data
├── notebooks/            # Reproducible exploration notebooks
├── outputs/               # Generated figures, reports, and rankings
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

Keep the supplied ZIP and extracted CSV files in `data/raw/` on your machine. Both `data/raw/` and `data/processed/` are ignored by Git; do not commit source data, credentials, or local environment files. Preserve the original files. The required CSV schemas and relationships will be inspected in the dataset-profiling phase before analysis is implemented. See [data/README.md](data/README.md).

## Run

The analysis pipeline will be wired up after profiling:

```powershell
python main.py
```

The dashboard will be implemented after the analytical outputs are available:

```powershell
streamlit run app/streamlit_app.py
```

## Reproducibility and methodology

Analysis settings will be centralized in `config.py`, including the random seed. Each cricket calculation, pressure definition, sample-size rule, baseline, and ranking component will be documented and validated before final selections are produced. See the project branches and pull requests for phase-by-phase progress.

## License and data

The supplied hackathon dataset is not redistributed by this repository. Dataset terms and any project license will be documented after the source materials are reviewed.
