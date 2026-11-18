# SupeRR Selector

An explainable cricket analytics project for the Rajasthan Royals SupeRR Selector Season 2 hackathon. The official Round 1 challenge is to identify the most reliable batters and bowlers under pressure using the supplied historical IPL and T20 International ball-by-ball data.

The analysis will compare pressure performance with each player's own baseline, account for cricket scoring rules and opportunity, quantify uncertainty, and publish reproducible ranked lists with a documented methodology. No player rankings are produced until the supplied data and methodology have been validated.

## Project status

**Phase 7 — final auction strategy.** Phases 2–6 provide dataset profiling, exploratory analysis, pre-delivery match states and pressure scores, and separate batting and bowling pressure evaluations. Phase 7 produces separate provisional IPL top-ten batter and bowler historical-performance shortlists from the Phase 5/6 eligibility and scoring outputs, with observed-role context, robustness evidence, a report, and a comparison chart. It does not combine formats or roles. These are not auction-eligible player pools or purchase recommendations: current official retention information identifies several ranked players as retained, while the final registered auction pool and the RR budget/role brief are not model inputs. All indices and scores describe observed outcomes under defined cricket contexts; they are not psychological measurements, causal pressure effects, or validated future performance predictions.

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

Run both dataset profiling and exploratory analysis from the repository root:

```powershell
python main.py
```

The phases can also be run separately:

```powershell
python main.py --phase profile
python main.py --phase eda
python main.py --phase pressure
python main.py --phase batting
python main.py --phase bowling
python main.py --phase final
```

Run the automated calculation and validation tests with:

```powershell
python -m pytest -q
```

The pipeline reads CSVs from `data/raw/`, writes reports and figures under `outputs/`, and writes the derived innings summary under the ignored `data/processed/` directory. It preserves every input row and does not modify the source CSVs.

Phase 2 creates `dataset_profile.txt`, `dataset_profile.json`, `data_quality_report.txt`, `missing_values.csv`, `relational_validation.csv`, `cricket_scoring_validation.csv`, `competition_coverage.csv`, and `innings_scoring_summary.csv` under `outputs/reports/`. Eight standalone SVG figures are written under `outputs/figures/`.

Phase 3 adds `eda_summary.txt`, `eda_summary.json`, `competition_analysis.csv`, `batting_analysis.csv`, `bowling_analysis.csv`, `innings_phase_analysis.csv`, `innings_context_analysis.csv`, `player_consistency.csv`, and `match_context_analysis.csv` under `outputs/reports/`, plus 16 standalone SVG charts under `outputs/figures/eda/`. Player opportunity and match-count thresholds only flag more interpretable descriptive comparisons; they do not establish statistical reliability or create rankings. Delivery states use information available before the delivery, and match outcomes are kept out of those state features. Uncertain delivery legality, extras attribution, and phase assignment remain explicitly flagged.

Phase 4 writes the generated `delivery_match_states.csv`, `pressure_features.csv`, and `pressure_scores.csv` under the local-only `outputs/processed/` directory. The reproducible, shareable reports are `pressure_methodology.txt`, `pressure_validation.txt`, `pressure_distribution.csv`, `pressure_threshold_analysis.csv`, and `pressure_scenario_tests.csv` under `outputs/reports/`, with eight standalone charts under `outputs/figures/pressure/`. Match-state features use information available before each delivery. Unknown legal deliveries are carried as explicit lower/upper bounds; pressure percentiles use only information through the previous calendar month within each competition group and innings phase. Match outcomes are excluded from score construction. With no independent pressure ground truth, the reports recommend context-adjusted P90 as a provisional relative label for high-pressure opportunities and compare it with fixed and competition-only percentile thresholds. This recommendation is not a validated player-ranking threshold.

Phase 5 writes eight local-only player/delivery tables under `outputs/processed/`, including delivery features, player-match and pressure statistics, baselines, differentials, scores, eligibility, and match-bootstrap uncertainty. Reports under `outputs/reports/` document methodology, model comparison, validation, eligibility sensitivity, pressure sensitivity, and key findings. Sixteen charts are generated under `outputs/figures/batting/`. The primary P90 labels use only earlier months within competition group and innings phase; personal baselines use only earlier months of non-P90, known-faced observations and use peer shrinkage when history is sparse. Unknown extras remain flagged. All players remain in outputs, and scores are exploratory rather than final top-10 selections.

Phase 5 thresholds and reproducibility settings are centralized in `BATTING_MODEL_CONFIG` in `config.py`. Run `python main.py --phase batting` to regenerate Phase 4 inputs and the batting outputs together. Derived delivery-level CSVs remain ignored by Git.

Phase 6 reuses the distinct Phase 4 `bowling_pressure_weighted` index and pre-delivery state, then assigns high-pressure opportunities using a prior-month P90 by competition group and innings phase. Walk-forward normal-pressure baselines use only earlier calendar dates; same-date matches are excluded. Bowling rates use confirmed legal-ball denominators, bowler-charged runs, and credited wickets, with uncertain legality and extras preserved as bounds. Match bootstrap draws resample matches jointly across bowlers; intervals are conditional on the fixed historical baseline, consistency, and evidence-volume reliability estimates. The four evaluation dimensions remain visible alongside four bounded composite models.

Run `python main.py --phase bowling` to regenerate Phase 4 inputs and Phase 6 outputs together. It writes the detailed delivery and player tables under ignored `outputs/processed/`, five methodology/validation/comparison/sensitivity/findings reports under `outputs/reports/`, and 14 figures under `outputs/figures/bowling/`. The default eligibility rules require 36 high-pressure legal balls across at least three matches and innings, 24 prior normal-pressure legal balls, and 25% baseline coverage. All bowlers remain in the eligibility table with exclusion reasons. IPL, T20I, and Other T20 results are kept separate; a group with no eligible records has no provisional ranking under the default threshold.

Run `python main.py --phase final` to regenerate the shared Phase 4 inputs, both Phase 5/6 evaluations, and the final strategy outputs. The two IPL-only shortlists, complete eligible candidate tables, joined evidence table, and audit metadata are written under ignored `outputs/processed/`; the narrative report is `outputs/reports/final_strategy_report.md`, and a score comparison is written to `outputs/figures/final_strategy/ipl_top10_shortlists.png`. Batters and bowlers use their own existing Phase 5/6 composite scores. Role descriptions are historical usage summaries. Alternative-score and pressure-definition retention is reported for batting; alternative-score and match-bootstrap top-ten stability is reported for bowling. The report includes a dated external retention check but does not treat the remaining players as available without an official registered auction list; it does not produce price ceilings or budget-specific recommendations.

## Phase 2 findings

The data pack contains 883 match records, 200,999 ball-by-ball rows, 141 competition catalog entries, 28,381 playing-eleven rows, 353 competition-team records, and 6,413 competition-team-player mappings. Match dates span 2022-01-22 through 2026-11-01. Competition metadata identifies five IPL seasons, 134 T20I competitions, and two other T20 events.

All delivery match IDs and all delivery batter/bowler player keys resolve to the supplied match and player mapping tables. Sixteen abandoned matches have no delivery records; three catalog competitions have no matches. The playing-eleven data has five unmapped non-playing roster rows and 35 repeated match/team/player keys. Extras include 264 `other` labels and two rows with nonzero extras but no type, so legal-ball and bowler-extra calculations retain explicit uncertainty. `matches.csv` has no independent innings score totals for external reconciliation. See the generated data-quality reports for details. Phase 4 pressure indices are descriptive candidates only; player rankings have not been produced.

The dashboard will be implemented after subsequent analytical outputs are available:

```powershell
streamlit run app/streamlit_app.py
```

## Reproducibility and methodology

Analysis settings will be centralized in `config.py`, including the random seed. Each cricket calculation, pressure definition, sample-size rule, baseline, and ranking component will be documented and validated before final selections are produced. See the project branches and pull requests for phase-by-phase progress.

## License and data

The supplied hackathon dataset is not redistributed by this repository. Dataset terms and any project license will be documented after the source materials are reviewed.
