# Q1: Explain the whole workflow from `6_cursor_model/pipeline.py`

`6_cursor_model/pipeline.py` is a single-script, end-to-end research pipeline for KSE-30 index fund flow and rebalancing analysis. It runs in this sequence:

1. **Setup and configuration (Section 0)**
- Sets paths: input data from `5_claude_pipeline/`, outputs in `6_cursor_model/`.
- Creates output folders (`figures/*`, sidecar results folder).
- Defines helper functions for metrics, chart saving, and a sidecar methodology script runner.

2. **Data ingestion and cleaning (Section 1)**
- Loads `kse30_daily_data.csv` and standardizes columns.
- Merges volume columns, converts numeric fields, normalizes company names.
- Removes trading-halt placeholder rows (Oct 2021) and invalid stale rows.
- Builds per-symbol features: log returns, rolling volatility, moving averages, computed FF market cap.
- Loads macro series from `macro_data.xlsx` (oil, interest rate, USD), CPI (`cpi.csv`), and optionally gold/GDP files.
- Loads and repairs fund NAV/AUM data from `funds_data.xlsx`, including zero-AUM gap repair.

3. **Master dataset construction (Section 2)**
- Reconstructs index-level daily series from constituent data.
- Builds monthly aggregates (index stats, macro, inflation, and total sector flow across AKD/NBP/NTI).
- Produces core modeling tables: daily and monthly master datasets.

4. **EDA (Section 3)**
- Generates descriptive statistics and exploratory plots for index behavior, volatility, and sector flow.

5. **GARCH volatility modeling (Section 4)**
- Fits/evaluates GARCH-style volatility models on index returns.
- Writes model-comparison outputs and volatility diagnostics.

6. **Aggregate fund-flow prediction (Section 5)**
- Predicts sector total flow (not per-fund) using train/test split (`TRAIN_END`).
- Compares Naive baseline vs ARIMAX vs VAR using RMSE/MAE/R2/directional accuracy.
- Saves forecasts and comparison tables.

7. **Market efficiency tests (Section 6)**
- Runs runs test, variance-ratio style checks, Ljung-Box type diagnostics, and Hurst-style persistence summary.
- Stores verdicts and test statistics.

8. **Rebalancing prediction (Section 7)**
- Builds rebalancing windows around historical KSE-30 rebalance dates.
- Two tasks:
  - **Weight prediction** (regression): Naive, Ridge, Random Forest.
  - **Inclusion/retention prediction** (classification): Naive, Logistic, Random Forest.
- Performs expanding-window CV for robustness.
- Produces forward next-rebalance risk forecast per symbol (retention probability, exclusion risk, predicted weight).

9. **Summary and outputs (Section 8)**
- Prints compact tables for all major sections.
- Creates a dashboard-style summary figure.
- Writes final artifacts, including:
  - `daily_master.csv`, `monthly_master.csv`, `kse30_stocks_clean.csv`
  - `results_garch.csv`, `results_fund_flow.csv`, `results_efficiency.csv`
  - `results_rebalancing.csv`, `results_rebalancing_weight_cv.csv`, `results_rebalancing_forecast.csv`
  - figures under `6_cursor_model/figures/`

10. **Sidecar methodology run**
- Also supports running `kse-30/recomposition_pipeline_kse30.py` and copying its outputs into `kse-30-methodology-results/` without altering main pipeline logic.

## Run command
From repo root:

```bash
python 6_cursor_model/pipeline.py
```
