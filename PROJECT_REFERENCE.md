# Project Reference

## 1. Purpose

This project studies the KSE-30 index and related index-fund behavior in Pakistan. It combines:

- market data extraction
- stock-level cleaning
- exploratory data analysis
- master dataset construction
- fund-flow proxy modeling
- volatility modeling
- market-efficiency testing
- rebalancing and weight-change prediction
- report production

The repository does not use a single official published "KSE-30 fund flow" series. Instead, later pipelines derive a proxy sector-flow signal from the NAV and AUM behavior of three KSE-30-related funds:

- `AKD`
- `NBP`
- `NTI`

## 2. Canonical Working Areas

Use these locations as primary:

- `7_codex_model/`: main self-contained pipeline
- `7_codex_model/data/`: local input data for the canonical pipeline
- `report_workspace_2/`: main report-writing workspace
- `0-docs/`: papers, proposal files, and report documents

Legacy but still useful:

- `5_claude_pipeline/`
- `6_cursor_model/`
- `report_workspace/`

Removed during cleanup:

- `8_last_model/`
- `9_march15_2026_backtest/`

## 3. Repository Map

### `0-docs/`

Stores proposal files, report PDFs and DOCX files, PSX-related research papers, and supporting literature.

Main subareas:

- `proposal/`
- `reports/`
- `psx-research-papers/`

### `0a_data_extraction/`

Early-stage extraction and acquisition work.

Important contents:

- `kse30_incremental_update.ipynb`
- `csvs/`
- `legacy-scripts/`

### `0b-raw-data/`

Raw source storage and archived input bundles.

Important contents:

- `csvs/`
- `xlsx/`
- `zips/`
- `extras/`

### `1a_data_cleaning/`

First-pass cleaning and simplification of KSE-30 daily files.

Important files:

- `create_kse30_basic.py`
- `kse-30-basic.csv`
- `kse-30-basic.xlsx`
- `kse30_daily_data.csv`

### `1b_eda/`

Exploratory analysis, descriptive statistics, correlation analysis, stationarity testing, and EDA figure production.

Important scripts:

- `eda_kse30.py`
- `eda_master_processed.py`

Important local datasets:

- `daily_master.csv`
- `monthly_master.csv`
- `kse30_stocks_clean.csv`
- `funds_data.xlsx`
- `macro_data.xlsx`
- `cpi.csv`
- `gold.csv`
- `inflation.xlsx`
- `gdp.xls`

Important output areas:

- `output/`
- `output_0/`
- `output_1/`
- `output_2/`

### `4_claude_model/`

Early task-by-task modeling stage with separate scripts per objective.

Core scripts:

- `nb0_preprocessing.py`
- `nb1_eda.py`
- `nb2_fund_flow_prediction.py`
- `nb3_garch_volatility.py`
- `nb4_portfolio_optimisation.py`
- `nb4b_rebalancing_prediction.py`
- `nb5_market_efficiency.py`
- `nb6_results_summary.py`
- `nb7_kse30_fund_flow_prediction.py`

### `4a_claude_model_merged/`

Merged-output bundle for an earlier Claude-based workflow.

Important contents:

- `merged_pipeline.py`
- `output_data/`
- `figures/`
- `README.md`

### `5_claude_pipeline/`

First large single-file end-to-end pipeline.

Core file:

- `pipeline.py`

Important inputs:

- `kse30_daily_data.csv`
- `funds_data.xlsx`
- `macro_data.xlsx`
- `cpi.csv`

Important outputs:

- `daily_master.csv`
- `monthly_master.csv`
- `results_fund_flow.csv`
- `results_garch.csv`
- `results_efficiency.csv`
- `results_rebalancing.csv`
- `results_rebalancing_forecast.csv`
- `figures/`

### `6_cursor_model/`

Improved integrated pipeline with methodology sidecar generation and rerun outputs.

Core files:

- `pipeline.py`
- `run_pipeline.py`
- `kse-30/recomposition_pipeline_kse30.py`

Important outputs:

- root-level result CSVs
- `figures/`
- `kse-30/`
- `kse-30-methodology-results/`
- `rerun_stationary/`
- `metrics.json`
- `metrics_updated.json`

### `7_codex_model/`

Canonical integrated pipeline after cleanup.

Core file:

- `pipeline.py`

Input area:

- `data/cpi.csv`
- `data/funds_data.xlsx`
- `data/kse30_daily_data.csv`
- `data/macro_data.xlsx`

Main outputs:

- `daily_master.csv`
- `monthly_master.csv`
- `kse30_stocks_clean.csv`
- `results_fund_flow.csv`
- `results_garch.csv`
- `results_efficiency.csv`
- `results_rebalancing.csv`
- `results_rebalancing_forecast.csv`
- `figures/`

Why it is canonical:

- self-contained inputs
- full integrated workflow in one place
- cleaner than the experimental and rerun-heavy variants

### `report_workspace/`

Legacy report-authoring workspace kept only as reference. Duplicate scripts were removed during cleanup.

### `report_workspace_2/`

Canonical report-authoring workspace.

Important contents:

- chapter folders for Chapters 3 to 8
- `appendices/`
- `graph_explanations/`
- `generate_additional_report_graphs.py`
- `generate_ch3_eda_graphs.py`
- `generate_rebalancing_risk_map_with_inbubble_labels.py`
- `transform_for_stationarity.py`
- `current-FYP Report (Analyzing Mutual Funds).md`
- `difference.md`
- `table-of-contents.txt`
- `table-of-contents-updated.txt`

### `overview/`

Lightweight summary material and supporting figures.

### `project-mds/`

Older folder-by-folder markdown notes.

### `qa/`

Answer bank and interpretation notes for expected project questions.

## 4. High-Level Workflow

1. Acquire raw source data in `0a_data_extraction/` and store it in `0b-raw-data/`.
2. Produce initial cleaned stock-level datasets in `1a_data_cleaning/`.
3. Explore the data and generate descriptive diagnostics in `1b_eda/`.
4. Build model-specific scripts in `4_claude_model/`.
5. Consolidate the workflow in `5_claude_pipeline/`.
6. Improve methodology and reruns in `6_cursor_model/`.
7. Keep the cleaned final integrated workflow in `7_codex_model/`.
8. Use `report_workspace_2/` to turn outputs into the written FYP report.

## 5. Canonical Pipeline Outputs

The canonical pipeline in `7_codex_model/` produces:

### Master datasets

- `daily_master.csv`: daily merged market, fund, and macro panel
- `monthly_master.csv`: monthly aggregated modeling panel
- `kse30_stocks_clean.csv`: clean stock-level dataset for downstream modeling

### Result tables

- `results_fund_flow.csv`
- `results_garch.csv`
- `results_efficiency.csv`
- `results_rebalancing.csv`
- `results_rebalancing_forecast.csv`

### Figures

The `figures/` directory is organized into:

- `eda/`
- `fund_flow/`
- `garch/`
- `efficiency/`
- `rebalancing/`
- `summary/`

## 6. Modeling Areas

### Fund-flow proxy modeling

Uses fund NAV, fund AUM, market returns, and macro variables to estimate sector-level flow behavior for AKD, NBP, and NTI.

### Volatility modeling

Estimates time-varying KSE-30 risk and backtest-style volatility diagnostics.

### Market-efficiency testing

Evaluates whether the market behaves like a random walk or shows predictable structure. Across the repo this includes stationarity checks, autocorrelation diagnostics, variance-ratio style evidence, and Granger-style predictability checks.

### Rebalancing and inclusion prediction

Predicts which stocks remain important and how their weights change in the index-related portfolio context.

## 7. Why Some Duplication Still Exists

Not all duplication is accidental:

- copied inputs inside model folders help older workflows run in isolation
- copied figures inside report folders support chapter-specific writing
- rerun outputs preserve experimental history

What this cleanup removed:

- obsolete final-model folders
- exact duplicate report-generation scripts in the legacy report workspace
- duplicate overview markdown that no longer added unique value

## 8. Practical Guidance

If you need to run analysis, use `7_codex_model/pipeline.py`.

If you need to inspect methodology evolution, compare `5_claude_pipeline/`, `6_cursor_model/`, and `7_codex_model/`.

If you need to edit the report, use `report_workspace_2/`.

If you need background papers, use `0-docs/`.

If you need descriptive figures or stationarity diagnostics, use `1b_eda/`.

## 9. Cleanup Summary

This restructure pass:

- removed `8_last_model/`
- removed `9_march15_2026_backtest/`
- removed duplicate source scripts from legacy `report_workspace/`
- removed redundant `overview/README_revised.md`
- established `7_codex_model/` as the canonical pipeline
- established `report_workspace_2/` as the canonical report workspace
- added a root `README.md`
- added this detailed `PROJECT_REFERENCE.md`
