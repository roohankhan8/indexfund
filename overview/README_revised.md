# FYP Index Funds - Project Overview

**Project title:** Analyzing KSE-30 Using Quantitative Finance and Machine Learning

## What This Project Is About

This project studies the **KSE-30 index** and the behavior of **KSE-30-related index funds** in Pakistan. The system combines data engineering, exploratory analysis, econometric modeling, and machine learning to answer four main questions:

1. Can we model or forecast **aggregate sector fund flow** related to KSE-30-tracking funds?
2. Can we estimate **market volatility** for the KSE-30 index?
3. Can we predict **index rebalancing outcomes**, such as which stocks are likely to stay in the index and how their weights may change?
4. Does the market show signs of **efficiency or predictability**?

Important clarification:

- The project does **not** directly observe official "KSE-30 fund flow" as a single published market series.
- Instead, it builds a **proxy sector flow series** using NAV and AUM data from three KSE-30-related funds:
  - `AKD`
  - `NBP`
  - `NTI`
- In the newer pipelines, these are usually combined into one **aggregate sector flow** series.

That distinction is important for both the report and the presentation.

---

## Executive Summary

The repository contains multiple generations of the project. The most useful presentation narrative is:

1. **Collect and clean KSE-30, fund, and macro data**
2. **Build daily and monthly master datasets**
3. **Run the integrated analysis pipeline**
4. **Produce figures, result tables, and rebalancing forecasts**
5. **Validate the rebalancing approach on a held-out March 2026 cycle**

For presentation purposes, the most important folders are:

- `1a_data_cleaning/`
- `1b_eda/`
- `5_claude_pipeline/`
- `6_cursor_model/`
- `8_last_model/`
- `9_march15_2026_backtest/`
- `report_workspace/` and `report_workspace_2/`
- `overview/figures/`

---

## Recommended Story To Present

If someone asks "Which version should we focus on?", use this answer:

- `6_cursor_model/` is the **main integrated pipeline** for the project story.
- `8_last_model/` is the **final focused rebalancing model**.
- `9_march15_2026_backtest/` is the **held-out validation step** used to check how the rebalancing model performs on unseen data.

Older folders such as `4_claude_model/`, `4a_claude_model_merged/`, `5_claude_pipeline/`, and `7_codex_model/` are still useful because they show the project evolution, but they should not all be presented as separate final systems.

---

## Repository Structure

## `0a_data_extraction/`
Purpose: raw data acquisition scripts.

- Used for initial extraction of PSX-related source data.
- This is the starting point of the project pipeline.

## `0b-raw-data/`
Purpose: raw source files and archived inputs.

- Contains raw CSVs and zip files.
- Useful if someone asks where the unprocessed market data came from.

## `0-docs/`
Purpose: research and proposal material.

- Proposal documents
- PSX research papers
- Supporting academic references

## `1a_data_cleaning/`
Purpose: early cleaning and preparation of KSE-30 stock-level data.

Main items:

- `kse30_daily_data.csv`
- `create_kse30_basic.py`
- initial cleaned/basic KSE-30 files

## `1b_eda/`
Purpose: exploratory data analysis and descriptive visual output.

Main items:

- `eda_kse30.py`
- `eda_master_processed.py`
- local copies of data files used for EDA
- generated figures and summary CSVs in `output/`, `output_0/`, `output_1/`, and `output_2/`

This folder is useful for:

- showing data coverage
- checking missingness
- reviewing distributions and correlations
- explaining stationarity checks visually

## `4_claude_model/`
Purpose: earlier notebook-style modeling stage.

Main scripts:

- `nb0_preprocessing.py`
- `nb1_eda.py`
- `nb2_fund_flow_prediction.py`
- `nb3_garch_volatility.py`
- `nb4_portfolio_optimisation.py`
- `nb4b_rebalancing_prediction.py`
- `nb5_market_efficiency.py`
- `nb6_results_summary.py`
- `nb7_kse30_fund_flow_prediction.py`

This folder is important historically, but it is **not** the cleanest final presentation entry point.

## `4a_claude_model_merged/`
Purpose: merged intermediate version of the earlier Claude work.

- Contains `merged_pipeline.py`
- Produces intermediate output tables and figures

## `5_claude_pipeline/`
Purpose: first unified single-script pipeline.

Main file:

- `pipeline.py`

Main role:

- merges stock, fund, and macro data
- creates `daily_master.csv` and `monthly_master.csv`
- generates `results_fund_flow.csv`, `results_garch.csv`, `results_efficiency.csv`, `results_rebalancing.csv`, and `results_rebalancing_forecast.csv`
- generates figures in `figures/`

This folder is still important because later pipelines reuse or build on the same data structure.

## `6_cursor_model/`
Purpose: main integrated pipeline for the presentation.

Main files:

- `pipeline.py`
- `run_pipeline.py`

Key points:

- `run_pipeline.py` is only a thin wrapper that executes `pipeline.py`
- the actual logic is in `pipeline.py`
- this pipeline reconstructs an index-level KSE-30 series from constituent data
- it models **aggregate sector flow** across AKD, NBP, and NTI
- it performs volatility modeling, efficiency testing, and rebalancing prediction in one place
- it also stores figures and result CSVs in a clean structure

This is the best place to explain the full workflow end to end.

## `7_codex_model/`
Purpose: refinement branch of the integrated pipeline.

- Similar structure to `6_cursor_model/`
- Useful as an alternative experiment branch
- Not necessary as the main presentation focus

## `8_last_model/`
Purpose: final specialized rebalancing model.

Main file:

- `kse30_rebalance_pipeline.py`

Main role:

- predicts whether a stock will **stay** in KSE-30 at the next rebalance
- predicts expected **weight change** at the next rebalance
- uses fund features from AKD, NBP, and NTI as mandatory inputs

Outputs:

- `output/tables/kse30_rebalance_training_panel.csv`
- `output/tables/kse30_rebalance_test_predictions.csv`
- `output/tables/kse30_next_rebalance_forecast.csv`
- `output/metrics/kse30_rebalance_metrics.json`
- `output/figures/...`

This is the folder to highlight if the panel asks about the project's most practical forecasting application.

## `9_march15_2026_backtest/`
Purpose: held-out validation of the rebalancing approach.

Main file:

- `pipeline.py`

Main role:

- trains using data up to `2025-12-31`
- evaluates a proxy for the March 2026 rebalance cycle
- writes actual-vs-predicted results for that unseen cycle

Main outputs:

- `march_2026_rebalance_actual_vs_pred.csv`
- `march_2026_metrics.json`
- validation figures in `output/figures/`

This folder is important because it demonstrates that the model was not only fit on historical data but also tested on a later cycle.

## `report_workspace/` and `report_workspace_2/`
Purpose: report-writing and chapter-level figure organization.

- Contains chapter text files, graphs, graph explanations, and generated report assets
- Useful when preparing slides or defending the written methodology

## `overview/`
Purpose: presentation support.

- this `README.md`
- presentation-ready copied figures in `overview/figures/`

---

## Actual End-to-End Workflow

The cleanest way to understand the project is as a pipeline:

### Step 1: Data collection

Inputs come from:

- PSX/KSE-30 constituent daily data
- fund NAV and AUM data for `AKD`, `NBP`, and `NTI`
- macroeconomic variables such as:
  - oil price
  - interest rate
  - USD/PKR
  - CPI
- some branches also use:
  - gold
  - GDP

### Step 2: Cleaning and feature engineering

The pipelines standardize columns, remove bad rows, repair gaps, and construct features such as:

- stock log returns
- rolling volatility
- moving averages
- free-float market-cap proxies
- fund flow estimates from NAV/AUM
- lagged macro variables
- lagged fund variables

### Step 3: Master dataset construction

The integrated pipelines create:

- `daily_master.csv`
- `monthly_master.csv`
- `kse30_stocks_clean.csv`

These files are the foundation for the later models.

### Step 4: Exploratory data analysis

EDA is used to understand:

- whether the data is complete enough
- whether distributions are skewed or heavy-tailed
- whether volatility clusters
- whether correlations are strong or weak
- whether transformations such as returns are needed

### Step 5: Modeling

The project has four major modeling tasks:

1. **Aggregate fund-flow prediction**
2. **Volatility modeling**
3. **Market efficiency testing**
4. **Rebalancing prediction**

### Step 6: Result generation

The main outputs are:

- result CSVs
- diagnostic metrics
- chapter/report figures
- next-rebalance forecasts

### Step 7: Validation

The later-stage work validates the rebalancing logic on a held-out cycle in March 2026.

---

## Main Models Used

The project evolved over time, but the clearest final model summary comes from `6_cursor_model/pipeline.py` plus `8_last_model/kse30_rebalance_pipeline.py`.

## 1. Aggregate Fund-Flow Forecasting

Purpose: forecast the combined sector flow of KSE-30-related funds.

Models used:

- `Naive baseline`
- `ARIMAX`
- `VAR(1)`

Why this matters:

- helps show whether macro and lagged flow information improve over a simple baseline
- provides evidence that the fund-flow proxy has some predictive structure

Important presentation note:

- In the newer pipelines, this is **aggregate sector flow**, not separate final modeling for each fund.

## 2. Volatility Modeling

Purpose: estimate time-varying volatility of KSE-30 returns.

Models/tests used:

- `GARCH(1,1)`
- `EGARCH(1,1)` in the integrated pipeline comparison
- VaR-style backtesting outputs in the figure set

Why this matters:

- volatility clustering is common in financial time series
- volatility forecasts are useful for risk discussion

## 3. Market Efficiency Diagnostics

Purpose: test whether KSE-30 returns behave like a random walk or show dependence.

Tests used:

- `Runs test`
- `Variance Ratio`
- `Ljung-Box Q`
- `Hurst exponent`
- `Granger causality` is also used in the broader project for predictive diagnostics

Why this matters:

- if the market were perfectly efficient, forecasting would be much harder to justify
- this section supports the motivation for modeling

## 4. Rebalancing Prediction

Purpose: predict future KSE-30 membership stability and weight changes.

Regression models:

- `Naive weight baseline`
- `Ridge Regression`
- `RandomForestRegressor`

Classification models:

- `Naive inclusion baseline`
- `LogisticRegression`
- `RandomForestClassifier`

Why this matters:

- this is the most practical and presentation-friendly part of the project
- it directly answers which stocks may stay, leave, gain weight, or lose weight

## 5. Portfolio Optimization

Portfolio optimization exists mainly in the older Claude-stage work:

- `4_claude_model/nb4_portfolio_optimisation.py`

This is part of the project evolution, but it is not the strongest central piece of the final integrated workflow. Mention it as a supporting component, not as the main final deliverable.

---

## Important Data Files

## Market data

- `1a_data_cleaning/kse30_daily_data.csv`
- `5_claude_pipeline/kse30_daily_data.csv`
- `8_last_model/data/kse30_daily_data.csv`

These represent KSE-30 constituent-level market data used across stages.

## Fund data

- `5_claude_pipeline/funds_data.xlsx`
- `7_codex_model/data/funds_data.xlsx`
- `8_last_model/data/funds_data.xlsx`
- `1b_eda/funds_data.xlsx`

Sheets used:

- `AKD`
- `NBP`
- `NTI`

Core columns:

- `DATE`
- `NAV`
- `AUM`

## Macro data

- `5_claude_pipeline/macro_data.xlsx`
- `7_codex_model/data/macro_data.xlsx`
- `8_last_model/data/macro_data.xlsx`
- `1b_eda/macro_data.xlsx`

Typical sheets:

- `OIL`
- `IR`
- `USD`

## Inflation and other optional data

- `5_claude_pipeline/cpi.csv`
- `6_cursor_model/gold.csv`
- `6_cursor_model/gdp.xls`
- `8_last_model/data/inflation.xlsx`

---

## Most Important Outputs

If you need to quickly show the project deliverables, point to these:

## From `6_cursor_model/`

- `daily_master.csv`
- `monthly_master.csv`
- `kse30_stocks_clean.csv`
- `results_fund_flow.csv`
- `results_garch.csv`
- `results_efficiency.csv`
- `results_rebalancing.csv`
- `results_rebalancing_weight_cv.csv`
- `results_rebalancing_forecast.csv`
- `figures/`

## From `8_last_model/`

- `output/tables/kse30_rebalance_training_panel.csv`
- `output/tables/kse30_rebalance_test_predictions.csv`
- `output/tables/kse30_next_rebalance_forecast.csv`
- `output/metrics/kse30_rebalance_metrics.json`
- `output/figures/`

## From `9_march15_2026_backtest/`

- `output/march_2026_rebalance_actual_vs_pred.csv`
- `output/march_2026_metrics.json`
- `output/figures/`

## Presentation-ready copied figures

- `overview/figures/eda/`
- `overview/figures/fund_flow/`
- `overview/figures/garch/`
- `overview/figures/efficiency/`
- `overview/figures/rebalancing/`
- `overview/figures/summary/`

---

## Best Figures To Show In The Presentation

If time is short, prioritize these figures:

## EDA

- `overview/figures/eda/E01_aum_trend.png`
- `overview/figures/eda/E03_fund_flows.png`
- `overview/figures/eda/E04_macro_overview.png`
- `overview/figures/eda/E06_index_cumulative_return.png`
- `overview/figures/eda/E07_top_weights.png`

Use these to explain the data and why the problem matters.

## Fund flow

- `overview/figures/fund_flow/FF01_total_flow_predictions.png`
- `overview/figures/fund_flow/FF02_granger.png`

Use these to explain the forecasting task and predictive relationships.

## Volatility

- `overview/figures/garch/G01_returns_and_vol.png`
- `overview/figures/garch/G02_var_backtest.png`

Use these to explain risk and volatility clustering.

## Market efficiency

- `overview/figures/efficiency/EF01_acf.png`
- `overview/figures/efficiency/EF02_variance_ratio.png`

Use these to justify why forecasting may be possible.

## Rebalancing

- `overview/figures/rebalancing/R01_retention_probability.png`
- `overview/figures/rebalancing/R02_feature_importances.png`
- `overview/figures/rebalancing/R03_weight_scatter.png`
- `overview/figures/rebalancing/R04_weight_changes.png`

These are the strongest practical results in the repo.

## Summary

- `overview/figures/summary/SUMMARY_dashboard.png`

Use this as the final wrap-up slide or before conclusion.

---

## How To Run The Main Parts

## Integrated pipeline

From repo root:

```powershell
python 6_cursor_model/pipeline.py
```

Equivalent wrapper:

```powershell
python 6_cursor_model/run_pipeline.py
```

## Final specialized rebalancing model

From `8_last_model/`:

```powershell
python kse30_rebalance_pipeline.py
```

## March 2026 held-out backtest

From repo root:

```powershell
python 9_march15_2026_backtest/pipeline.py
```

---

## Tech Stack

The repository uses Python-based data science tooling. Based on the project files and `requirements.txt`, the main stack is:

- `Python`
- `pandas`
- `numpy`
- `scipy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `openpyxl`

The repo also lists optional or experimental packages in some branches, including:

- `statsmodels`
- `arch`
- `tensorflow`
- `torch`
- `xgboost`
- `lightgbm`
- `shap`
- `cvxpy`
- `PyPortfolioOpt`

For presentation, do not claim that every listed package is central to the final pipeline. Some are optional, experimental, or used in earlier branches.

---

## What Makes This Project Strong

## Strengths

- uses multiple data types: market, fund, and macro
- does not stop at EDA and includes predictive modeling
- includes both econometric and machine learning methods
- provides a practical rebalancing use case
- includes a held-out validation stage for March 2026
- has a report workspace and figure organization suitable for defense

## Realistic limitations

- the fund-flow target is a **proxy**, not an official single published KSE-30 flow series
- only three related funds are used in the aggregate flow proxy
- some branches are experimental and should not all be presented as equally final
- macro coverage and data history are limited relative to large developed-market studies
- backtesting is narrower than a production-grade institutional system

These limitations are not weaknesses to hide; they are the correct academic framing.

---

## Suggested 5-7 Minute Presentation Flow

1. **Problem statement**
   Explain KSE-30, why index funds and rebalancing matter, and why Pakistan is an interesting emerging-market case.

2. **Data sources**
   Explain PSX constituent data, AKD/NBP/NTI fund data, and macro variables.

3. **Pipeline**
   Show the flow: data collection -> cleaning -> master datasets -> modeling -> outputs -> validation.

4. **Core models**
   Briefly explain:
   - aggregate fund-flow forecasting
   - volatility modeling
   - market efficiency diagnostics
   - rebalancing prediction

5. **Main results**
   Use the rebalancing figures and summary dashboard.

6. **Validation**
   Mention the March 2026 held-out backtest.

7. **Conclusion**
   Emphasize that the project builds a practical framework for KSE-30 analysis and rebalancing support.

---

## Likely Viva Questions And Good Answers

## Q: Are you predicting the KSE-30 index itself or fund behavior?

Best answer:

We analyze the KSE-30 market and also construct an aggregate sector-flow proxy from AKD, NBP, and NTI fund data. In the newer pipeline, the flow prediction task is about that combined sector flow rather than a single official KSE-30 flow series.

## Q: Why use both econometrics and machine learning?

Best answer:

Because they solve different parts of the problem. Econometric models are strong for time-series structure and interpretation, while machine learning models are useful for nonlinear relationships and rebalancing classification/regression tasks.

## Q: Which folder is the final one?

Best answer:

For the full integrated workflow, use `6_cursor_model/`. For the final specialized rebalancing model, use `8_last_model/`. For validation on unseen data, use `9_march15_2026_backtest/`.

## Q: Why do you test market efficiency?

Best answer:

Because if the market were fully random and efficient, forecasting would be much harder to justify. The efficiency diagnostics help motivate the modeling work.

## Q: What is the most practical output of the project?

Best answer:

The rebalancing forecasts: predicted retention probability, exclusion risk, and expected weight changes for KSE-30 constituents.

---

## Final Notes For The Team

- Use `6_cursor_model` as the main code walkthrough.
- Use `overview/figures` for presentation slides.
- Use `8_last_model` and `9_march15_2026_backtest` when discussing practical forecasting and validation.
- Do not overclaim the flow series as official KSE-30 flow; call it an **aggregate proxy from KSE-30-related funds**.
- If the examiners ask about project evolution, explain that the repo preserves earlier model generations for transparency and comparison.

Prepared for project understanding and presentation alignment.
