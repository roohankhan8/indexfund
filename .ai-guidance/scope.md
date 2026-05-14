# Project Scope: What Has Actually Been Completed

## 1. Problem and Goal
This project studies KSE-30 index-fund behavior in Pakistan by combining:
- mutual/index fund flow dynamics,
- macro and market signals,
- market efficiency diagnostics,
- and rebalancing/inclusion prediction.

Primary goal implemented in code:
- Predict aggregate net flows for KSE-30 index-tracking funds (AKD, NBP, NTI) and use that signal to support index-weight/rebalancing decisions.

Core flow identity used across pipelines:

`flow_t = AUM_t - AUM_(t-1) * (NAV_t / NAV_(t-1))`

## 2. Data Scope Implemented
### 2.1 Covered Data
- PSX KSE-30 constituent panel (daily prices, weights, free-float shares/mcap, volume)
- Fund NAV/AUM for AKD, NBP, NTI
- Macroeconomic series:
  - Brent oil
  - USD/PKR
  - Interest rate
  - CPI YoY

### 2.2 Date Window in Practice
- Main stock panel in current runs spans roughly `2020-01` to `2026-04`.
- Modeling windows are filtered/lagged by pipeline sections (daily and monthly master tables).

### 2.3 Data Engineering Implemented
- Column harmonization and cleaning
- Trading-halt row removal for known anomalous dates
- Per-symbol features:
  - log returns
  - rolling vol (30d)
  - moving averages (20, 50)
- Monthly flow construction from NAV/AUM
- Macro alignment to daily/monthly calendars
- CPI month parsing and joining

## 3. Modeling Scope Implemented
The work is not one model; it is a multi-track framework.

### 3.1 Track A: Final-model ML lane (`3_final_model`)
- Ridge
- ElasticNet
- Gradient boosting variants
- XGBoost (where available)
- LightGBM (where available)
- Early/legacy experiments include Random Forest and LSTM attempts (LSTM not retained as core final approach due sample/robustness concerns).

### 3.2 Track B: Econometric + diagnostics lane (`4_claude_model`, `5_claude_pipeline`)
- Flow forecasting:
  - ARIMAX(1,0,1)-style implementation
  - VAR(1)
  - Granger causality tests
- Volatility:
  - GARCH(1,1)
  - EGARCH(1,1)
- Efficiency diagnostics:
  - Runs test
  - Variance Ratio
  - Ljung-Box style autocorrelation checks
  - Hurst exponent
- Rebalancing:
  - Ridge regression
  - RandomForestRegressor
  - LogisticRegression
  - RandomForestClassifier

### 3.3 Track C: KSE-30 index-focused variant (`6_cursor_model`)
- Same core modeling family as Track B with stronger index-level framing
- Aggregate sector flow as single target series
- Rebalancing inclusion/weight forecasting and forward composition outputs

### 3.4 Additional Executed Variant (`7/`)
- `6_cursor_model` pipeline was executed on a new staged dataset in folder `7/` (local `7/data` inputs, outputs written to `7/`).

## 4. Statistical Testing Scope Implemented
- Stationarity testing performed in project workflow:
  - ADF
  - PP (Phillips-Perron)
  - KPSS
- Practical transformation workflow implemented:
  - level series -> log differences/returns/deltas
  - flow ratios (`flow_pct`) for stabilization
- Before/after stationarity tables and transformed datasets exported for report usage.

## 5. EDA and Reporting Scope Implemented
### 5.1 EDA Coverage
- KSE-30 panel EDA:
  - constituent coverage
  - price/weight distributions
  - aggregate mcap/volume/weight diagnostics
  - return distribution/correlation/volatility visuals
  - liquidity-vs-weight and missingness checks
- Funds EDA:
  - NAV/AUM levels
  - daily return distributions
  - rolling volatility
  - monthly flows and flow correlations
- Macro + CPI EDA:
  - level charts
  - macro return charts
  - monthly macro/CPI correlation panel

### 5.2 Report Workspace Outputs
Dedicated report-ready outputs were generated under:
- `report_workspace/chapter-03-methodology/images`

Including:
- chapter EDA figure sets,
- stationarity tables and graphics,
- transformed modeling CSVs for thesis-ready reference.

## 6. Deliverables Produced in Repo
### 6.1 Core Tables/CSVs
- `daily_master.csv`
- `monthly_master.csv`
- `kse30_stocks_clean.csv`
- `results_garch.csv`
- `results_fund_flow.csv`
- `results_efficiency.csv`
- `results_rebalancing.csv`
- `results_rebalancing_forecast.csv`

### 6.2 Core Figure Families
- `figures/eda`
- `figures/garch`
- `figures/fund_flow`
- `figures/efficiency`
- `figures/rebalancing`
- `figures/summary`

These exist in main pipelines and index-focused variants (`5_claude_pipeline`, `6_cursor_model`, `7` run folder).

## 7. In-Scope Conclusions Supported by Code
- Aggregate index-fund flow prediction is feasible but sensitive to small monthly sample sizes and outlier months.
- Level series are often non-stationary; transformed return/difference/rate series are more suitable for modeling.
- Volatility clustering and tail risk behavior are observable in KSE-30 reconstructed series through GARCH-family diagnostics.
- Rebalancing prediction can produce practical forward ranking/probability outputs for inclusion and weight shifts.

## 8. Explicit Boundaries (Out of Scope)
- No live trading system, broker integration, or execution engine
- No causal claims beyond implemented statistical diagnostics
- No guarantee of production-grade strategy profitability under real transaction costs/slippage
- No full official KSE index-rule replication engine
- No external real-time data APIs in the delivered pipeline

## 9. Known Data/Method Caveats Captured in Work
- Volume gaps/missing periods exist in source panel and affect some aggregates.
- Some tests/models are custom implementations (not always statsmodels-native exact forms).
- Monthly sample size constraints limit confidence in complex/nonlinear models.
- Directional signal interpretation is generally more stable than exact magnitude forecasting.

## 10. Operational Scope (How It Is Run)
- Canonical full pipeline:
  - `python 5_claude_pipeline/pipeline.py`
- Index-focused pipeline:
  - `python 6_cursor_model/pipeline.py`
- New-dataset staged run:
  - `python 7/pipeline.py`
- Focused EDA:
  - `python 1b/eda_kse30.py`

This scope document reflects completed implementation state in the repository, not just proposal intent.
