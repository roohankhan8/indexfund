# Project Team Overview (No-code / How-it-was-built Explanation)

## 1) What this project does (business story)
This project studies **KSE-30** (Pakistan’s top 30 stock index) and aims to produce **actionable, data-driven expectations** about:

1. **Fund flows affecting KSE-30**
   - We cannot observe one official “KSE-30 fund flow” market series.
   - Instead, we build a **proxy “aggregate sector flow”** derived from mutual fund data from **AKD, NBP, and NTI** (funds tracking/linked to KSE-30).

2. **Market volatility / risk**
   - We estimate time-varying volatility (risk) for KSE-30 returns so the system can discuss risk conditions, not just direction.

3. **Index rebalancing outcomes**
   - KSE-30 constituents change at rebalance events.
   - We predict (i) **which stocks are likely to stay** and (ii) **how weights may shift**.

4. **Market efficiency / predictability**
   - We test whether returns behave close to a random walk or contain predictable components.

---

## 2) What “done” means for this FYP
The project is considered complete when:

- Data for KSE-30 constituents + fund and macro variables is cleaned and standardized.
- Master datasets are produced (daily + monthly), enabling consistent modeling.
- Multiple analytical components are computed:
  - fund-flow proxy forecasting results
  - volatility modeling outputs
  - market efficiency test results
  - rebalancing prediction outputs
- The system includes a **held-out validation** stage using **March 2026** as the later-cycle check.

The final deliverable is the rebalancing prediction framework (especially the final specialized pipeline), supported by the forecasting + diagnostics work.

---

## 3) How the project was completed (end-to-end pipeline)

### Phase A — Data collection (inputs)
**Sources used by the project**
- **Market data (KSE-30 constituents)**: daily stock data for the index constituents.
- **Mutual fund data (AKD, NBP, NTI)**: NAV and AUM time series.
- **Macro variables** (selected economic indicators), such as:
  - Oil price
  - Interest rate
  - USD/PKR exchange rate
  - CPI / inflation
- Some branches also used additional series (e.g., gold and GDP).

**Key reasoning**
- Mutual funds tracking/linked to KSE-30 are used to infer “flow-like” behavior that relates to index demand.

---

### Phase B — Data cleaning & feature engineering
The pipelines perform consistent transformations so modeling inputs are reliable:

- Standardize dataset structure (consistent column naming and date alignment).
- Remove or fix problematic rows (e.g., duplicates / gaps).
- Convert raw prices into **returns** and compute rolling/summary metrics.
- Engineer features such as:
  - log returns
  - rolling volatility
  - moving averages
  - free-float market-cap proxies
  - lagged fund-related quantities
  - lagged macro variables

**Important modeling choice**
- Time-series models rely on appropriate transformations (e.g., **log returns** rather than raw price levels) because return series are more suitable statistically.

---

### Phase C — Build master datasets (shared foundation)
The integrated pipelines create datasets used across all models:

- **daily_master.csv**
- **monthly_master.csv**
- **kse30_stocks_clean.csv** (stock-level cleaned panel)

These “master” files are the internal product that ensures every later step uses the same aligned and processed data.

---

### Phase D — Exploratory Data Analysis (EDA)
Before committing to modeling, the project checks whether the data supports the assumptions and what patterns are visible.

EDA focuses on:
- missingness / coverage
- distribution shape (e.g., fat tails)
- volatility clustering behavior
- correlations among fund flow proxies and macro indicators
- stationarity and transformation needs

---

### Phase E — Modeling components (what was actually modeled)
The project’s modeling work is grouped into four major tasks.

#### 1) Aggregate fund-flow prediction (forecasting flows proxy)
Goal: forecast how the aggregate sector flow (built from AKD/NBP/NTI fund data) may change.

Approach used across iterations:
- **Naive baseline** (directional/reference benchmark)
- **ARIMAX / ARIMA-family** modeling (captures time series structure)
- **VAR(1)** style modeling (captures interdependencies between series)

Output is evaluated against naive and summarized into performance tables/figures.

#### 2) Volatility modeling (risk estimation)
Goal: produce a volatility estimate for KSE-30 returns that changes over time.

Approach used:
- **GARCH(1,1)** (main volatility clustering model)
- **EGARCH(1,1)** in comparisons / branching
- Includes risk-oriented validation style outputs (e.g., VaR backtesting visuals)

#### 3) Market efficiency diagnostics (is predictability plausible?)
Goal: validate whether the market is close to random walk behavior or contains structure that supports forecasting.

Diagnostics include:
- Runs test
- Variance Ratio test
- Ljung–Box Q
- Hurst exponent
- Granger-causality-style predictive diagnostics (used in broader workflow)

#### 4) Rebalancing prediction (main practical forecasting application)
Goal: predict rebalancing outcomes.

Two prediction targets are used:
- **Inclusion/retention probability**: which stocks likely stay in KSE-30 at rebalance.
- **Weight change / weight shift**: how weights may increase/decrease.

Model family used in the final story:
- regression + classification variants (e.g., Ridge / Random Forest style approaches)
- includes **cross-validation** to reduce overfitting risk

This rebalancing block is the most presentation-friendly and operationally meaningful output.

---

### Phase F — Result generation (artifacts)
The system produces:
- result CSVs
- evaluation/metrics JSON files
- figures organized into thematic subfolders (EDA, fund flow, volatility, efficiency, rebalancing, summary)

A key part of completion is that outputs are organized so results can be directly shown in the report and slides.

---

### Phase G — Held-out validation (March 2026)
To avoid presenting only “in-sample” performance, a held-out check was performed:

- Train using data up to **2025-12-31**
- Evaluate on a **proxy for the March 2026 rebalance cycle**
- Compare predicted vs actual outcomes
- Record metrics and “actual vs predicted” tables

This step supports the argument that the rebalancing approach can generalize beyond the training window.

---

## 4) Where each part lives in the repository (for navigation)
Your team can use these folders as the “chapter map”:

- `1a_data_cleaning/` — early cleaning for KSE-30 stock data
- `1b_eda/` — exploratory analysis scripts + early figures
- `5_claude_pipeline/` — first unified pipeline producing master datasets + results
- `6_cursor_model/` — **main integrated pipeline** (best for the end-to-end walkthrough)
- `8_last_model/` — **final specialized rebalancing model** (most important for practical results)
- `9_march15_2026_backtest/` — held-out validation for March 2026
- `report_workspace/` and `report_workspace_2/` — report chapter text + figure organization
- `overview/figures/` — presentation-ready figures for team/slides

---

## 5) Most important outputs to reference in your team presentation

### Rebalancing (final practical results)
- Training panel used to learn the rebalancing patterns
- Test predictions for model evaluation
- Next rebalance forecast files
- Metrics summary JSON
- Rebalancing figures (retention probability, feature importance, weight scatter, weight changes)

### Fund flow, volatility, efficiency (supporting evidence)
- Fund flow forecasting performance figures
- Volatility plots and VaR-style backtest visuals
- Market efficiency diagnostics visuals

### Summary dashboard
- A “single slide” style dashboard figure intended as the final wrap-up artifact.

---

## 6) The figures your team can show (what they mean)
The project includes a set of curated figures grouped by topic. In a walkthrough:

- **EDA:** show how funds and macro variables evolve and how returns behave.
- **Fund flow:** show predictive performance and whether flows relate to macro signals.
- **Volatility:** show volatility clustering and risk calibration style outputs.
- **Efficiency:** show whether returns contain predictable components.
- **Rebalancing:** show retention probability, the most important drivers, and predicted weight changes.
- **Summary dashboard:** show end-to-end performance in one place.

---

## 7) Key decisions & constraints (important “how it was framed” points)
- The “fund flow” is treated as a **proxy** built from mutual fund NAV/AUM (AKD, NBP, NTI), not an officially published single series.
- The project contains multiple generations of modeling work; the **final narrative** highlights:
  - integrated pipeline story (`6_cursor_model/`)
  - the final rebalancing implementation (`8_last_model/`)
  - held-out validation (`9_march15_2026_backtest/`)

---

## 8) Completion statement
This project was completed in June 2026 and validated via a later-cycle test (March 2026) to support the final rebalancing prediction framework.

