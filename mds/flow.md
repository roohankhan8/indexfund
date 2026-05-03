# FYP flow — end-to-end project narrative

This document ties together **everything done in this repository**: data origin, cleaning, experiments, multiple modeling tracks, outputs, and what to cite in your report.

## High-level objective

Predict **aggregate net investor flows** into three **KSE-30 index-tracking mutual funds** (AKD, NBP, NTI) using **lagged market and macroeconomic signals**. Use predicted **direction** (inflow vs outflow) to **tilt** portfolio construction (e.g. overweight return leaders on inflows, lean toward liquidity/equal-weight on outflows).

**Fund-flow identity (strip market-driven NAV effect):**

\(flow_t = AUM_t - AUM_{t-1} \times (NAV_t / NAV_{t-1})\)

## Stage map (recommended report structure)

```
0-docs ──► 0-raw-data ──► 1_data_extraction ──► 2_midyear_model
                                                      │
                                                      ▼
                                            3_final_model (enhanced v1–v8)
                                                      │
                           ┌──────────────────────────┴──────────────────────────┐
                           ▼                                                     ▼
                 4_claude_model (modular nb*.py pipeline)           5_claude_pipeline (single pipeline.py)
                                                                           │
                                                                           ▼
                                                           6_cursor_model (KSE-30 index-focused variant)
```

| Stage | Role | Primary outputs |
|-------|------|----------------|
| **0-docs** | Literature + proposal | PDFs/MD summaries for citations |
| **0-raw-data** | Immutable source artifacts | ZIP/XLSX/CSV raw dumps |
| **1_data_extraction** | Normalize PSX dumps → daily panels | notebooks + separation script |
| **2_midyear_model** | Mid-year experiments | plots + engineered snapshot |
| **3_final_model** | Final iterated models + figures | enhanced notebooks/scripts, `output*` PNGs |
| **4_claude_model** | Scriptable chapters (EDA/GARCH/flows/…) | `processed_data/*.csv`, `figures/**/*.png` |
| **5_claude_pipeline** | One-command full run | masters + CSV results + dashboards |
| **6_cursor_model** | KSE-30 index-focused variant | aggregated flows, index-centric GARCH/efficiency (reads from `5_claude_pipeline/`, outputs `results_*.csv`, `figures/`) |

## What was modeled (tracks)

### Track A — `3_final_model/` (ElasticNet/Ridge/boosted trees)

- **Frequency**: Monthly in v5-style work; weekly + technical indicators in **v7** (`scripts/enhanced-v7.py`).
- **Models** (representative suite):
  - **Ridge**, **ElasticNet** (scaled linear, small-N friendly)
  - **Gradient boosting** (regularized shallow trees when used)
  - **XGBoost** / **LightGBM** (when installed; boosted trees)
  - Earlier iterations included **Random Forest** and attempted **LSTM**; LSTM dropped for small-sample reasons (see folder docs + `explanations/v5-explanation.md`).
- **Results (monthly enhanced-v5 interpretation)** — documented in detail in `3_final_model/explanations/v5-explanation.md`:
  - Weak linear correlations between engineered features and target.
  - Holdout \(R^2\) near zero or **negative for tree boosts** ⇒ overfitting on tiny holdouts / rare-event months.
  - Directional metrics on ~10-point holdouts are noisy.
- **Results (weekly v7)** — see `3_final_model/explanations/v7-explanation.md`:
  - More observations (weekly) + technical features; company returns **winsorized** for selection stability.

### Track B — `4_claude_model/` + `5_claude_pipeline/` (econometrics + classical ML)

- **Fund flows**: **ARIMAX(1,0,1)**-style AR with macro exogenous terms (OLS / least-squares walk-forward implementations in module code), plus **VAR(1)** variants; **Granger causality** tests macro → flows.
- **Volatility**: **GARCH(1,1)** and **EGARCH(1,1)** fit via optimization in `pipeline.py`; VaR-style backtests in figures.
- **Efficiency**: **Runs test**, **variance ratio**, **Ljung–Box–style autocorrelation summaries**, **Hurst exponent** approximations.
- **Rebalancing / weights**: **Ridge** regressors + **random forests** (+ **logistic** classifiers where used) on rebalancing-window panels in `pipeline.py` / sister scripts.
- **Results**: Persisted primarily as **`results_*.csv`** alongside interpretive PNGs (`figures/` or folder-level outputs). Treat these CSVs as the quantitative “truth tables” when writing your report.
### Track C — `6_cursor_model/` (Index-focused variant, downstream of Track B)

- **Data source**: Reads clean/preprocessed outputs from `5_claude_pipeline/` (`daily_master.csv`, `monthly_master.csv`, funds data, macro aggregates).
- **Fund flows**: Aggregate **AKD + NBP + NTI** combined flows (single time series); same **ARIMAX(1,0,1)** + **VAR(1)** methodology as Track B, optimized for KSE-30 index-level narrative.
- **Volatility**: **GARCH(1,1)** and **EGARCH(1,1)** on KSE-30 reconstructed index returns; conditional VaR backtests; tail-risk diagnostic PNGs.
- **Efficiency**: Market-level **Runs test**, **Variance Ratio** (multiple lag tiers), **Ljung–Box Q** p-values, rolling **Hurst exponent** (random walk / mean reversion evidence).
- **Rebalancing**: **Ridge** + **Random Forest** weight predictor; forward-looking symbol-level inclusion probabilities; hypothetical future compositions.
- **Results**: Eight CSV tables (`results_garch.csv`, `results_fund_flow.csv`, `results_efficiency.csv`, `results_rebalancing.csv`, etc.); 20+ PNG figures organized in 6 subdirectories (`figures/{eda,garch,fund_flow,efficiency,rebalancing,summary}/`).
- **Integration**: Can run **after** `5_claude_pipeline/` completes; results cited directly in thesis Discussion for KSE-30–specific narratives (efficiency, rebalancing mechanics, volatility regimes).
## Principles you should defend in your thesis

1. **Small \(N\) for monthly macro-regimes** (~47–60 clean months typical) ⇒ prefer **parsimonious** models + time-series CV, not unconstrained forests or deep nets.
2. **Rare shocks** dominate fund flows ⇒ RMSE blows up unless winsorized; still, **prediction of spikes** remains structurally difficult.
3. **Multi-track validation**: econometric baseline (ARIMAX/VAR/GARCH) + ML comparison (ridge/elastic/boosting) strengthens the thesis (not “one notebook says so”).
4. **Direction vs magnitude**: sign of flows is often **more plausible** than point forecasts due to volatility and outliers.

## Where to cite in the repo (quick index)

| Need | Location |
|------|----------|
| Literature | `0-docs/psx-research-papers/` |
| Raw downloads | `0-raw-data/` |
| Evolution story | `2_midyear_model/` notebooks + PNGs |
| Final ML figures | `3_final_model/output*` + `explanations/` |
| Modular writeup | `4_claude_model/figures/` + `processed_data/` |
| One-shot full run | `5_claude_pipeline/pipeline.py` outputs |
| Index-focused variant | `6_cursor_model/pipeline.py` outputs + figures |
| Agent running notes | `AGENTS.md` |

## Suggested “Methods” paragraph (copy/adapt)

We construct monthly aggregate fund flows from daily NAV/AUM for three KSE-30 index funds (AKD, NBP, NTI) using a standard flow residualization identity. Macro variables (oil price, USD/PKR, KIBOR policy rate) and index-level diagnostics are aligned on a calendar and lagged 1–3 months prior to forecasting. **Predictive modeling follows three parallel validation tracks**:

1. **Track A** (`3_final_model/`): Regularized regressions (Ridge, ElasticNet) + shallow boosted trees (XGBoost, LightGBM) with engineered features (lag returns, volatility, abnormal volume); monthly data; time-series cross-validation.

2. **Track B** (`4_claude_model/`, `5_claude_pipeline/`): Econometric baselines—pseudo-ARIMAX(1,0,1) and VAR(1) fits via OLS walk-forward; Granger causality diagnostics (macro → flows); GARCH(1,1) and EGARCH(1,1) volatility clustering; market efficiency tests (Runs, Variance Ratio, Ljung–Box, Hurst).

3. **Track C** (`6_cursor_model/`): Index-focused replication of Track B using KSE-30 reconstructed index returns; rebalancing weight prediction (Ridge + Random Forest); forward-looking inclusion probabilities.

Volatility clustering, conditional VaR, and market microstructure are examined with GARCH-type models and efficiency diagnostics. Forecast performance is summarized with RMSE, MAE, \(R^2\), directional accuracy, and feature importances—with explicit caveats for rare-event months, small holdout sizes, and non-stationarity.

## Per-folder detailed indexes (this repo)

- `mds/0-docs.md`, `mds/0-raw-data.md`, `mds/1_data_extraction.md`, `mds/2_midyear_model.md`, `mds/3_final_model.md`, `mds/4_claude_model.md`, `mds/5_claude_pipeline.md`, `mds/6_cursor_model.md`
