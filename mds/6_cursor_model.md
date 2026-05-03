# `6_cursor_model/` — KSE-30 index-focused variant of pipeline

Purpose: **`pipeline.py`** reads preprocessed data from sibling folder `5_claude_pipeline/` and replicates the same analytical sections but with **aggregated KSE-30 index-level focus**:

1. **Data ingestion** — consume `5_claude_pipeline/` outputs (masters, cleaned stocks)
2. **GARCH volatility** — index-level returns conditioning (GARCH(1,1), EGARCH(1,1))
3. **Aggregate fund flows** — combined AKD+NBP+NTI flows as single time series
4. **Market efficiency** — index-level statistical tests (Runs, Variance Ratio, Ljung–Box, Hurst)
5. **Rebalancing prediction** — index constituent inclusion/weight forecasting
6. **Results synthesis** — CSV artefacts + PNG visuals

**Key design difference from `5_claude_pipeline/`**: Expects clean input CSVs already staged by the upstream pipeline (reduces I/O duplication, improves modularity).

---

## Operational usage

Activate venv (`AGENTS.md`) then, from repo root:

```powershell
python 6_cursor_model/pipeline.py
```

Alternatively (wrapper entry point):

```powershell
python 6_cursor_model/run_pipeline.py
```

Runtime prints numeric tables and progress messages; key metrics also saved to **`txts/pipeline.txt`** for appendix inclusion.

---

## Source code + runnable I/O

| File | Role |
|------|------|
| `pipeline.py` | Main orchestrator (eight sections: ingest → EDA → GARCH → flows → efficiency → rebalancing → summary) |
| `run_pipeline.py` | Thin wrapper; calls `runpy.run_path(pipeline.py)` |
| **Inputs** (from `../5_claude_pipeline/`) | `kse30_daily_data.csv`, `funds_data.xlsx`, `macro_data.xlsx`, `cpi.csv` |
| **Output CSVs** (this dir) | `daily_master.csv`, `monthly_master.csv`, `kse30_stocks_clean.csv`, `results_*` tables |
| **Output PNGs** (this dir) | `figures/**/*.png` — 6 subdirectories (eda, garch, fund_flow, efficiency, rebalancing, summary) |
| **Log** | `txts/pipeline.txt` (captured print output) |

---

## Produced CSV artefacts

| Artifact | Generated in | Holds |
|----------|--------------|-------|
| `daily_master.csv` | Data ingestion section | Calendar-aligned daily constituent, macro, NAV/AUM composites |
| `monthly_master.csv` | Aggregation section | Month-bucketed data + total aggregate `fund_flow` |
| `kse30_stocks_clean.csv` | Data hygiene | Duplicate-free KSE-30 constituent history |
| `results_garch.csv` | §4 Volatility | GARCH(1,1) + EGARCH(1,1) parameter summaries; conditional variance diagnostics |
| `results_fund_flow.csv` | §5 Flows | Benchmark comparisons (Naïve vs ARIMAX(1,0,1) vs VAR(1)); Granger causality F-stats |
| `results_efficiency.csv` | §6 Efficiency | Statistical test metrics (Runs Z, Variance Ratio tiers, Ljung–Box Q p-values, Hurst exponent rolling estimates) |
| `results_rebalancing.csv` | §7 Rebalancing | Ridge + Random Forest weight & inclusion model performance; holdout cross-validation metrics |
| `results_rebalancing_forecast.csv` | §7 Extrapolation | Forward-looking symbol-level inclusion probabilities + hypothetical future KSE-30 weights |

> **Interpretation**: Treat `results_*.csv` rows and numbers as **quantitative ground truth** — cite cell values directly in your Methods/Results sections.

---

## Figures (`figures/`) inventory

### `figures/eda/`

- **E01_aum_trend.png** — Time series of AUM for AKD, NBP, NTI (separate lines) + aggregate
- **E02_nav_return_dist.png** — NAV-based return histograms + kernel density overlays (3 funds)
- **E03_fund_flows.png** — Aggregate monthly flows time series (inflow/outflow bars)
- **E04_macro_overview.png** — Oil (Brent USD/bbl), USD/PKR, KIBOR (IR) stacked or multi-axis
- **E05_monthly_correlation.png** — Heatmap: flows vs lagged macro + index returns
- **E06_index_cumulative_return.png** — KSE-30 reconstructed index cumulative return (vs benchmark if available)
- **E07_top_weights.png** — Time-varying top-10 index weights (stacked area or faceted bars)

### `figures/garch/`

- **G01_returns_and_vol.png** — KSE-30 daily returns + rolling 20-day volatility envelope; GARCH conditional σ overlay
- **G02_var_backtest.png** — Value-at-Risk (95%, 99%) backtesting results; empirical vs predicted tail losses

### `figures/fund_flow/`

- **FF01_total_flow_predictions.png** — Observed vs predicted flows (Naïve benchmark, ARIMAX, VAR); 1-step ahead + rolling forecast
- **FF02_granger.png** — Granger causality bar chart: F-stats for macro → flows direction; p-value thresholds

### `figures/efficiency/`

- **EF01_acf.png** — Autocorrelation function plots (ACF + PACF) for daily returns; lag-based inspection for mean reversion
- **EF02_variance_ratio.png** — Variance Ratio test results across multiple lag horizons (1:2, 1:4, 1:8, etc.); stepping vs random walk hypothesis

### `figures/rebalancing/`

- **R01_retention_probability.png** — Predicted inclusion probability by stock; ROC or calibration curve
- **R02_feature_importances.png** — Random Forest / Ridge feature weights (lag returns, volatility, volume, macro)
- **R03_weight_scatter.png** — Observed vs predicted KSE-30 weights (scatter + regression line)
- **R04_weight_changes.png** — Residual histograms + Q–Q plots for rebalancing prediction errors

### `figures/summary/`

- **SUMMARY_dashboard.png** — Multi-panel collage: AUM trends, flows, volatility, efficiency, rebalancing metrics in compact layout

---

## Models embedded (explicit)

| Tier | Algorithms | Section | Notes |
|------|------------|---------|-------|
| **Volatility** | GARCH(1,1), EGARCH(1,1) | §4 | `scipy.optimize.minimize` NLL; custom loss, no statsmodels |
| **Fund flows** | Pseudo-ARIMAX(1,0,1), VAR(1), Granger F-tests | §5 | Least-squares solve partitioned; bespoke implementation |
| **Efficiency** | Runs test Z, Variance Ratio, Ljung–Box Q, Hurst exponent | §6 | Rolling/windows; custom permutation tests where needed |
| **Rebalancing** | Ridge, RandomForest (regressor + classifier), Logistic | §7 | sklearn; weight + inclusion prediction; forward forecast |

All algorithms use **custom OLS time-series cross-validation** (no statsmodels); document clearly in dissertation methods.

---

## Result reading & interpretation guide

| Question | Where to look | What to extract |
|----------|--------------|-----------------|
| Do macro indicators causally precede aggregate flows? | `FF02_granger.png`, `results_fund_flow.csv` (Granger rows) | F-stats, p-values; lag structure |
| Is KSE-30 volatility persistent or mean-reverting? | `G01_returns_and_vol.png`, `results_garch.csv` | GARCH α + β (persistence = α+β); σ_t scaling |
| Which features best predict index composition changes? | `R02_feature_importances.png`, `results_rebalancing.csv` | Feature names ranked by Gini/MSE reduction |
| Can we forecast weight rotations months ahead? | `R04_weight_changes.png`, `results_rebalancing_forecast.csv` | MAE / RMSE; symbol-level prob distributions |
| Does KSE-30 exhibit statistical inefficiency? | `EF01_acf.png`, `EF02_variance_ratio.png`, `results_efficiency.csv` | ACF decay, VR(k) vs 1.0, runs count significance |

---

## Data flow summary (this module)

```
5_claude_pipeline/
  ├── kse30_daily_data.csv    ──┐
  ├── funds_data.xlsx         ──┼─► 6_cursor_model/pipeline.py
  ├── macro_data.xlsx         ──┤   (read from ../5_claude_pipeline/)
  └── cpi.csv                 ──┘
                                      │
                                      ▼
                         ┌─────── Sections 0–8 ─────────┐
                         │                               │
                    ┌────▼─────┐               ┌────────▼────┐
                    │ CSVs     │               │ PNGs        │
                    ├──────────┤               ├─────────────┤
                    │ Masters  │               │ 6 subfolders│
                    │ results_*│               │ (eda, ...) │
                    └──────────┘               └─────────────┘
                         │                            │
                         └────► 6_cursor_model/ ◄─────┘
                                (outputs next to code)
```

---

## Integration with broader pipeline

- **Upstream**: Consumes cleaned/merged outputs from `5_claude_pipeline/` (reduces code repetition, improves stability).
- **Downstream**: Results feed into report/dissertation narratives; key metrics cite `results_*.csv` directly.
- **Parallel track**: Can run **concurrently** with `4_claude_model/` (modular track) or **sequentially** after `5_claude_pipeline/` completes.

---

## Operational notes

1. **Virtual environment**: Must activate `.venv` before running (see `AGENTS.md`).
2. **Dependencies**: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `openpyxl`.
3. **Paths**: Script auto-detects `_SCRIPT_DIR` and `_REPO_ROOT` using `__file__` (portable).
4. **Output locations**: All CSVs and figures written **next to `pipeline.py`** (`6_cursor_model/`).
5. **Reproducibility**: `np.random.seed(42)` set at top of script; results deterministic given same input data.

---

## Typical workflow

1. Run `5_claude_pipeline/pipeline.py` (generates masters, base results).
2. Run `6_cursor_model/pipeline.py` (consumes upstream outputs, generates index-focused results).
3. Cross-reference `results_*.csv` and PNG figures in thesis Discussion section.
4. For sensitivity analysis, edit Section 0 config (e.g. lag windows, model hyperparams).
