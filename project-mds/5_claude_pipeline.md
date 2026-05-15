# `5_claude_pipeline/` — standalone end-to-end `pipeline.py` inventory

Purpose: **`pipeline.py`** replicates sections 1–9 of modular research lane *without* juggling multiple entry points — load → cleanse → visualize → quantify across:

1. ingestion & merges,
2. master tables,
3. EDA visuals,
4. GARCH stacks,
5. **ARIMAX(1,0,1)** + **VAR(1)** fund-flow regressions (+ Granger),
6. market efficiency diagnostics,
7. rebalancing predictor (Ridge regression + sklearn tree classifiers/regressors ensemble),
8. summary dashboard collage.

Outputs default **next to the script** (`BASE = dirname(__file__)`).

Identity again:

\(flow_t = AUM_t - AUM_{t-1} × (NAV_t / NAV_{t-1})\).

---

## Source code + runnable IO

| File | Role |
|------|------|
| `pipeline.py` | Sole orchestrator |
| `kse30_daily_data.csv` | Constituents daily feed |
| `funds_data.xlsx`, `macro_data.xlsx`, `cpi.csv` | Nav/AUM drivers + Brent + FX + POLICY rate + CPI |
| `figures/**` PNGs see below |
| CSV outputs | `daily_master.csv`, `monthly_master.csv`, cleaned stock file `kse30_stocks_clean.csv`, results tables prefixed `results_*` |

---

## Produced CSV artefacts

| Artifact | Produced in section | Holds |
|----------|---------------------|-------|
| `daily_master.csv` | §2 | Calendar-aligned macro+NAV composites |
| `monthly_master.csv` | §2 | Month buckets + summed flows (`total_fund_flow`) |
| `kse30_stocks_clean.csv` | §1–2 hygiene | Duplicate-free constituent history post cleaning |
| `results_garch.csv` | §4 | GARCH(1,1) parameter summaries per fund NAV |
| `results_fund_flow.csv` | §5 | Rows for Naïve vs ARIMAX vs VAR benchmarks (+ per fund ARIMAX blocks) |
| `results_efficiency.csv` | §6 | Statistical test metrics (Runs, Variance Ratio tiers, LB p-values, Hurst) |
| `results_rebalancing.csv` | §7 | Ridge / RF weight & inclusion benchmarks |
| `results_rebalancing_forecast.csv` | §7 extrapolation forward | Symbol-level probabilities + hypothetical future weights |

> **Interpret “results_*” CSVs verbatim in your Discussion section** rather than rewriting numbers from PNG axes.

---

## Figures (`figures/`) exhaustive list

### `figures/eda/`

- `E01_aum_trend.png`
- `E02_nav_return_dist.png`
- `E03_fund_flows.png`
- `E04_macro_overview.png`
- `E05_monthly_correlation.png`
- `E06_index_cumulative_return.png`
- `E07_top_weights.png`

### `figures/garch/`

- `G01_returns_and_vol.png`
- `G02_var_backtest.png`

### `figures/fund_flow/`

- `FF01_total_flow_predictions.png`
- `FF02_granger.png`

### `figures/efficiency/`

- `EF01_acf.png`
- `EF02_variance_ratio.png`

### `figures/rebalancing/`

- `R01_retention_probability.png`
- `R02_feature_importances.png`
- `R03_weight_scatter.png`
- `R04_weight_changes.png`

### `figures/summary/`

- `SUMMARY_dashboard.png` (facet collage of KPI panels)

Supporting text log:

| File | Notes |
|------|-------|
| `txts/pipeline.txt` | Captured textual trace of numeric tables printed |

---

## Models embedded (explicit)

| Tier | Algorithms | Section |
|------|------------|---------|
| Volatility | **GARCH(1,1)** & **EGARCH(1,1)** via **`scipy.optimize.minimize`** NLL surrogates | §4 |
| Flow regression | Pseudo-**ARIMAX(1,0,1)** solved by partitioned **least squares**, **VAR(1)** stack; **ADF** heuristic; bespoke **Granger** F-stats | §5 |
| Efficiency | Runs test **Z**, **Variance Ratio**, **Ljung–Box Q** analogue, Rolling **Hurst** regression | §6 |
| Rebalancing supervised | **`sklearn.linear_model.Ridge`**, **`RandomForestRegressor`**, **`RandomForestClassifier`**, **`LogisticRegression`** (see Section 7 printouts) | §7 |

Because many blocks are coded without `statsmodels`, label them *custom OLS TSCV equivalents* transparently inside your dissertation.

---

## Result reading guide

| Question | Inspect |
|-----------|---------|
| Do macro indicators Granger-precede aggregate flows? | `FF02_granger.png`, `results_fund_flow.csv` rows with Granger-derived tags |
| How stable is conditional volatility? | `G01`, `results_garch.csv` persistence numbers |
| Can strategies front-run recompositions probabilistically? | `R01`-`R04` series + CSV forecasts |
| Do efficiency tests contradict semi-strong EMH? Mixture signals? | `EF01`, `EF02`, `results_efficiency.csv` |

---

## Operational note

Activate venv (`AGENTS.md`) then:

```powershell
python 5_claude_pipeline/pipeline.py
```

Runtime prints echo **descriptive statistics tables** echoed also into **`txts/pipeline.txt`** for inclusion in appendix.
