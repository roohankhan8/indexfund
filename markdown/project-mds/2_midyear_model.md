# `2_midyear_model/` — complete file index (exploratory + mid checkpoints)

Purpose: Sandbox notebooks before consolidating into **`3_final_model/`**.

### Notebooks (.ipynb) — conceptual map

| File | Detailed role | Typical models explored | Outputs / takeaway |
|------|----------------|--------------------------|---------------------|
| `theory.ipynb` | Literature-aligned reasoning (econometric vs ML viability, hypotheses). | N/A explanatory | Guided later feature choices |
| `index.ipynb` | Index-level exploratory analytics (ranges, dispersion, correlations). | Descriptive/regression stubs | Validates index aggregates |
| `returns.ipynb` | Experiments translating daily prices → weekly/monthly log returns/vol regimes. | Rolling stats sometimes plus linear probe | Foundations for lag features |
| `symbol.ipynb` | Micro-level firm panels (winner/laggard divergence, dispersion). | Cross-section regressions exploratory | Validates stock-selection module later |
| `copilot-fyp_pipeline.ipynb` | Early stitched pipeline prototyping (EDA + simplistic prediction). | **RandomForestRegressor**/sklearn baselines historically | Produced early plots + sanity checks |
| `claude-fund_flow_predictor.ipynb` | Mid-year predictor focusing on NAV/AUM flow mechanics + regressors. | **Ridge**/tree ensembles exploratory | Produced `graphs/` comparative PNGs |

> Exact hyperparameters drift between saved notebook runs — treat PNGs/logs as artefacts, cite **conceptual lineage** vs final metrics.

### Data snapshot

| File | Role |
|------|------|
| `kse30_daily_data_engineered.xlsx` | Frozen workbook snapshot aligning mid-year experimentation with constituent panel + engineered columns (lags, anomalies). |

### Generated plots (`graphs/`)

| PNG | Visualization |
|-----|---------------|
| `scatter.png`, `corr.png` | Scatter & correlation probes between macro & flow proxies |
| `pred-bahl.png`, `pred-mcb.png` | Per-bank/fund style prediction overlays (historic naming conventions) |
| `backtest-bahl.png`, `backtest-hbl.png`, `backtest-mcb.png` | Simple backtests / directional overlays |

### Generated plots (`output/`)

| PNG | Visualization |
|-----|---------------|
| `pred_vs_actual.png` | Prediction vs realization chart |
| `feature_corr_heatmap.png`, `corr-with-new-features.png` | Feature correlation evolution |
| `reg_feature_importance.png`, `reg_shap_summary.png` | Interpretability dashboards (trees + SHAP) |

### Claude iteration (`claude_output/`)

| PNG | Role |
|-----|------|
| `predictions.png` | Mid-run forecast overlays |
| `feature_importance.png` | Comparative importances |

**Models / quantitative results:** exploratory—**prioritize citations from `3_final_model/explanations/*` and `processed_data/results_*.csv`** for authoritative numbers.
