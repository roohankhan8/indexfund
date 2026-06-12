# `4_claude_model/` — every file grouped + models + outputs

Purpose: Modular **research scripts** mirroring notebooks (`nb*_....py`). Each builds or consumes artefacts in `processed_data/` and emits structured plots under `figures/`.

Macro identity for flows remains

\(flow_t = AUM_t - AUM_{t-1} × (NAV_t / NAV_{t-1})\)

---

## Python modules (numeric engines)

| File | Responsibility | Algorithms / Models | Outputs |
|------|------------------|---------------------|---------|
| `nb0_preprocessing.py` | Ingest constituent XLSX (`data/kse-30-basic.xlsx`), optional **`kse30_index_level.csv`**, funds + macro | Data transforms only (**ADF helper** summaries printed) | `processed_data/daily_master.csv`, `monthly_master.csv`, `kse30_stocks_daily.csv`, `txts/preprocessing_report.txt`, copies `kse30_daily_data.csv` |
| `nb1_eda.py` | Descriptive dashboards (NAV densities, correlation heatmaps) | Descriptive statistics | `figures/eda/A*.png`, `figures/eda/B*.png` (+ console tables) |
| `nb2_fund_flow_prediction.py` | Forecast aggregate + fund-specific flows vs macro suite | Manual **OLS** TSCV analogue; **ARIMAX(1,0,1)** style AR+macro stacking; **VAR(1)** w/ exogenous; **Granger** F-tests implemented via linear algebra; residual diagnostics | `figures/fund_flow/F*.png`, `processed_data/results_fund_flow_prediction.csv` |
| `nb3_garch_volatility.py` | Volatility regimes & VaR overlays | **`scipy.optimize.minimize`** tuned **GARCH(1,1)** & **EGARCH(1,1)** surrogate loss (see script) | `figures/garch/G*.png`, `processed_data/results_garch.csv` |
| `nb4_portfolio_optimisation.py` | Frontier / clustered correlation portfolios | Convex **mean-variance**/frontier scaffolding (SciPy/Sklearn combos); inspect file for covariance shrinkage choices | `figures/portfolio/P*.png`, `processed_data/results_portfolio.csv`, `portfolio_weights.csv` pathway |
| `nb4b_rebalancing_prediction.py` | PSX recombination weight dynamics | Rolling window **Ridge regression** plus **RandomForest**/tree classifiers (file header enumerates splits) plus confusion matrices visualization | extra `figures/rebalancing/R*` beyond core pipeline duplication |
| `nb5_market_efficiency.py` | Classical tests on NAV/idiosyncratic proxies | Runs test statistics, Variance Ratio Monte-style loops, LB Q approximations, Hurst regressions | `figures/efficiency/E*.png`, `processed_data/results_efficiency.csv` |
| `nb6_results_summary.py` | Glues CSV metrics into dashboard plots | Aggregation only | `figures/summary/S*.png` |
| `nb7_kse30_fund_flow_prediction.py` | Extended narrative variant of flow stack (alternate plots / exports) | Same econometric scaffolding as nb2 lineage | supplementary `figures/fund_flow/*.png`, `processed_data/results_fund_flow_prediction.csv` refresh + `txts/nb7.txt` |

---

## Data inputs (`data/`)

| Asset | Meaning |
|-------|---------|
| `kse-30-basic.xlsx` | Daily constituents |
| `kse30_index_level.csv` | Index-level synthesized returns / macro liquidity companion |
| `funds_data.xlsx`, `macro_data.xlsx`, `cpi.csv` | Identical semantics as elsewhere in repo |

---

## Processed tables (`processed_data/`)

| CSV | Produced by | Statistical content |
|-----|---------------|---------------------|
| `daily_master.csv` | nb0 | Merged NAV + macro aligned on trading lattice |
| `monthly_master.csv` | nb0 | Aggregated macros + summed fund flows (`total_fund_flow`) |
| `kse30_stocks_daily.csv`, `kse30_daily_data.csv` | Copies / cleaned constituent exports | Long panels for cross-section modelling |
| `results_descriptive.csv` | nb1/nb6 synergy | Moments & distributional KPIs |
| `results_garch.csv` | nb3 | Parameter tables + criterion metrics |
| `results_fund_flow_prediction.csv` | nb2 | RMSE/R²/dir accuracy rows per modelling branch |
| `results_efficiency.csv` | nb5 | Z-stats, verdict flags per fund/time window |
| `results_portfolio.csv` | nb4 | Optimizer objective values / constraints diagnostic |
| `results_rebalancing.csv`, `results_rebalancing_forecast.csv`, `portfolio_weights.csv` | nb4b / rebal extension | Inclusion probabilities + predicted horizon weights |

> **Treat these CSVs as canonical numeric tables** — mention exact column names inside your analytical chapter.

---

## Figures (`figures/`) complete listing

Below are **every** PNG currently tracked (sorted by subdirectory).

### `figures/eda/`

- `A1_macro_daily.png`, `A2_nav_levels.png`, `A3_nav_return_dist.png`, `A4_rolling_volatility.png`, `A5_macro_vs_nav_scatter.png`
- `A6_daily_correlation_heatmap.png`, `A7_squared_returns_clustering.png`, `A8_interest_rate_vs_nav.png`
- `B1_aum_over_time.png`, `B2_fund_flows_bar.png`, `B3_flow_vs_macro.png`, `B4_monthly_nav_returns.png`
- `B5_monthly_correlation_heatmap.png`, `B6_cross_correlation_flow_macro.png`, `B7_flow_vs_cpi_scatter.png`

### `figures/efficiency/`

- `E1_acf_nav_returns.png`, `E2_variance_ratio.png`, `E3_rolling_autocorrelation.png`
- `E4_flow_vs_acf_scatter.png`, `E5_hurst_exponents.png`, `E6_stock_efficiency_heatmap.png`

### `figures/fund_flow/`

- `F1_total_flow_predictions.png`, `F1_AKD_predictions.png`, `F1_NBP_predictions.png`, `F1_NIT_predictions.png`
- `F2_individual_fund_predictions.png`, `F2_training_loss_curves.png`, `F3_model_comparison.png`, `F3_rmse_comparison.png`
- `F4_actual_vs_predicted_scatter.png`, `F4_granger_causality.png`, `F5_residual_diagnostics.png`, `F6_flow_efficiency_link.png`

### `figures/garch/`

- `G1_returns_and_garch_vol.png`, `G2_egarch_news_impact.png`, `G3_AKD_residual_diagnostics.png`
- `G3_NBP_residual_diagnostics.png`, `G3_NIT_residual_diagnostics.png`
- `G4_var_backtest.png`, `G5_model_selection_aic_bic.png`, `G6_vol_model_comparison_AKD.png`

### `figures/portfolio/`

- `P1_stock_correlation_heatmap.png`, `P2_clustered_correlation.png`, `P3_return_risk_scatter.png`
- `P4_efficient_frontier.png`, `P5_weight_comparison_bar.png`, `P5b_weight_heatmap.png`
- `P6_cumulative_returns.png`, `P7_rolling_sharpe.png`, `P8_return_distributions_violin.png`, `P9_drawdowns.png`

### `figures/rebalancing/`

- `R1_composition_history.png`, `R2_weight_prediction_scatter.png`, `R3_feature_importances.png`
- `R4_retention_probability.png`, `R5_predicted_weight_changes.png`, `R6_confusion_matrices.png`, `R7_feature_distributions.png`

### `figures/summary/`

- `S1_results_dashboard.png`, `S2_portfolio_weights_final.png`, `S3_efficiency_summary_by_fund.png`

---

## TXT logs (`txts/`)

| File | Produced by |
|------|---------------|
| `preprocessing_report.txt` | nb0 consolidated console log |
| `nb1.txt` … `nb5.txt`, `nb4b.txt`, `nb7.txt` | Captured streamed narrative from respective script executions |

---

## Results interpretation tips

| Theme | Guidance |
|-------|----------|
| Fund-flow models | Highlight **econometric anchors** vs pure ML speculative fits |
| GARCH CSV | Narrate volatility persistence clusters + VaR exceedances plotted |
| Efficiency metrics | Discuss mixed evidence depending on NAV window slicing |
| Rebalancing artefacts | Probability bars show **risk of deletion** vs mechanical weight inertia |
