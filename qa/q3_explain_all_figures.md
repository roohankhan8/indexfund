# Q3: Explain each graph in `6_cursor_model/figures` and the concepts used

## EDA (`figures/eda`)
1. **E01_aum_trend.png**
- **What it shows:** Monthly combined AUM (PKR mn) of KSE-30 index-tracking funds.
- **Concept:** Asset growth/decline over time, investor participation trend, structural expansion vs contraction.

2. **E02_nav_return_dist.png**
- **What it shows:** Histogram of daily KSE-30 log returns with an overlaid normal density curve.
- **Concept:** Return distribution shape (mean, volatility, skewness), normality comparison, fat-tail/asymmetry intuition.

3. **E03_fund_flows.png**
- **What it shows:** Monthly aggregate net flows (positive/negative bars) with spike markers (>2s).
- **Concept:** Capital inflow/outflow cycles, liquidity pressure periods, outlier flow detection via standard-deviation rule.

4. **E04_macro_overview.png**
- **What it shows:** Multi-panel time series of oil, USD/PKR, policy rate, gold (if available), GDP proxy.
- **Concept:** Macro regime tracking and exogenous drivers likely linked to index returns and fund flows.

5. **E05_monthly_correlation.png**
- **What it shows:** Correlation heatmap among macro returns/rates, KSE-30 return/volatility, sector flow %.
- **Concept:** Linear co-movement and sign/magnitude of pairwise relationships (exploratory, not causality).

6. **E06_index_cumulative_return.png**
- **What it shows:** Reconstructed KSE-30 cumulative log return path with positive/negative shading.
- **Concept:** Wealth-path style performance view, drawup/drawdown regime perspective.

7. **E07_top_weights.png**
- **What it shows:** Top 15 stocks by average KSE-30 index weight (2021 onward).
- **Concept:** Index concentration and constituent dominance (which names drive index behavior more heavily).

## GARCH (`figures/garch`)
8. **G01_returns_and_vol.png**
- **What it shows:** Daily return series and model-implied conditional volatility from best GARCH specification.
- **Concept:** Volatility clustering and time-varying risk (heteroskedasticity).

9. **G02_var_backtest.png**
- **What it shows:** Returns vs model 5% VaR line with exception points.
- **Concept:** Value-at-Risk calibration/backtesting; exception frequency tests tail-risk adequacy.

## Fund Flow (`figures/fund_flow`)
10. **FF01_total_flow_predictions.png**
- **What it shows:** Train/test actual transformed sector flow with ARIMAX and VAR(1) fit/forecast lines.
- **Concept:** Out-of-sample forecasting comparison, dynamic model fit vs baseline behavior.

11. **FF02_granger.png**
- **What it shows:** Bar chart of lag-1 Granger p-values for macro variables predicting aggregate sector flow.
- **Concept:** Predictive precedence (Granger causality), significance thresholds at 5% and 10%.

## Market Efficiency (`figures/efficiency`)
12. **EF01_acf.png**
- **What it shows:** ACF bars for lags 1..20 with confidence bounds.
- **Concept:** Serial dependence in returns; significant autocorrelation indicates deviation from strict weak-form efficiency.

13. **EF02_variance_ratio.png**
- **What it shows:** Variance ratio VR(q) across horizons q={2,4,8,16}, benchmark line at VR=1.
- **Concept:** Random-walk diagnostics; VR?1 suggests mean-reversion or momentum effects.

## Rebalancing (`figures/rebalancing`)
14. **R01_retention_probability.png**
- **What it shows:** Predicted retention probability for next rebalance by symbol, with risk color bands and 0.65 cutoff.
- **Concept:** Inclusion/exclusion risk scoring from classification models.

15. **R02_feature_importances.png**
- **What it shows:** Random-forest feature importances for two tasks: weight prediction and inclusion prediction.
- **Concept:** Relative predictive contribution of engineered features (model-specific importance, not causal effect).

16. **R03_weight_scatter.png**
- **What it shows:** Actual vs predicted constituent weights (test set) with 45° reference line.
- **Concept:** Calibration/accuracy of weight regression; distance from diagonal = prediction error.

17. **R04_weight_changes.png**
- **What it shows:** Predicted weight change (next rebalance minus current weight) per stock.
- **Concept:** Expected reallocation direction and magnitude (potential gainers vs decliners in index weight).

## Summary (`figures/summary`)
18. **SUMMARY_dashboard.png**
- **What it shows:** Combined dashboard of key outcomes: GARCH persistence, flow model directional accuracy, efficiency flags, sector flow with forecast, bottom retention names.
- **Concept:** Executive synthesis of risk, predictability, efficiency, and rebalance risk in one decision view.

## Important interpretation note
- Most plots are **diagnostic and predictive**, not causal proof. Correlation, Granger significance, and feature importance help prioritize hypotheses and forecasting signals, but do not by themselves establish economic causation.
