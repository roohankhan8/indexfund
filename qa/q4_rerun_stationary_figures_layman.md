# Q4: Explain all graphs in `6_cursor_model/rerun_stationary/figures` in layman terms

These 18 graphs are the same set as the main pipeline, just from the `rerun_stationary` run.

1. `eda/E01_aum_trend.png`

- It shows how much total money investors kept in KSE-30 index funds each month.
- Difficult words:
`AUM` = Assets Under Management, total money in the fund.
`Trend` = overall direction over time (up/down/flat).

1. `eda/E02_nav_return_dist.png`

- It shows how often small/large daily market moves happened, and compares that shape to a �normal bell curve.�
- Difficult words:
`NAV` = Net Asset Value, per-unit fund value.
`Log return` = percentage-style return using logarithms (better for math over time).
`Distribution` = spread/pattern of values.
`Skew` = whether moves lean more to one side (more big gains or more big losses).

1. `eda/E03_fund_flows.png`

- It shows monthly money entering or leaving the funds. Bars above zero mean inflow; below zero mean outflow.
- Difficult words:
`Net flow` = money in minus money out.
`Spike >2s` = unusually large move; more than about 2 standard deviations from normal.
`Standard deviation (s)` = typical amount of variation.

1. `eda/E04_macro_overview.png`

- It shows big economy variables (oil, USD/PKR, interest rate, gold, GDP) over time to compare with market behavior.
- Difficult words:
`Macro indicators` = economy-wide measures.
`USD/PKR` = dollar-to-rupee exchange rate.
`Interest rate` = policy borrowing cost level.
`GDP YoY` = yearly growth rate versus same period last year.

1. `eda/E05_monthly_correlation.png`

- It shows which variables move together and which move opposite.
- Difficult words:
`Correlation` = relationship in movement direction/strength (-1 to +1).
`Heatmap` = color table showing values.
`+1 / -1 / 0` = strong same direction / strong opposite / no linear relation.

1. `eda/E06_index_cumulative_return.png`

- It shows total market gain/loss path over time if returns are added continuously (log scale style).
- Difficult words:
`Cumulative return` = running total return over time.
`Reconstructed` = rebuilt from underlying stock data.

1. `eda/E07_top_weights.png`

- It shows the 15 stocks with the biggest average share inside the index.
- Difficult words:
`Weight` = percent importance of a stock in the index.
`Average weight` = typical weight over the period.

1. `garch/G01_returns_and_vol.png`

- Top part: daily returns. Bottom part: estimated changing risk level over time.
- Difficult words:
`GARCH` = model that lets volatility change over time.
`Volatility` = how strongly prices swing.
`Conditional volatility` = model�s estimate of current risk given recent history.

1. `garch/G02_var_backtest.png`

- It compares actual losses to a risk limit line (5% VaR). Dots crossing below are risk breaches.
- Difficult words:
`VaR (Value at Risk)` = loss level expected to be exceeded only rarely (here 5% days).
`Backtest` = check model against real past data.
`Exception/Exceedance` = day when actual loss is worse than VaR line.

1. `fund_flow/FF01_total_flow_predictions.png`

- It compares real flow values with model forecasts in train/test periods.
- Difficult words:
`ARIMAX` = time-series model with its own past + outside variables.
`VAR(1)` = model where multiple variables predict each other using 1 lag.
`Stationarity-transformed` = data reshaped to remove drifting level/trend, so model assumptions hold.
`R�` = how much variation model explains (higher usually better).

1. `fund_flow/FF02_granger.png`

- It tests whether past macro values help predict future fund flows.
- Difficult words:
`Granger causality` = �predictive usefulness� test, not true cause-and-effect proof.
`p-value` = probability result is just random chance (smaller = stronger evidence).
`5% / 10% line` = common significance cutoffs.

1. `efficiency/EF01_acf.png`

- It checks whether today�s return is related to recent past returns (lag 1 to 20).
- Difficult words:
`ACF` = autocorrelation function, correlation with past lags.
`Lag` = how many time steps back.
`Confidence bounds` = range where values may be noise.

1. `efficiency/EF02_variance_ratio.png`

- It checks if returns behave like a random walk across different horizons.
- Difficult words:
`Variance ratio (VR)` = compares short-horizon vs long-horizon variance behavior.
`Random walk` = future moves are not predictable from past moves.
`VR=1` = random-walk-like benchmark.

1. `rebalancing/R01_retention_probability.png`

- It shows each stock�s chance of staying in the index at next rebalance. Lower probability means higher exclusion risk.
- Difficult words:
`Retention probability` = chance stock remains included.
`Exclusion risk` = chance stock gets removed.
`Rebalancing` = periodic index reshuffle/update.

1. `rebalancing/R02_feature_importances.png`

- It shows which input signals mattered most in Random Forest models.
- Difficult words:
`Feature importance` = model�s internal ranking of input usefulness.
`Random Forest` = many decision trees combined for prediction.

1. `rebalancing/R03_weight_scatter.png`

- It compares predicted stock weights to actual weights. Closer to diagonal line means better prediction.
- Difficult words:
`Scatter plot` = dots showing two variables pairwise.
`Diagonal (45�) line` = perfect prediction line.

1. `rebalancing/R04_weight_changes.png`

- It shows expected increase/decrease in each stock�s index weight at next rebalance.
- Difficult words:
`Predicted weight change` = forecasted future weight minus current weight.

1. `summary/SUMMARY_dashboard.png`

- It combines key signals from all sections into one decision snapshot.
- Difficult words:
`Dashboard` = one-page combined summary view.
`Directional accuracy` = how often model gets up/down direction right.
`Persistence (GARCH a+�)` = how long volatility shocks tend to last.

