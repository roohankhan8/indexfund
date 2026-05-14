# Chapter 5 Graph Explanations

## Figure 5.1
Path: `report_workspace/chapter-05-results-and-analysis/images/E06_index_cumulative_return.png`

- This graph shows the cumulative return path of the reconstructed KSE-30 index over the final analysis window.
- Use it to give the reader a high-level picture of market direction before discussing volatility and efficiency.

## Figure 5.2
Path: `report_workspace/chapter-05-results-and-analysis/images/E01_aum_trend.png`

- This line chart shows the combined sector AUM of the KSE-30 related funds over time.
- It is mainly used to show concentration and growth in the tracked fund universe.
- A raw data gap in May 2024 had all three funds recorded at zero AUM. The pipeline now repairs that isolated month before plotting, so the figure should be read as a cleaned trend rather than a literal liquidation event.

## Figure 5.3
Path: `report_workspace/chapter-05-results-and-analysis/images/E02_nav_return_dist.png`

- This distribution plot shows that daily returns are non-normal and fat-tailed.
- Use it to support the later choice of volatility models instead of assuming smooth Gaussian behavior.

## Figure 5.4
Path: `report_workspace/chapter-05-results-and-analysis/images/E03_fund_flows.png`

- This bar chart shows monthly aggregate KSE-30 sector net flow, separating inflow and outflow months visually.
- After the AUM repair, the largest negative observed flow month in the final monthly master is 2024-12-31 with approximately -44.01 PKR million, while the strongest positive month is 2025-12-31 with approximately 234.86 PKR million.
- Use this graph to explain that the sector flow series is episodic and shock-prone even after data cleaning.

## Figure 5.5
Path: `report_workspace/chapter-05-results-and-analysis/images/E05_monthly_correlation.png`

- This heatmap summarizes contemporaneous monthly correlations between sector flow, macro variables, and KSE-30 market measures.
- Use it to argue that simple same-period linear relationships are weak, which motivates lagged time-series modelling.

## Figure 5.6
Path: `report_workspace/chapter-05-results-and-analysis/images/FF02_granger.png`

- This graph summarizes Granger-causality p-values for the selected macro variables against aggregate flow.
- Its purpose is to show that no single macro variable dominates the monthly flow process at lag 1.

## Figure 5.7
Path: `report_workspace/chapter-05-results-and-analysis/images/FF01_total_flow_predictions.png`

- This figure compares actual aggregate flow against the ARIMAX and VAR predictions.
- Use it to show that the final models are better interpreted as directional tools rather than exact point estimators.
- In the cleaned final run, ARIMAX has the lower RMSE, while both retained dynamic models reach 75.0% directional accuracy.

## Figure 5.8
Path: `report_workspace/chapter-05-results-and-analysis/images/G01_returns_and_vol.png`

- This chart overlays KSE-30 returns with model-implied conditional volatility.
- It is used to illustrate volatility clustering and the persistence of high-risk episodes.

## Figure 5.9
Path: `report_workspace/chapter-05-results-and-analysis/images/G02_var_backtest.png`

- This figure shows the Value-at-Risk backtest for the preferred volatility specification.
- Use it to explain whether the model's downside risk envelope is broadly calibrated to actual tail events.

## Figure 5.10
Path: `report_workspace/chapter-05-results-and-analysis/images/EF02_variance_ratio.png`

- This graph visualizes the variance-ratio evidence across horizons or sub-periods.
- It supports the claim that some efficiency tests are closer to random-walk behavior than others.

## Figure 5.11
Path: `report_workspace/chapter-05-results-and-analysis/images/EF01_acf.png`

- This autocorrelation plot highlights whether the reconstructed KSE-30 return series contains serial structure across lags.
- Use it together with the Ljung-Box and Hurst results when discussing mixed efficiency evidence.
