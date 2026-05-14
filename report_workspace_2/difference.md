# Differences and Required Changes

Target file to update:

- `.ai-guidance/current-FYP Report (Analyzing Mutual Funds).md`

Reference baseline from project implementation:

- `5_claude_pipeline/`
- `6_cursor_model/`
- `report_workspace/chapter-03-methodology/images/`
- `1b/eda_kse30.py` outputs

This note lists what must be changed so the report matches what is actually implemented and improves technical quality.

## 1. Title and Focus Correction

Current report language is still broad around mutual funds in general.

Update to explicit implemented scope:

- Primary empirical focus is **KSE-30 index-tracking fund flow dynamics** (AKD, NBP, NTI aggregate and fund-level diagnostics).
- Add that index-level reconstruction and rebalancing prediction are core outcomes.

Recommended wording:

- "Analyzing KSE-30 Index Fund Flow Patterns, Market Efficiency, and Rebalancing Signals in Pakistan Using Predictive Models"

## 2. Data and Period Section Needs Correction

Replace generic statements with implemented details:

- Core stock panel: `kse30_daily_data.csv` (PSX constituents)
- Fund data: `funds_data.xlsx` (AKD/NBP/NTI NAV + AUM)
- Macro: `macro_data.xlsx` (Oil, IR, USD) + `cpi.csv`
- Practical analysis windows in pipeline outputs:
  - Daily master: `2021-01-04` to `2026-04-30` (1300 rows in latest run)
  - Monthly master: `2021-02-26` to `2026-01-30` (60 rows in latest run)

Also add explicit data-quality caveat:

- Mid-period volume gaps/zeros exist and were analyzed explicitly in EDA.

## 3. Methodology Chapter Needs Structural Upgrade

Current methodology text should be rewritten around the implemented multi-track architecture.

Add these subsections:

1. Data ingestion and cleaning rules
2. Feature engineering
3. Stationarity and transformations
4. Flow forecasting models
5. Volatility models
6. Efficiency diagnostics
7. Rebalancing/inclusion prediction
8. Evaluation metrics and holdout design

### 3.1 Feature engineering that must be explicitly documented

- Daily log return
- Rolling volatility (30d)
- Moving averages (20/50)
- Free-float mcap proxies
- Monthly flow construction identity:
  - `flow_t = AUM_t - AUM_(t-1) * (NAV_t / NAV_(t-1))`

### 3.2 Stationarity section is currently missing depth

Add:

- ADF, PP, KPSS tests
- Before/after transformation logic:
  - levels -> log-diff / return / centered / flow_pct
- Mention exported artifacts:
  - `C3_stationarity_before_after.csv`
  - `C3_modeling_daily_transformed.csv`
  - `C3_modeling_monthly_transformed.csv`

## 4. Model List Must Match Actual Code

Ensure report names all used models:

- ARIMAX(1,0,1) style implementation
- VAR(1)
- GARCH(1,1), EGARCH(1,1)
- Ridge, RandomForestRegressor
- LogisticRegression, RandomForestClassifier
- (Mid year models) ElasticNet, XGBoost, LightGBM, GradientBoosting variants

## 5. Results Chapter Must Use Actual Output Tables

Replace narrative-only claims with values from:

- `6_cursor_model/results_fund_flow.csv`
- `6_cursor_model/results_garch.csv`
- `6_cursor_model/results_efficiency.csv`
- `6_cursor_model/results_rebalancing.csv`
- `6_cursor_model/results_rebalancing_forecast.csv`

Examples to include directly:

- Flow model comparison:
  - Naive RMSE `83.10`
  - ARIMAX RMSE `58.56`
  - VAR RMSE `63.10`
- GARCH persistence:
  - `alpha + beta = 0.9667`
- Efficiency snapshot:
  - Runs p `0.0561`
  - VR(2) `1.0069`, p `0.8752`
  - Hurst `0.6559`
- Rebalancing:
  - Ridge/Naive weight RMSE around `0.559`
  - Logistic inclusion AUC around `0.8214`

## 6. Discussion/Limitations Must Be Strengthened

Add explicit caveats already observed in project:

- Small monthly sample size (roughly 47-60 months depending on split)
- Rare-event spikes in flows dominate error metrics
- Some macro series non-stationary in levels
- Volume data gaps can distort liquidity-sensitive aggregates
- Directional signal can be more stable than exact magnitude forecasting

## 7. Recommendation Chapter Needs Practicality Upgrade

Move from generic recommendations to implementation-backed items:

- Maintain transformed/stationary feature set in production
- Add robust missing-volume handling (mask/imputation regimes)
- Add rolling retraining and period-specific model diagnostics
- Expand feature set with policy/event dummies for shock months

## 8. Figure Set to Insert in Report

The following graphs are copied in this folder under `graphs/` and should be used.

### Chapter 3 (Methodology + EDA Quality)

- `C3_EDA_01_coverage_and_volume_quality.png`
- `C3_EDA_02_volume_gap_heatmap.png`
- `C3_EDA_03_market_aggregates.png`
- `C3_EDA_04_return_distribution_and_qq.png`
- `C3_EDA_06_symbol_risk_return_map.png`
- `C3_EDA_07_funds_nav_aum.png`
- `C3_EDA_08_funds_flow_and_corr.png`
- `C3_EDA_ST_01_pvalue_heatmap.png`
- `C3_EDA_ST_03_level_vs_return_examples.png`

### Chapter 4/5 (Model Results and Interpretation)

- `FF01_total_flow_predictions.png`
- `FF02_granger.png`
- `G01_returns_and_vol.png`
- `G02_var_backtest.png`
- `EF01_acf.png`
- `EF02_variance_ratio.png`
- `R01_retention_probability.png`
- `R02_feature_importances.png`
- `R03_weight_scatter.png`
- `R04_weight_changes.png`
- `SUMMARY_dashboard.png`

## 9. Writing Quality Fixes Needed in Current Markdown

The generated markdown has merged cover/table text in places. Before final report drafting:

- Reformat front matter blocks (title page, declaration, contents)
- Standardize terminology:
  - Use "index fund flow" where appropriate
  - Distinguish fund-level vs aggregate sector-level targets
- Remove proposal-tense wording ("will be done") and convert to completion tense ("was implemented", "results show")

## 10. Deliverable in This Folder

This folder now contains:

- `difference.md` (this file)
- `graphs/` (curated report-ready figures copied from project outputs)

