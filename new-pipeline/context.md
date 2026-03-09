# enhanced.ipynb — Context & Documentation

## Overview

Predicts monthly mutual fund flows (net investor inflows/outflows) into KSE-30 index funds using lagged market signals and macroeconomic variables. The predicted flow is used to tilt portfolio weights — overweight high-volume stocks on inflows, go equal-weight on outflows.

---

## Data Sources

| File | Sheet | Columns | Frequency | Range |
|------|-------|---------|-----------|-------|
| `kse-30-basic.xlsx` | — | date, symbol, company, price, idx_wt_%, volume | Daily | 2020–2026 |
| `funds_data.xlsx` | AKD, NBP, NTI | DATE, NAV, AUM | Daily | 2020–2026 |
| `macro_data.xlsx` | OIL | DATE, PRICE (Brent USD) | Daily | 2021–2026 |
| `macro_data.xlsx` | USD | DATE, USD (PKR/USD) | Daily | 2021–2026 |
| `macro_data.xlsx` | IR | DATE, RATE (KIBOR %) | Irregular | 2021–2026 |

---

## Pipeline Steps

**Fund flow formula (net investor flow, stripping out market-driven AUM change):**
```
flow_t = AUM_t - AUM_{t-1} × (NAV_t / NAV_{t-1})
```

1. **Load KSE-30 data** — parse Excel serial dates, clean nulls, sort by date+company
2. **Compute weighted price** — `weighted_price = price × idx_wt_%` per stock per row
3. **Daily aggregate** — group by date: `total_volume` (sum), `weighted_return` (sum of weighted prices across all 30 stocks)
4. **Monthly resample** — `total_volume` (sum), `weighted_return` (mean); compute `log_return = log(WR_t / WR_{t-1})`
5. **Fund flow per fund** — iterate over AKD, NBP, NTI; compute `nav_ratio` and `flow`; concat and sum to `total_fund_flow` per month
6. **Macro processing:**
   - Oil: daily → monthly mean
   - Interest rate: irregular changes → daily forward-fill → monthly last value
   - USD: daily → monthly mean
7. **Merge** all monthly series on date (left join from KSE-30 monthly series)
8. **Imputation** — `total_fund_flow` NaN → 0; remaining NaN via `ffill`; remaining zeros replaced with column mean *(problematic — see Known Issues)*
9. **Lag features** (1-month lag): `lag_volume`, `lag_return`, `lag_oil`, `lag_ir`, `lag_usd`; drop NaN rows from shift
10. **Train/test split** — 80% train (chronological), 20% test (~14 rows)
11. **Random Forest** — 100 estimators, no depth limit, `random_state=42`; fit on 5 lag features; predict `total_fund_flow`
12. **Evaluate** — RMSE, **R² = −0.2896**, Adjusted R² = −1.0957
13. **Next-month prediction** — predict from the most recent row's lag features
14. **Portfolio suggestion** — predicted flow > mean → overweight top-10 volume stocks; otherwise equal-weight

---

## Known Issues

1. **Macro data starts 2021; KSE-30 starts 2020** — all of 2020 (12 rows) has macro values imputed to the column mean, which are factually wrong and corrupt training data
2. **Negative R²** — model performs worse than predicting the mean (see below)
3. **Zero-to-mean imputation** — replacing zeros with column mean injects artificial observations into `total_fund_flow` for months before fund data begins

---

## Why R² is Negative

R² = −0.2896 means the model predicts **worse than simply guessing the mean** every time. The formula is `R² = 1 − SS_res / SS_tot`; when residual errors (SS_res) exceed the total variance of the target (SS_tot), R² goes negative.

### Root Causes

**1. Corrupted macro features for all of 2020 (12 months — worst cause)**

Macro data starts in 2021. `ffill` finds nothing to fill backward into 2020, so the zero-to-mean substitution step produces identical macro values for every 2020 row:
- `oil_price = 80.56` (actual 2020 average: ~42 USD — COVID crash year)
- `interest_rate = 15.81` (actual 2020 SBP rate: ~7%)
- `usd_exchange = 238.80` (actual 2020 rate: ~160 PKR/USD)

The model trains on these fabricated flat values, then at test time encounters real varying macro data. The distributions are completely different, causing systematic prediction errors.

**2. Very small test set (14 observations)**

~57 total monthly rows → 80/20 split → ~14 test rows. One large misprediction accounts for a huge fraction of SS_res and is enough to push R² negative. At this scale R² is statistically unreliable.

**3. Extreme outlier in target variable**

`total_fund_flow` has one major outlier: **−62.7 in April 2022** (Pakistan political crisis). Most months are in the −5 to +15 range. The model predicts −176 for that period, which is wildly off and dominates the error metric.

**4. Random Forest overfitting on 56 training rows**

100 trees with no `max_depth` constraint memorises the training set entirely. The test period is a structurally different macro regime (stable high interest rate, recovered USD), so training patterns don't transfer.

**5. Only 1-month lags, only 3 funds**

Signal from 3 fund flows is noisy. Single-month lags give the model very limited memory of market dynamics.

---

## How to Improve

### Data Fixes (highest impact — do these first)

- **Drop 2020 rows** or source real 2020 macro data from SBP/PSX archives. Training on fabricated values actively harms the model.
- **Winsorize `total_fund_flow`** at ±3 standard deviations. The April 2022 outlier is a one-off political event, not a learnable pattern.
- **Replace zero-to-mean imputation** with `ffill` only, or drop rows with missing macro.

### Evaluation Fixes

- Replace single 80/20 split with **`TimeSeriesSplit(n_splits=5)`** from sklearn — gives 5 test windows and more honest metrics.
- Use **RMSE** as the primary metric; R² on 14 samples is meaningless.

### Feature Engineering

- Add 3-month and 6-month rolling lags (not just 1-month)
- Add `interest_rate_change = ir_t − ir_{t-1}` and `usd_pct_change` — direction matters more than level
- Add return volatility (rolling 3-month std of `log_return`)

### Model Recommendations

| Model | Why it suits this data | Notes |
|-------|----------------------|-------|
| **Lasso / Ridge Regression** | Low variance; handles small samples + correlated macro features; interpretable | Start here — run as baseline first |
| **Elastic Net** | Combines L1+L2; performs automatic feature selection | Good when some lags are irrelevant |
| **ARIMAX** | Purpose-built for monthly time series + exogenous macro variables; uses past flows as AR terms | Best choice while dataset stays small; no overfitting risk |
| **Gradient Boosting** (`max_depth=2`, `n_estimators=50`) | Learning rate shrinkage prevents overfitting vs. RF | sklearn `GradientBoostingRegressor` |
| **LightGBM** (`num_leaves=15`, `min_data_in_leaf=5`, `lambda_l1=1`) | Better regularization than XGBoost on small data | Tune via TimeSeriesSplit CV |
| **Bayesian Ridge** | Produces prediction intervals; robust to small samples | Useful for communicating confidence |
| **LSTM / GRU** | Deep learning for sequences | **Not viable at 57 rows** — needs 500+ observations minimum |

### Recommended Sequence

1. Fix data quality (drop 2020 or get real macro; winsorize outlier)
2. Run **Lasso + TimeSeriesSplit** as baseline
3. Run **ARIMAX** — compare on AIC/BIC and RMSE
4. Benchmark **LightGBM** with regularized hyperparameters
5. Compare all models on RMSE and **direction accuracy** (% of months where sign of prediction matches actual sign of flow)
