# Graph Explanations — enhanced-v7 (Weekly + Technical Indicators)

---

## Overview

**enhanced-v7** switches from monthly to weekly predictions to increase the number of data points from ~48 to ~200+. It also adds technical indicators (RSI, Bollinger Bands, MACD) as features computed from daily KSE-30 index data.

**Key Changes vs v6:**
- Weekly fund flows instead of monthly
- Technical indicators: RSI, Bollinger Band Position, MACD
- Momentum features: 4-week cumulative return
- 12-week forecast instead of 6-month

---

## 01 — Feature Correlation Heatmap

**What it shows:** Lower-triangular Pearson correlation matrix of 13 features plus the `fund_flow` target.

**Key observations:**
- `lag1_return` ↔ `lag2_return` — same time series at different lags, moderate correlation
- `lag1_vol4_ret` ↔ `lag1_momentum` — related (volatility vs momentum)
- Technical indicators (`lag1_rsi`, `lag1_bb_pos`, `lag1_macd`) show some correlation with each other
- **Bottom row (`fund_flow`):** Correlations with all features should be checked — if still near zero, the fundamental data limitation persists

**Verdict:** Should show similar patterns to v6 but with the new technical indicator features.

---

## 02 — Weekly Fund Flow History

**What it shows:** Weekly fund flow (sum of AKD + NBP + NTI) from 2021 to late 2025. Green bars = inflow, red = outflow.

**Key observations:**
- Much more granular data (200+ bars vs 48)
- Values are smaller (weekly vs monthly) — typically ±50-100 PKR
- More noise visible due to higher frequency
- Orange dashed lines show ±3σ winsorization bounds

**Improvements:**
- Dual-axis not needed since values are more uniform
- More visible pattern with many small flows

---

## 03 — TimeSeriesSplit CV Folds (5 × 5 grid)

**What it shows:** 25 subplots (5 models × 5 folds). Each subplot compares actual vs predicted weekly fund flow.

**Key observations:**
- With more data points per fold, patterns should be more visible
- Tree models may show less overfitting with more training data
- Early folds (2021-2022) should show predictions near zero (minimal activity)
- Later folds (2024-2025) capture more market activity

**Note:** The weekly model has ~200 rows, so each fold has ~40 training and ~40 validation points (vs ~10 in monthly).

---

## 04 — Holdout Predictions

**What it shows:** All 5 models predict on the last 20% of weekly data (~40 points). Actual = thick blue line.

**Key observations:**
- More holdout points (40 vs 10 in monthly) = more statistically meaningful
- R² values should be more stable with more data
- Watch for negative R² indicating overfitting

**Metric interpretation with 40 points:**
- 1 correct call = 2.5% difference
- Still needs caution but more reliable than 10-point holdout

---

## 05 — Model Comparison (RMSE + Directional Accuracy)

**What it shows:** Two side-by-side bar charts ranking models by holdout RMSE (left) and directional accuracy (right).

**Key observations:**
- RMSE values will be smaller (weekly flows are smaller magnitude)
- Directional accuracy more meaningful with 40 points
- Ridge typically best for RMSE, tree models for DA

---

## 06 — Feature Importance (Tree Models)

**What it shows:** Horizontal bar charts of normalized feature importance for GradBoost, XGBoost, and LightGBM.

**Key observations:**
- **New features to watch:** `lag1_rsi`, `lag1_bb_pos`, `lag1_macd`, `lag1_momentum`
- If technical indicators rank high, they may capture investor sentiment
- Compare across all three tree models for consistency

---

## 07 — Linear Model Coefficients

**What it shows:** Standardised coefficients for Ridge and ElasticNet, ordered by absolute magnitude.

**Key observations:**
- Technical indicator coefficients show their predictive contribution
- Multicollinearity warning still applies (large values may be artefacts)
- Compare coefficient signs with intuition (e.g., high RSI might mean overbought → outflow)

---

## 08 — 12-Week Forecast

**What it shows:** Historical weekly flows (last 12 weeks as bars) plus 12-week forward ensemble forecast.

**Key observations:**
- Weekly granularity for tactical decisions
- Direction signal more actionable (rebalance monthly)
- Error bands not shown — treat as directional only
- Ensemble combines 5 models for robustness

---

## 09 — Company Selection

**What it shows:** Top 10 KSE-30 companies by return and volume during positive flow periods.

**Key observations:**
- Uses winsorized returns (1-99%) to handle outliers
- Strategy selection based on forecast direction:
  - OUTFLOW → Equal-weight (volume ranking primary)
  - INFLOW → Overweight (return ranking primary)

**Bug fix from v6:** Returns are now properly winsorized to avoid BOP outlier issue.

---

## Summary of Changes v6 → v7

| Aspect | v6 (Monthly) | v7 (Weekly) |
|--------|---------------|-------------|
| Data points | ~48 | ~200 |
| Features | 12 | 13 (+RSI, BB, MACD) |
| Forecast horizon | 6 months | 12 weeks |
| Fund flow values | ±800 PKR | ±50 PKR |
| Holdout points | 10 | ~40 |
| Technical indicators | No | Yes (RSI, MACD, Bollinger) |

---

## Data Limitations (Still Apply)

Despite weekly granularity, the fundamental challenge remains:

1. **Near-zero correlations:** Features likely still have weak correlation with fund flow
2. **Efficient market:** Hurst exponent ~0.4, VR ~0.25 suggests random walk
3. **Small sample problem:** Even 200 weekly observations may not be enough for reliable ML
4. **Noise vs signal:** Weekly fund flows are noisy — most weeks have minimal activity

---

## Recommendations for Interpretation

1. **Focus on direction, not magnitude** — predictions are better used for flow direction (inflow/outflow)
2. **Technical indicators are exploratory** — RSI/MACD may capture overbought/sold conditions affecting investor behavior
3. **Compare to v6** — run both and see if weekly + technical indicators improve directional accuracy
4. **Use ensemble** — combining 5 models reduces variance
5. **Re-validate quarterly** — with more data accumulating, retest model performance

---

## Technical Details

### Fund Flow Formula (Weekly)
```
flow_week = AUM_end_of_week - AUM_start_of_week × (NAV_end / NAV_start)
```

### Technical Indicators
- **RSI (14-day):** Relative Strength Index — values >70 = overbought, <30 = oversold
- **Bollinger Band Position:** Where price sits relative to 20-day bands — captures volatility
- **MACD:** 12-day EMA - 26-day EMA — trend momentum indicator
- **4-week momentum:** Sum of last 4 weeks' returns — trend strength

### Aggregation
- KSE-30: Daily → Weekly (Fridays) via sum
- Fund flows: Daily → Weekly via sum (computed directly from daily data)
- Macro: Monthly → Weekly via forward-fill