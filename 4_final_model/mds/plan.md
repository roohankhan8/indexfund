# enhanced-v5.py — Analysis & Implementation Plan

## 1. Notebook Lineage Analysis

### enhanced-v1 → Baseline Pipeline
- Established core architecture: KSE-30 + 3 funds (AKD, NBP, NTI) + macro (oil, IR, USD)
- Fund flow formula: `flow = AUM_t - AUM_{t-1} × (NAV_t / NAV_{t-1})`
- Model: Random Forest (100 trees, unconstrained) — high overfitting risk at ~57 rows
- 5 lagged level features only
- **Defect**: `bfill()` fabricates 2020 macro values from 2021 data (oil ~80 vs actual ~42, IR ~15% vs ~7%, USD ~238 vs ~160)
- **Good**: Winsorization at ±3 std to handle the April 2022 political event outlier (-62.7)
- **Good**: Introduced TimeSeriesSplit(5) CV and plot per fold

### enhanced-v2 → Data Quality Fix + Better Model
- **Fix**: Drops all 2020 rows → avoids fabricated macro data that corrupted training
- **Fix**: `ffill()` only (no `bfill`) for macros
- **Fix**: Ridge + StandardScaler replaces Random Forest (appropriate for ~47 rows)
- **Added**: First-difference features `lag_ir_change`, `lag_usd_pct_change` (more stationary across regimes)
- **Reduced prediction to directional signal only** (magnitude unreliable at this size)
- Mean CV RMSE: 140.50 (high std of 130.59 — fold 3 RMSE 339.55 is still dominated by outliers)

### enhanced-v3 → Multi-Model Comparison + Directional Accuracy
- Three models: Ridge, XGBoost (max_depth=3, lr=0.05), LSTM
- **Best addition**: Directional accuracy as explicit metric (most actionable for investment signal)
- **Reverted** to including 2020 rows (regression from v2)
- Same 7 features as v2
- Ridge coefficient inspection for interpretability
- LSTM: 1 timestep → no sequence benefit, essentially a feedforward net

### grok-notebook-v1 → Independent Approach (No Fund Data)
- **Unique**: Synthetic flow proxy = `abnormal_volume × sign(return) × |return|`
- **Unique**: Hurst Exponent (1.052) + Variance Ratio (2.724) — confirms market persistence
- Multi-lag features: 1, 3, 6 months for returns and abnormal volume
- `avg_weight_change` = mean abs daily weight change/month (rebalancing pressure signal)
- No macro data — useful as a fallback if fund data is unavailable
- Target shifted -1 (next month's proxy) — proper predictive horizon

### grok-notebook-v2 → Per-Fund Normalized Flows + Correlation Analysis
- **Unique**: Flow normalized as fraction of AUM (percentage not absolute PKR)
- Multi-lag (1 and 3 months) for both returns and all 3 macro variables
- **Unique**: Correlation heatmap between flows and macro variables
- Per-fund level tracking across AKD, NBP, NTI
- Hurst Exponent + Variance Ratio included
- Known runtime error in portfolio cell (idx_wt_% missing after merge)

### enhanced-v4 → CPI + 6-Month Forecast + Auto Model Selection
- **Unique**: CPI (YoY%) as 6th feature — inflation signal
- **Unique**: 6-month iterative forward forecast (other notebooks: 1 month only)
- Auto-selects 2 best models by RMSE, averages their predictions
- Company selection by avg return during positive flow months (vs volume rank in v1-v3)
- **LSTM failure**: relu activation + no dropout + 1 timestep → RMSE 188,366
- First-difference features dropped (regression from v2/v3)
- 2020 rows not dropped (regression from v2)

---

## 2. Critical Issues to Fix in v5

| Issue | Present In | Fix |
|---|---|---|
| Fabricated 2020 macro values via bfill | v1, v3, v4 | Drop rows where year < 2021 |
| LSTM with 1 timestep and relu | v3, v4 | Drop LSTM entirely |
| Future lag update: `log(1 + avg_pred/1000)` | v4 | Hold macros at last known, document assumption |
| Zero-fill on fund flow NaN | all | Keep fillna(0) for fund flow only; ffill for macros; drop remaining NaN |
| No winsorization in v4 | v4 | Add back at ±3 std before feature engineering |
| Only 1-month lags (v1-v4 enhanced) | v1-v4 | Add 3-month lags for all key features |
| No directional accuracy in v4 | v4 | Add back from v3 |
| CPI but dropped first-diff features | v4 | Keep both CPI and first-diff macros |
| TSCV dropped in v4 | v4 | Use TimeSeriesSplit(5) instead of single 80/20 split |

---

## 3. Proposed Architecture for enhanced-v5.py

### 3.1 Data Pipeline (No Changes to Raw Loading Logic)

```
Load KSE-30 → aggregate daily → resample monthly → log_return
Load AKD/NBP/NTI → nav_ratio → flow → monthly sum
Load oil/IR/USD → monthly resample
Load CPI → monthly last
Merge all on date (left join from KSE-30 monthly)
```

**Filtering:**
- Winsorize `total_fund_flow` at ±3 std (before merging features)
- Drop year < 2021 (macro starts 2021)
- ffill macros within range, fillna(0) only for fund flow first-row NaN

### 3.2 Feature Set (12 features total)

| Feature | Source | Why |
|---|---|---|
| `lag1_volume` | KSE-30 | Trading activity (1-month lag) |
| `lag3_volume` | KSE-30 | Medium-term trading trend |
| `lag1_return` | KSE-30 | Market performance signal |
| `lag3_return` | KSE-30 | Trend persistence |
| `vol3_return` | KSE-30 | 3-month rolling std of log_return — volatility signal |
| `lag1_abnorm_vol` | KSE-30 | Abnormal volume = volume / 3m rolling avg |
| `lag1_ir` | IR | Level matters for absolute cost of capital |
| `lag1_ir_change` | IR | Rate direction (cut/hike cycle) |
| `lag1_usd` | USD | Exchange rate level |
| `lag1_usd_pct` | USD | Currency pressure direction |
| `lag1_oil` | OIL | Macro global signal |
| `lag1_cpi` | CPI | Inflation eats real returns → flow deterrent |

**Rationale for each addition vs v4:**
- `lag3_*`: multi-month memory of macro regime (from grok-v2 insight)
- `vol3_return`: volatility deters flow (from context.md recommendations)
- `lag1_abnorm_vol`: captures institutional vs retail flow pressure (from grok-v1)
- `lag1_ir_change` + `lag1_usd_pct`: first-diff features from v2 — direction more stationary than level

### 3.3 Model Suite (4 models, no LSTM)

| Model | Config | Rationale |
|---|---|---|
| **ElasticNet** | `l1_ratio=0.5, alpha=0.1`, StandardScaler | Auto feature selection + L2 stability; never tried yet |
| **Ridge** | `alpha=1.0`, StandardScaler | Proven baseline from v2/v3/v4 |
| **LightGBM** | `num_leaves=15, min_data_in_leaf=5, lambda_l1=1, n_estimators=100` | Better regularization than XGBoost on small data |
| **GradientBoosting** | `max_depth=2, n_estimators=50, learning_rate=0.05` | From context.md recommendation; shallower than v3/v4 XGB |

Drop: LSTM (not viable at ~50 rows), Random Forest (high variance, from v1).

### 3.4 Evaluation

- **CV**: `TimeSeriesSplit(n_splits=5)` — not single 80/20 (avoids v4 mistake)
- **Metrics per fold and overall**:
  - RMSE (primary)
  - MAE
  - Directional Accuracy: `mean(sign(y_true) == sign(y_pred)) * 100`
- **Final holdout**: last 20% for final metrics table
- **Feature importance** table from LightGBM + GradientBoosting
- **Ridge/ElasticNet coefficients** for interpretability

### 3.5 Forecasting

- **Horizon**: 6 months (from v4)
- **Assumption**: macro features held at last known values (documented in code)
- **Ensemble**: average predictions across all 4 models (not just top 2)
- **Output**: predicted direction with confidence (majority vote on sign)

### 3.6 Portfolio / Company Selection

- Use v4 approach: top 10 companies by avg return **during months with positive predicted flow**
- Secondary: top 10 by avg volume during positive flow months (v1-v3 approach) as a fallback
- Output: ranked table with both metrics side by side

---

## 4. File Structure

`enhanced-v5.py` as a single Python script (not notebook), organized in sections:

```
# --- CONFIG ---
# --- STEP 1: DATA LOADING ---
# --- STEP 2: FUND FLOW COMPUTATION ---
# --- STEP 3: MACRO + CPI PROCESSING ---
# --- STEP 4: MERGE AND CLEAN ---
# --- STEP 5: OUTLIER HANDLING (WINSORIZE) ---
# --- STEP 6: FEATURE ENGINEERING ---
# --- STEP 7: MODEL TRAINING (TimeSeriesSplit) ---
# --- STEP 8: HOLDOUT EVALUATION ---
# --- STEP 9: 6-MONTH FORECAST ---
# --- STEP 10: COMPANY SELECTION ---
# --- STEP 11: SUMMARY OUTPUT ---
```

---

## 5. What v5 Fixes vs v4 (all issues from bugs.md)

| Bug # | Description | v5 Fix |
|---|---|---|
| 1 | Incorrect return calc (weighted price ≠ return) | Document limitation clearly; use KSE-30 index level from `kse30_index_level.csv` if available |
| 2 | Placeholder lag update formula | Hold all macros constant; document this as the assumption |
| 3 | Incomplete future feature updates | All 12 features documented as held constant for forecast |
| 4 | Missing data: fillna(0) on fund flow | Only fill NaN with 0 for fund flow first row; ffill macros; dropna on features |
| 5 | First row flow NaN → 0 | Same — accept this as minor data loss |
| 6 | LSTM 1 timestep | Dropped entirely |
| 7 | Hardcoded 80/20 split | Replaced with TimeSeriesSplit(5) |
| 8 | No date range validation | Added: print date range and NaN count per column after merge |
| 9 | Company selection historical bias | Acknowledged in comments |
| 10 | NaN in company returns | Add `.dropna()` filter before averaging |
| 11 | IR forward-fill limitation | Same approach; print the IR coverage dates |
| 12 | Unused TimeSeriesSplit import | Clean imports — only import what's used |
| 13 | Duplicate date handling | Add `drop_duplicates(subset=['date','company'])` on load |
| 14 | Bare except | Replace with `except (ValueError, TypeError)` |
| 15 | LSTM no random seed | Dropped — not applicable |

---

## 6. Key Design Decisions

1. **Python script over notebook**: Easier to run headlessly, cleaner prints, no cell ordering issues
2. **No LSTM**: Dataset too small (~47 rows after cleaning); adds noise not signal
3. **ElasticNet as new model**: Never tested across the v1-v4 series; combines L1 + L2, handles correlated macro features with automatic zero-ing out of weak features
4. **LightGBM over XGBoost**: Better handling of small `min_data_in_leaf`, stronger built-in regularization, faster
5. **kse30_index_level.csv**: There is a `kse30_index_level.csv` file in the data folder — this should be used to compute actual index log returns instead of the weighted_price proxy that all v1-v4 notebooks use
6. **12 features**: Adds meaningful signal (multi-lag, volatility, abnormal volume, CPI) without exploding the feature space relative to ~47 training rows (ratio ~4:1 rows per feature — acceptable for regularized linear models)

---

## 7. kse30_index_level.csv — Confirmed Useful

Inspected the file. Columns:
```
date, index_return, log_return, total_volume, avg_weight_change, num_companies
```

This is already a **pre-computed daily index dataset** with:
- `log_return`: proper index log return (fixes Bug #1 — no more weighted_price proxy)
- `total_volume`: same as sum from kse-30-basic.xlsx
- `avg_weight_change`: mean absolute daily weight change — same signal as grok-v1's unique feature

**v5 will use this file as the primary KSE-30 source** for returns and volume. The `kse-30-basic.xlsx` is only needed for company-level price data (Step 10: company selection). This simplifies Step 1 significantly.

---

## 8. Open Questions Before Proceeding

1. **Script vs Notebook**: The task says `enhanced-v5.py` — confirmed as a plain Python file, not a Jupyter notebook?
2. **Matplotlib output**: Should plots be saved to files (PNG) or shown interactively (`plt.show()`)?
3. **LightGBM dependency**: Is `lightgbm` installed in the project venv, or should we substitute with sklearn `GradientBoostingRegressor` only?

---

## 9. For advanced: Use PyPortfolioOpt (pip install PyPortfolioOpt)
``` from pypfopt import expected_returns, risk_models, EfficientFrontier ```
... compute optimal weights based on predicted returns = f(flow_pred)