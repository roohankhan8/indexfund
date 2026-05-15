# `3_final_model/` — Planning + results + limitations (enhanced-v1…v8)

This file is the “single source of truth” plan for the `3_final_model/` workstream, but it also captures the **observed results**, **what models were tried**, and the **limitations** you should explicitly state in the FYP report.

## 1. What files exist in this folder (and what they’re for)

### 1.1 Notebooks

- **`enhanced-v1.ipynb` … `enhanced-v8.ipynb`**
  - Iterative experiments for: feature engineering, modeling, evaluation, and forecast → company selection.
  - These are the “research timeline” artifacts and support the report’s *iteration story*.
- **`grok/grok-notebook-v1.ipynb`, `grok/grok-notebook-v2.ipynb`**
  - Alternative idea exploration: proxy-flow approaches, volatility/efficiency metrics, and correlation analysis.

### 1.2 Scripts (reproducible entry points)

- **`scripts/enhanced-v5.py`**
  - Monthly pipeline with multi-lags + macro + CPI, multiple models, CV/holdout, 6-month forecast, and company selection.
- **`scripts/enhanced-v7.py`**
  - Weekly pipeline (more observations) + technical indicators; CV/holdout; 12-week + 2-year forecast; company selection.

### 1.3 Inputs and outputs

- **`data/`**: local copies of model inputs (KSE-30, funds, macro, CPI, index level).
- **`output/`, `output-v5/`, `output-v6/`, `output-v7/`**: generated figures by version.
- **`explanations/`**
  - `v5-explanation.md`, `v7-explanation.md`: narrative interpretation of the saved figures (useful report text).
- **`mds/`**
  - planning and debugging notes (`context.md`, `bugs.md`, `diff.md`, `extra.md`, this `plan.md`).

## 2. Data + target definition (core thesis)

### 2.1 Fund flow (target)

Fund flow is computed per fund and then aggregated:

- Per fund: \(flow_t = AUM_t - AUM_{t-1} \times (NAV_t/NAV_{t-1})\)
- Aggregate target used in most models: `total_fund_flow = flow_AKD + flow_NBP + flow_NTI`

### 2.2 Why modeling is hard here

From the v5 diagnostics (`explanations/v5-explanation.md`):

- Features have **near-zero linear correlation** with the target (all \(\le |0.19|\)).
- Target has **rare extreme events** (large outflow months) that dominate RMSE and are not predictable from lagged features.
- Monthly dataset has only **~47–57 usable rows** after cleaning → high variance estimates and unstable validation.

## 3. Models tried (and why)

### 3.1 Monthly (v5/v6 family)

Typical suite:

- **Ridge** + `StandardScaler` (robust baseline for small N)
- **ElasticNet** + `StandardScaler` (handles correlated features + sparsity)
- **GradientBoostingRegressor** (shallow trees)
- **XGBoostRegressor** and/or **LightGBM** (tree boosting, but high overfit risk on small monthly N)

### 3.2 Weekly + technical indicators (v7 family)

Enhancements:

- Higher frequency aggregation (weekly) to increase sample size (~200+ points).
- Technical indicators on index-level daily returns:
  - **RSI**, **Bollinger Band position**, **MACD**
  - **4-week momentum** and volatility/abnormal volume style signals

## 4. Key results (use directly in the report)

### 4.1 enhanced-v5 (monthly) — holdout results summary

From `explanations/v5-explanation.md` (holdout is ~10 points):

- **R² (holdout)**:
  - Ridge ≈ **0.04**
  - ElasticNet ≈ **0.02**
  - LightGBM ≈ **−0.02**
  - GradBoost ≈ **−0.12**
  - XGBoost ≈ **−0.14**
- **Interpretation**:
  - Negative R² means “worse than predicting the mean”.
  - Directional accuracy on 10 points is not statistically meaningful (1 correct call = 10%).

### 4.2 enhanced-v5 — important failure modes (what you learned)

Also from `explanations/v5-explanation.md`:

- **Rare spikes dominate** (models miss the biggest outflows even after winsorization).
- **Multicollinearity** inflates linear coefficients (large coefficients do not imply genuine predictive power).
- **Feature importance plots can mislead** unless importance scales are normalized across tree libraries.
- **Company selection can be broken by a single outlier return** unless company returns are winsorized/clipped.

### 4.3 enhanced-v7 (weekly + technical)

From `explanations/v7-explanation.md` + code changes visible in `scripts/enhanced-v7.py`:

- **Sample size increases** from ~48 monthly points to **~200 weekly points**, making CV/holdout more stable.
- **Fix applied**: company return ranking uses **winsorized returns (1–99%)** to prevent single-stock outliers dominating “top-10” selection.
- **Limitations still apply**: if the underlying relationship is weak, more frequency adds noise as well as data.

## 5. Limitations (write these explicitly in the thesis)

### 5.1 Data limitations

- **Small effective monthly sample**: after merges/cleaning you typically have <60 rows.
- **Rare-event target**: investor flows include shock months (political/exogenous events) that are not learnable from simple lags.
- **Macro coverage / alignment**: macro series begin later than some market/fund series → aggressive filling can fabricate history if not handled carefully.

### 5.2 Modeling limitations

- **Deep learning (LSTM/GRU)**: not viable at monthly scale; even weekly is small relative to typical DL needs.
- **Tree boosting overfit risk**: boosted trees can easily achieve negative R² on small holdouts if not carefully regularized.
- **Directional accuracy volatility**: with small holdouts, DA changes in large steps and can’t be over-interpreted.

### 5.3 Forecasting limitations (multi-step)

- Any multi-step iterative forecast that “updates” only a subset of lagged features (or holds macros constant) should be described as:
  - **scenario-style** / **directional** forecast, not a point-accurate quantitative forecast.

## 6. What to present as “final” in the report (recommended)

- **Primary thesis pipeline**: monthly flow prediction + direction-based portfolio tilting (conceptual).
- **Evidence for limitations**: show v5 holdout metrics and explain why performance is weak (near-zero correlations + rare events + small N).
- **Improvement attempt**: v7 weekly + technical indicators to increase N; present as an attempt to reduce variance and test whether technical indicators add signal.

## 7. Pointers to the “report-ready” text

- Use these as your figure captions / interpretation text:
  - `3_final_model/explanations/v5-explanation.md`
  - `3_final_model/explanations/v7-explanation.md`