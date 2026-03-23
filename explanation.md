# Graph Explanations — enhanced-v5 Output

---

## 01 — Feature Correlation Heatmap

**What it shows:** Lower-triangular Pearson correlation matrix of all 12 features plus the
`fund_flow` target.

**Key observations:**
- `lag1_return` ↔ `lag3_return` = **0.83** — strong multicollinearity. These are the same time
  series shifted by different lags. Both appear in the model simultaneously as separate features,
  which inflates Ridge coefficients (as visible in chart 07).
- `lag1_ir` ↔ `lag1_usd` = 0.58 and `lag1_cpi` ↔ `lag1_ir` = 0.66 — macro variables are
  correlated (expected: tightening cycles raise both rates and CPI).
- `lag1_abnorm_vol` ↔ `lag1_volume` = 0.59 — partly redundant because `abnorm_vol` is derived
  from volume.
- **Bottom row (`fund_flow`):** all correlations ≤ |0.19|. The target has near-zero linear
  relationship with every feature. This is the root cause of poor model performance seen in
  charts 04 and 05.

**Verdict:** Correct chart. The near-zero target correlations are genuine and not a bug — they
confirm the dataset is a very hard prediction problem.

---

## 02 — Fund Flow History

**What it shows:** Monthly sum of AKD + NBP + NTI fund flows (after winsorizing at ±3σ), from
Nov 2021 to Nov 2025. Green bars = inflow, red = outflow. Orange dashed lines show the ±3σ
winsorization bounds.

**Key observations:**
- Most months hover near zero. Only 3–4 months have flows large enough to be visible as bars.
- Two months hit exactly the −3σ = −798 floor (bars touching the lower dashed line), meaning
  those months were clipped — their true flows were even more negative.
- The asymmetric bounds (−798 vs +734) are correct: they reflect `mean ± 3×std`, not a
  zero-centred clip.

**Bug — label mix-up in legend:**
The legend reads `−3σ = -798` for the line plotted as the *upper* dashed line visually. In the
code the lines are plotted as `lower_w` (−798) and `upper_w` (+734). The labels in the legend
are assigned in draw order and appear swapped relative to which line is on top vs bottom
visually. This is a cosmetic legend ordering issue, not a data issue.

**Bug — scale makes chart nearly unreadable:**
Most flows are small (|flow| < 50) while the axis spans ±800. The majority of bars render as
hairlines indistinguishable from zero. A log-scale or a secondary zoomed inset would make the
chart useful.

---

## 03 — TimeSeriesSplit CV Folds (5 × 5 grid)

**What it shows:** 25 subplots (5 models × 5 folds). Each subplot compares actual vs predicted
fund flow over the validation window of that fold.

**Key observations:**
- **Folds 1 & 2 (early 2022–2023):** Actual and predicted are both near zero. Models appear to
  work but are essentially predicting the mean of a near-zero series — not learning a pattern.
- **Fold 3 (spans the Jul 2024 large outflow):** Actual drops sharply to −798 (winsorization
  floor). Every model completely misses this — all predict ≈ 0. This is the largest event in
  the dataset and was unpredictable from the lagged features.
- **Folds 4 & 5:** GradBoost and XGBoost predict excessively positive values while actuals are
  near zero or slightly positive. These tree models have memorised the training set (which
  happened to include some positive months) and extrapolate positively.

**Logic issue — fold 3 miss is structural, not a bug:**
The Jul 2024 spike is a rare event with no feature signal. Its being missed is an expected
consequence of the small dataset (~47 usable rows). No code error here, but the pattern
demonstrates why directional accuracy at the fold level fluctuates wildly.

---

## 04 — Holdout Predictions

**What it shows:** All 5 models predict on the last 20% of data (Feb 2025 – Nov 2025,
approximately 10 points). Actual = thick blue line.

**Key observations:**
- May 2025 actual = −798 (winsorization floor). No model comes close. Predictions range from
  −50 to −200.
- All R² values are near 0 or negative: Ridge = 0.04, ElasticNet = 0.02, LightGBM = −0.02,
  GradBoost = −0.12, XGBoost = −0.14.

**Bug — negative R² means the model is worse than predicting the mean:**
GradBoost, XGBoost, and LightGBM all have R² < 0 on the holdout set. A negative R² means the
model's predictions have *higher* MSE than simply using `ȳ_train` as a constant predictor.
The tree models have overfit to the tiny training set.

**Interpretation issue — XGBoost directional accuracy = 40% (worse than coin flip):**
On a 10-point holdout this is 4/10 correct direction calls. With only 10 data points,
the difference between 40% and 60% is exactly 2 observations. No statistical inference can
be drawn from these directional accuracy figures.

---

## 05 — Model Comparison (RMSE + Directional Accuracy)

**What it shows:** Two side-by-side bar charts ranking models by holdout RMSE (left, lower is
better) and directional accuracy (right, higher is better).

**Key observations:**
- All RMSEs are ~240–263, which is large relative to typical month-to-month flow variability.
- Ridge has the best RMSE (240.6) but only 50% directional accuracy — coin flip.
- GradBoost and LightGBM have 60% DA but worse RMSE.

**Logic issue — RMSE highlight and DA highlight are contradictory:**
The green bar in the RMSE chart marks Ridge as "best", but Ridge calls direction correctly only
50% of the time. The green bar in the DA chart marks GradBoost/LightGBM as "best". These two
metrics point to different models, and the chart makes no note of this contradiction.

**Statistical issue — 10-point holdout:**
60% DA vs 50% DA on 10 points = 1 extra correct call. This is not statistically significant
and should not be used to select a "best" model.

---

## 06 — Feature Importance (Tree Models)

**What it shows:** Horizontal bar charts of feature importance for GradBoost, XGBoost, and
LightGBM.

**Bug — incomparable scales across models:**
- GradBoost and XGBoost use **gain-based** importance (scaled 0 to ~0.30).
- LightGBM defaults to **split count** importance (scaled 0 to ~140).

Placing these three plots side-by-side with the same title implies their values are comparable.
They are not. LightGBM's values cannot be meaningfully compared against GradBoost's values
without normalising or using the same `importance_type`.

**What is consistent:**
- `lag1_return` and `lag1_abnorm_vol` rank high in all three models — this aligns with what the
  heatmap shows (their inter-feature correlations are high).
- `lag1_ir_change` is least important across all models — consistent with its near-zero
  correlation with fund_flow.

---

## 07 — Linear Model Coefficients

**What it shows:** Standardised (per-unit of feature std) coefficients for Ridge and ElasticNet,
ordered by absolute magnitude.

**Key observations:**
- Ridge: `lag1_ir` ≈ +55 is the largest positive coefficient and `lag1_abnorm_vol` ≈ −80 is
  the most negative.
- ElasticNet: `lag1_volume` ≈ +38, `lag1_abnorm_vol` ≈ −80 most negative.

**Bug — large inflated coefficients from multicollinearity:**
From the heatmap, `fund_flow`'s correlation with `lag1_ir` is only 0.05 — near zero.
Yet Ridge assigns it a standardised coefficient of ≈ 55. This is a classic Ridge artefact:
`lag1_ir` is correlated with `lag1_usd` (0.58) and `lag1_cpi` (0.66), so Ridge distributes
coefficient weight across a correlated cluster in a compensating pattern. The large coefficients
do NOT indicate `lag1_ir` genuinely predicts fund flow.

**ElasticNet's L1 penalty should shrink irrelevant features to zero**, but even here `lag1_oil`
retains a sizable negative coefficient despite a near-zero raw correlation with `fund_flow`.
This is a sign there are too few training rows for regularisation to fully correct for
multicollinearity at 12 features.

---

## 08 — 6-Month Forecast

**What it shows:** Historical monthly flows (last 12 months as bars) plus a 6-month forward
ensemble forecast (Dec 2025 – May 2026) with individual model lines.

**Bug — historical bars nearly invisible:**
The y-axis spans ±800+ to accommodate the winsorization bounds. Most historical monthly flows
are in the range ±50, rendering as hairlines. The visual impression is that history is all
zeros, which misrepresents the data.

**Bug — individual model forecasts diverge wildly with no error bands:**
Ridge (≈ −450) and ElasticNet (≈ −420) predict deep outflows. GradBoost (≈ −230) moderately
negative. LightGBM (≈ +50) predicts inflow. XGBoost ≈ 0. The ensemble bar sits around −50
to −75, which is much closer to zero than the mean of those model values (which would be
approximately (−450 − 420 − 230 + 50 + 0) / 5 ≈ −210). This discrepancy suggests the
ensemble bar values and the plotted per-model lines may be from different forecast states
(the per-model lines are plotted from `forecast_df[name]` columns which are filled correctly,
but the bar uses `forecast_df['ensemble']`). This is worth verifying.

**Logic issue — feedback mechanism is artificial:**
At each forecast step, only `lag1_return` is updated via:
```python
flow_norm   = ensemble / (abs(y).mean() + 1e-9)
prev_return = np.clip(flow_norm * 0.01, -0.05, 0.05)
```
This converts a *flow prediction* into a pseudo-return via an arbitrary scale factor (0.01).
`lag3_return`, `lag3_volume`, `lag1_vol3_ret`, `lag1_abnorm_vol`, and all macro lags remain
frozen at the last observed values across all 6 months. Month 3's `lag3_return` should reflect
month 0's actual return, not the initial value. The iterative forecast is a partial stub.

---

## 09 — Company Selection

**What it shows:** Two bar charts of top-10 KSE-30 companies by (a) average return and (b)
average volume, computed across months where total fund flow was positive.

**Critical Bug — Bank Of Punjab outlier (110% average monthly return):**
"Bank Of Punjab" shows an average return of ≈1.1 (110% per month) during positive-flow months.
This is physically implausible for a normal market return. This is almost certainly caused by a
price data error in `kse-30-basic.xlsx` (e.g., a single row with an erroneous price causing
`pct_change()` to compute a 1000%+ return). The company-level returns are **never winsorized**
in the pipeline — only fund flows are. This single outlier completely dominates the chart scale,
making all other bars indistinguishable and rendering the top-10 ranking meaningless.

**Fix required:** Winsorize or clip company-level monthly returns before ranking (e.g., at 1–99
percentile), or investigate the raw BOP price data for a data entry error.

**Logic issue — chart shows inflow-strategy candidates but the signal is OUTFLOW:**
The subtitle reads "Next signal: OUTFLOW ↓". According to the project design, outflow → equal
weight portfolio (not the overweight strategy). Yet the chart presents the overweight candidates
prominently. This framing may mislead a reader into applying the inflow-tilt even when the model
says to go equal-weight.

---

## Summary of Issues

| Chart | Type | Severity |
|---|---|---|
| 02 | Legend ordering; unreadable scale | Minor |
| 04 | Negative R² (overfitting tree models) | Major |
| 05 | Statistically meaningless DA on 10 points | Moderate |
| 06 | Incomparable importance scales (split vs gain) | Moderate |
| 07 | Inflated Ridge coefficients from multicollinearity | Moderate |
| 08 | Invisible history bars; artificial feedback loop | Moderate |
| 09 | Unwinsorized company return → BOP outlier | **Critical** |
| 09 | OUTFLOW signal shown with INFLOW strategy chart | Moderate |
