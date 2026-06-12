# Project Team Overview (With Graph/Result Explanations)

This document explains **what was built** and **how the key graphs/results should be interpreted** so your team can present and defend the work without reading the code.

---

## 1) Project story in one page (what we completed)
We built an end-to-end quantitative pipeline that studies **Pakistan’s KSE-30 index** using:

- **Market data**: KSE-30 constituent prices/weights (from PSX)
- **Fund data**: NAV/AUM time series from three KSE-30-related index-tracking funds: **AKD, NBP, NTI**
- **Macro data**: oil price, interest rate, USD/PKR, CPI (and optionally others in earlier branches)

### The 4 main questions the project answered
1. Can we forecast an **aggregate sector fund-flow proxy** related to KSE-30?
2. Can we estimate **time-varying volatility** (risk) for KSE-30?
3. Does KSE-30 look **efficient** (random-walk-like) or contain **predictable structure**?
4. Can we predict **rebalancing outcomes**: which stocks are likely to stay and how weights might change?

### Crucial clarification for your presentation
We do **not** use a single official “KSE-30 fund flow” series. Instead, we construct a **proxy** called **aggregate sector flow**, computed from the fund-level AUM dynamics (AKD + NBP + NTI).

---

## 2) Completion process (how the system was built)
Think of it as a pipeline with seven major “product stages”:

1. **Collect & standardize input time series**
   - KSE-30 constituent panels (prices, weights, volume, market-cap proxies)
   - Fund NAV/AUM series for AKD/NBP/NTI
   - Macro variables (oil, interest rate, USD/PKR, CPI)

2. **Clean + transform**
   - remove duplicates / handle missingness
   - convert prices to **log returns**
   - compute **rolling volatility** and other technical features

3. **Build master datasets** (shared foundation)
   - `daily_master.csv` (daily market + fund + macro features)
   - `monthly_master.csv` (month-end aggregated features for forecasting fund flows)
   - `kse30_stocks_clean.csv` (cleaned constituent panel)

4. **Exploratory Data Analysis (EDA)**
   - check distributions, heavy tails, stationarity behavior
   - quantify concentration of index weights

5. **Modeling block A: fund-flow forecasting**
   - naive baseline
   - ARIMAX-style and VAR-style time-series forecasts

6. **Modeling block B: volatility (GARCH family)**
   - estimate conditional volatility (GARCH + EGARCH)
   - validate using a VaR-style backtest

7. **Modeling block C: market efficiency diagnostics**
   - runs test, variance ratio, Ljung–Box, Hurst exponent

8. **Modeling block D: rebalancing prediction (final practical output)**
   - predict **retention/inclusion** probability (classification)
   - predict **weights/weight changes** (regression)
   - forward forecast exclusion risk across constituents

9. **Held-out validation**
   - evaluate the rebalancing approach on a later unseen cycle (March 2026)

---

## 3) Key graphs and how to explain them (from the final report)
Below are the graphs your team should mention, with an interpretation that matches the **final report**.

### A) Data/EDA graphs

#### Figure 3.1 — “Top 15 stocks by average KSE-30 weight”
**What it shows:** Index concentration.

**Team explanation:** A small set of high-cap stocks dominate index movement. This matters because when rebalancing happens, changes in the largest constituents can dominate the portfolio’s risk-return profile.

---

#### Figure 3.2 — “Monthly correlation matrix for KSE-30 flow, macroeconomic variables, and index measures”
**What it shows:** Linear relationships across variables.

**Team explanation:** Correlations between aggregate flow and macro factors are weak contemporaneously. This is why time-series models that use **lags and joint dynamics** (ARIMAX/VAR) are justified.

---

#### Figure 3.3 — “KSE-30 return distribution and Q-Q plot”
**What it shows:** Non-normality (fat tails).

**Team explanation:** Returns have heavy tails and deviate from normality. This supports using **GARCH-family volatility modeling** rather than assuming constant variance.

---

#### Figure 3.4 — “Comparison of price levels (non-stationary) vs. returns (stationary)”
**What it shows:** Stationarity difference.

**Team explanation:** Price levels trend (non-stationary), while log returns look stationary. Modeling uses returns because many time-series methods assume stationarity.

---

#### Figure 3.5 — “P-value heatmap for ADF stationarity tests”
**What it shows:** Which series become stationary after differencing.

**Team explanation:** Raw levels (e.g., CPI/interest rate) require transformation to achieve stationarity, ensuring the econometric forecasting block uses appropriate data forms.

---

### B) Fund-flow forecasting graphs

#### Figure 4.1 — “Reconstructed cumulative KSE-30 return”
**What it shows:** Cumulative behavior of the reconstructed KSE-30 return series.

**Team explanation:** Provides context for the full modeling sample and supports later volatility and efficiency interpretations.

---

#### Figure 4.2 — “AUM trend and aggregate KSE-30 sector total”
**What it shows:** Fund concentration and sector AUM growth.

**Team explanation:** AKD dominates the aggregate sector AUM, meaning aggregate flow dynamics are heavily influenced by the largest fund’s behavior.

---

#### Figure 4.3 — “Aggregate monthly KSE-30 sector fund flows showing alternating inflow/outflow episodes”
**What it shows:** Flow episodes and outliers.

**Team explanation:** Flows are not smooth; they move in inflow/outflow episodes with sharp shocks. That explains why models need to capture time dependence and why directional performance can be more realistic than precise PKR magnitude prediction.

---

#### Figure 4.4 — “Lag-1 Granger causality p-values for macro variables vs aggregate fund flow”
**What it shows:** Whether single macro variables provide isolated predictive lead power.

**Team explanation:** No macro variable strongly passes a strict significance threshold alone. This is why the final approach relies on **multivariate joint dynamics** (ARIMAX/VAR) rather than one-factor signals.

---

#### Figure 4.5 — “Actual vs predicted aggregate sector flow (ARIMAX and VAR)”
**What it shows:** Forecast fit vs reality.

**Team explanation:** Even when exact magnitude is difficult (negative R²), the models improve **directional accuracy** (inflow vs outflow), which is the operationally useful signal.

---

### C) Volatility modeling graphs

#### Figure 4.6 — “KSE-30 daily returns and model-implied conditional volatility”
**What it shows:** Volatility over time.

**Team explanation:** Conditional volatility rises and falls over time, consistent with volatility clustering. The GARCH/EGARCH fit provides time-varying risk estimates.

---

#### Figure 4.7 — “5% VaR backtesting for the preferred volatility model”
**What it shows:** Risk model calibration.

**Team explanation:** The number of VaR exceedances is close to the expected 5% level, supporting that the selected volatility model produces a reasonable downside risk envelope.

---

### D) Market efficiency graphs

#### Figure 4.8 — “Autocorrelation structure of reconstructed KSE-30 returns”
**What it shows:** Serial dependence signal.

**Team explanation:** Combined efficiency tests are **mixed**: variance ratio and runs test suggest near-random-walk behavior in parts of the sample, but Ljung–Box and Hurst indicate persistence/serial dependence.

**How to summarize this slide in one sentence:** KSE-30 is not purely efficient; it has **pockets of predictability**, especially over longer horizons.

---

#### Figure 6.2 — “Summary of market efficiency evidence for KSE-30”
**What it shows:** A consolidated view of multiple efficiency lenses.

**Team explanation:** Different tests disagree because they measure different properties (sign randomness, variance scaling across horizons, serial correlation, long-memory). Together, they justify using a forecasting/tilt approach.

---

### E) Rebalancing application graphs (core results)

#### Figure 5.1 — “Integrated rebalancing and portfolio tilt framework”
**What it shows:** How statistical signals become a portfolio decision.

**Team explanation:** The pipeline outputs inclusion/retention probability and weight-shift forecasts, used to tilt holdings before official index reviews.

---

#### Figure 5.2 — “Estimated retention probabilities for current KSE-30 constituents”
**What it shows:** Inclusion likelihood distribution.

**Team explanation:** Logistic regression outputs a probability-like score that separates “high-retention” stocks from “likely exclusion” stocks. This is the practical foundation for exclusion-risk ranking.

---

#### Figure 5.3 — “Feature importance ranking for inclusion prediction model”
**What it shows:** What drives inclusion.

**Team explanation:** Lagged weight and market capitalization dominate, but liquidity also matters. This aligns with how index providers consider tradability and size.

---

#### Figure 5.4 — “Scatter plot of actual vs predicted constituent weights”
**What it shows:** Regression quality.

**Team explanation:** Ridge regression matches actual weights well for large-cap names, with more variance for smaller constituents. Since weights are inherently stable, high R² is partly expected.

---

#### Figure 5.5 — “Forecasted direction and magnitude of KSE-30 weight changes”
**What it shows:** Tilt signal.

**Team explanation:** Stocks predicted to gain weight are candidates to buy/overweight; predicted losers become candidates to underweight/exit. This is the key “action” output.

---

#### Figure 6.5 — “KSE-30 rebalancing risk and opportunity map”
**What it shows:** Risk/return opportunity framing.

**Team explanation:** Combines exclusion risk and weight-change expectations into a decision map. Emphasize the strategy is decision-support/probabilistic (not a live trading system in the report).

---

## 4) How to talk about results without deep stats
Use these presentation-ready “message frames”:

1. **Fund flows are forecastable in direction**
   - Exact PKR magnitudes are noisy (negative R²).
   - Directional accuracy improves significantly vs naive baseline.

2. **Volatility is persistent and asymmetric**
   - EGARCH indicates leverage: negative shocks increase volatility more.
   - VaR backtest supports calibration.

3. **Efficiency is mixed**
   - No single test fully rejects the random-walk hypothesis.
   - Long-memory/persistence evidence motivates forecasting/tilt.

4. **Rebalancing is the main practical output**
   - Retention probability + expected weight change enable index-tilt decisions.
   - Classification (Logistic Regression) is strong on AUC; regression (Ridge) captures weight stability.

---

## 5) Where graphs come from (so team can locate them)
For quick slide building, the project keeps a presentation-ready figure set under:

- `overview/figures/`
  - `eda/` (E01–E07)
  - `fund_flow/` (FF01–FF02)
  - `garch/` (G01–G02)
  - `efficiency/` (EF01–EF02)
  - `rebalancing/` (R01–R04)
  - `summary/` (SUMMARY_dashboard)

The PDF report also references the key figures by number (Figures 3.1 through 6.5), and the explanations above are aligned to those report sections.

---

## 6) Final “what we delivered” statement (closing slide)
We delivered a complete quantitative pipeline that:

- builds an **aggregate KSE-30 fund-flow proxy** from AKD/NBP/NTI
- models **KSE-30 volatility** using GARCH-family methods and validates via VaR backtesting
- tests **market efficiency** with multiple statistical lenses (mixed evidence)
- outputs a **rebalancing/tilt decision-support framework** using probabilistic retention and predicted weight changes
- validates the approach using a later-cycle evaluation (March 2026)

This transforms research findings into practical decision signals for index-tracking and portfolio tilting.

