# Concepts & Theories Explainer (Report + PPT)

This file explains the main **finance theories, econometric/statistical concepts, and ML modeling ideas** referenced across:
- `FYP REPORT FINAL.pdf`
- `Analyzing Fundflow patterns - Presentation.pdf.pdf`

It is written for team members who need conceptual understanding (no code required).

---

## 1) Core Concepts: Fund Flows and Investor Behavior

### 1.1 Fund flows
**Definition:** Net movement of money into or out of an investment fund over a period.

- **Positive flow (inflow):** New subscriptions exceed redemptions → investors show confidence/demand.
- **Negative flow (outflow):** Redemptions exceed subscriptions → investors reduce exposure / show risk aversion.

**Why it matters in a market:**
- Fund flows change liquidity demand/supply.
- Large inflows/outflows can influence price pressure (especially in tracking funds that hold index constituents).

### 1.2 NAV and AUM relationship
- **NAV (Net Asset Value):** “Price” of the fund unit; changes reflect fund return.
- **AUM (Assets Under Management):** Total market value of assets held by the fund.

**Flow intuition from NAV & AUM:**
If AUM rises more than can be explained by NAV return, money likely entered the fund (net inflow). If AUM falls more than explained by NAV return, money likely exited (net outflow).

---

## 2) Market Efficiency Theory

### 2.1 Efficient Market Hypothesis (EMH)
**Core idea:** Prices fully and quickly reflect information. If true:
- it’s hard to consistently earn abnormal profits using publicly available information.

**Forms:**
- **Weak-form:** Past prices/volume are reflected in current prices.
- **Semi-strong:** All public information is reflected.
- **Strong-form:** Even private information is reflected.

### 2.2 Weak-form efficiency and random walk
In weak-form EMH, returns behave like a **random walk**:
- Future returns are not predictable from past returns/signs/variances.

### 2.3 Why “efficiency can be mixed” in emerging markets
Emerging markets can exhibit:
- liquidity constraints
- information asymmetry
- behavioral bias / delayed reaction

So instead of perfect randomness, you can see:
- persistence (memory)
- serial correlation at certain lags
- conditional volatility clustering

That motivates using multiple statistical tests.

---

## 3) Stationarity and Time-Series Transformations

### 3.1 Stationarity
**Definition:** Statistical properties of a series (mean/variance) do not change over time.

Many models assume stationarity. Non-stationary series can create spurious results.

### 3.2 ADF and KPSS
- **ADF (Augmented Dickey–Fuller):** tests for a **unit root** (non-stationary). 
  - If p-value < threshold → likely stationary.
- **KPSS (Kwiatkowski–Phillips–Schmidt–Shin):** tests stationarity around a mean.
  - If p-value > threshold → likely non-stationary.

### 3.3 Price levels vs returns (log returns)
- **Price levels:** typically trend → non-stationary.
- **Returns/log returns:** often more stable → closer to stationary.

That’s why the pipeline uses log returns for ARIMA/GARCH/efficiency diagnostics.

---

## 4) Econometric Models for Forecasting

### 4.1 ARIMA
**ARIMA** = Autoregressive + Integrated (differencing) + Moving Average.

- Captures linear patterns and autocorrelation in time series.

### 4.2 ARIMAX
**ARIMAX** is ARIMA with **exogenous variables** (macro variables like interest rate, oil return, exchange rate, CPI).

- Goal: explain/forecast flow dynamics using lagged relationships between flow and macro.

### 4.3 VAR (Vector AutoRegression)
**VAR** models multiple time series jointly, where each series is predicted by its own lags and the lags of other series.

Why it’s used:
- allows interdependencies between flow and macro variables
- avoids forcing all influence through one isolated factor

---

## 5) Volatility Modeling: ARCH/GARCH Family

### 5.1 Volatility clustering
Financial markets often show:
- big moves cluster together
- calm periods cluster together

This means variance is not constant.

### 5.2 ARCH and GARCH
- **ARCH:** current volatility depends on past squared errors.
- **GARCH(1,1):** most common form:
  - Conditional variance = constant + alpha * yesterday’s shock^2 + beta * yesterday’s variance.

### 5.3 Interpreting GARCH parameters
- **alpha (α):** sensitivity to recent shocks.
- **beta (β):** persistence of volatility.
- **alpha + beta:** how long volatility shocks last.
  - close to 1 → shocks decay slowly (high persistence).

### 5.4 EGARCH (asymmetric volatility / leverage effect)
**EGARCH** models log variance and can capture asymmetry:
- **leverage effect:** negative news increases volatility more than positive news of the same size.

In the report/ppt, the key statement is:
- EGARCH had significant leverage (negative gamma term).

---

## 6) VaR Backtesting

### 6.1 Value at Risk (VaR)
**VaR at 5%:** an estimate of the loss threshold such that losses worse than VaR should occur about 5% of the time under the model.

### 6.2 Backtesting logic
If the model is calibrated:
- about 5% of observed outcomes should breach the VaR threshold.

In your results, the backtest exceedance rate is close to nominal, supporting that the volatility model is reasonably calibrated.

---

## 7) Market Efficiency Tests Used in the Project

Your project uses several complementary tests because each test measures a different aspect of efficiency.

### 7.1 Runs test
**Purpose:** checks whether the signs of returns are random.
- If return signs show non-random clustering, it contradicts random walk behavior.

### 7.2 Variance Ratio (VR) test
**Purpose:** compares multi-period variance to what’s expected under a random walk.

- VR(k) ≈ 1 supports random walk.
- VR(k) > 1 suggests momentum behavior.
- VR(k) < 1 suggests mean reversion.

### 7.3 Ljung–Box Q test
**Purpose:** tests whether there is serial correlation in returns.
- If p-value is small → reject “no autocorrelation”.

### 7.4 Hurst exponent
**Purpose:** measures long-memory / persistence.
- H ≈ 0.5 → random walk
- H > 0.5 → persistence (trend-like memory)
- H < 0.5 → mean reversion

### 7.5 Why tests conflict (mixed efficiency)
- A market can look random under some tests/horizons but show dependence under others.
- Hence “mixed” efficiency is a correct academic framing.

---

## 8) Machine Learning Concepts Used in Rebalancing

### 8.1 Logistic Regression (classification)
**Task:** predict probability that a stock will be **retained/included** after a recomposition.

- Outputs a score interpretable as a probability-like value.

### 8.2 Ridge Regression (regression with L2 regularization)
**Task:** predict constituent weights.

- Ridge reduces overfitting by shrinking coefficients.
- Particularly useful when features are correlated or high-dimensional.

### 8.3 Random Forest (ensemble learning)
**Task:** used as a competitor model for both regression and classification.

- Built from many decision trees.
- Typically robust to non-linearities.

### 8.4 Why AUC matters when accuracy is high
If most stocks are retained most of the time, a naive classifier can achieve high **accuracy** by predicting “retain” always.

AUC (Area Under ROC Curve) is better because it measures ranking quality:
- how well the model separates likely retained vs likely excluded stocks.

---

## 9) Portfolio Tilt & Rebalancing Framework

### 9.1 Inclusion/Exclusion Risk ranking
- **Retention probability** estimates “stays in the index”.
- **Exclusion risk** can be computed as 1 − retention probability.

### 9.2 Weight change prediction
Using weight forecasts to estimate how portfolio weights might drift at next rebalance.

### 9.3 Integrated strategy logic
The overall decision-support strategy combines:
1) **Directional flow signals** (ARIMAX/VAR)
2) **Risk scaling** from EGARCH volatility and VaR calibration
3) **Rebalancing signals** (retention probability, predicted weight changes)

### 9.4 Important realism disclaimer (as stated in report)
Your report emphasizes:
- no transaction cost model
- not a fully validated live trading backtest

So the strategy is best described as **decision support / probabilistic tilt**, not a ready-to-deploy trading algorithm.

---

## 10) “Proxy aggregate sector flow” concept (important for presentation)
In your project, the “flow” variable is not directly observed as one official KSE-30 flow series.

Instead:
- it is derived as an aggregate sector flow proxy from fund-level NAV and AUM dynamics of AKD + NBP + NTI.

This is crucial to keep the research framing consistent.

---

## Quick Summary (for team speaking)
- **Flows** capture investor behavior.
- **Efficiency** tells whether predictability exists.
- **Stationarity** ensures time-series models are applied correctly.
- **ARIMAX/VAR** forecast flow direction using lagged joint dynamics.
- **GARCH/EGARCH** model volatility clustering and leverage.
- **VaR backtest** validates risk calibration.
- **Runs/VR/Ljung–Box/Hurst** diagnose mixed efficiency.
- **Logistic/Ridge/RandomForest** translate predictions into rebalancing decisions.

---

## Extended Concept Notes (More Detail for Team)

### A) Why “directional accuracy” can be more meaningful than point prediction (flow R² can be negative)
Financial flows are noisy and jumpy. If you predict exact PKR amounts month-by-month, you often get poor R² even if the *sign* is correct.
- **Directional accuracy** (inflow vs outflow) tests whether the model gets the *regime* right.
- A **negative R²** means the model’s point forecast is worse than a simple baseline (or not much better than the mean), but that does not invalidate the sign-based signal.

Team framing:
- “We treat flow models as regime/direction tools, not precise magnitude estimators.”

### B) ARIMAX / VAR intuition with lags
Econometric models use the idea: today’s outcome is influenced by its past.
- **ARIMAX**: flow depends on its own past (ARIMA part) plus macro variables (X).
- **VAR**: flow and macro evolve together; each variable can help predict the other through lag structure.

This is consistent with your report’s observation:
- contemporaneous correlation can be weak, but lagged dynamics can still carry predictive information.

### C) What the “persistence” parameter in GARCH means practically
In GARCH, **α** controls sensitivity to shocks, **β** controls how much of yesterday’s variance remains.
- If α + β is close to 1, volatility shocks are long-lasting.
- Practically: after a shock event, risk remains elevated for many days rather than reverting immediately.

### D) Leverage effect (why EGARCH’s gamma matters)
Leverage effect is the empirically observed asymmetry:
- negative return shocks increase future volatility more than positive shocks of same magnitude.

In risk management terms:
- downside moves are “more dangerous” for future risk than upside moves.

### E) VaR backtest interpretation (why 58/1300 is “good enough”)
VaR backtesting checks whether tail events occur at the expected frequency.
- For a nominal **5% VaR**, you expect breaches roughly 5% of the time.
- Your observed breach rate (~4.46%) is close, suggesting the model’s risk envelope is plausibly calibrated.

Important nuance:
- This supports calibration; it does not guarantee profit opportunity.

### F) Efficiency tests—how to explain them simply
Your project uses multiple tests. Each test probes a different “randomness” property:
- **Runs test**: are signs randomly ordered?
- **Variance ratio**: does variance scale linearly with horizon like a random walk?
- **Ljung–Box**: is there autocorrelation in returns?
- **Hurst**: is there long memory?

Mixed results are expected in emerging markets.

### G) Rebalancing tasks—why classification vs regression
- **Classification (Logistic Regression)** answers: “Will the stock stay or be excluded?” This produces a **retention probability**.
- **Regression (Ridge)** answers: “What weight will it have if it stays / in the new composition?”

Team framing:
- Probabilities are most actionable for ranking “at-risk” stocks.

### H) AUC vs accuracy with class imbalance
If most stocks are retained most of the time:
- A naive “always retain” classifier may produce high accuracy.
- **AUC** is more informative because it evaluates ranking quality across thresholds.

Team framing:
- “We use AUC to ensure the model identifies excluded-risk stocks, not just the majority class.”


