# Presentation Concepts, Theories, Symbols, and Models Explainer

This file explains the concepts, theories, formulas, symbols, models, and evaluation terms used across:

- `Analyzing Fundflow patterns - Presentation.pdf.pdf`
- `FYP REPORT FINAL (1).docx`

It is written as a speaking and revision companion for the viva/presentation.

## 1. Big Picture of the Project

The project combines three layers of analysis:

1. **Fund-flow analysis**
   To understand how money moves into and out of KSE-30-related funds.
2. **Market efficiency and volatility analysis**
   To test whether the KSE-30 behaves like a random walk and to model its risk dynamics.
3. **Portfolio rebalancing prediction**
   To predict which stocks are likely to remain in the KSE-30 and what their future weights may be.

The practical logic is:

- If flows contain directional information,
- and market behavior is not perfectly random,
- and future rebalancing changes can be estimated,
- then investors can build a probabilistic decision-support framework.

## 2. Core Finance Concepts

### 2.1 Fund Flow

**Fund flow** means net money entering or leaving a fund over a period.

- Positive flow = net inflow
- Negative flow = net outflow

Interpretation:

- Inflows usually suggest stronger investor demand or confidence.
- Outflows usually suggest redemptions, caution, or risk aversion.

### 2.2 NAV

**NAV** means **Net Asset Value**.

It is the value per unit of the mutual fund and changes with the value of the underlying holdings.

### 2.3 AUM

**AUM** means **Assets Under Management**.

It is the total market value of assets managed by the fund.

### 2.4 Why NAV and AUM are used to derive flows

If AUM changes, that change can happen for two reasons:

- the fund's holdings changed in value because NAV changed
- investors added or withdrew money

So the report uses a flow identity to isolate the investor-money component.

### 2.5 Fund Flow Identity

From the report:

`flow(t) = AUM(t) - AUM(t-1) × [ NAV(t) / NAV(t-1) ]`

Meaning:

- `AUM(t)` = current period assets under management
- `AUM(t-1)` = previous period AUM
- `NAV(t) / NAV(t-1)` = growth in fund value due to market performance

Interpretation:

- If actual AUM is higher than the market-performance-adjusted AUM, the difference is net inflow.
- If actual AUM is lower, the difference is net outflow.

### 2.6 Aggregate Sector Fund Flow

The project tracks three funds: **AKD, NBP, and NTI**.

Their flows are aggregated as:

`total_fund_flow(t) = flow_AKD(t) + flow_NBP(t) + flow_NTI(t)`

So the study does not use one official observed "KSE-30 fund flow" series. It builds a **proxy aggregate sector flow** from these three tracked funds.

### 2.7 Flow Normalization

From the report:

`flow_pct_sector(t) = total_fund_flow(t) / sector_aum(t-1)`

This expresses flow relative to sector size, which makes comparison easier across periods.

## 3. Market Efficiency Theory

### 3.1 Efficient Market Hypothesis

The central theory behind the efficiency part is the **Efficient Market Hypothesis (EMH)**.

It says that prices reflect available information quickly.

If that is fully true, it becomes very difficult to earn abnormal returns using public information.

### 3.2 Weak-form efficiency

The report mainly deals with **weak-form efficiency**.

Weak-form efficiency means:

- past price and return information is already incorporated in current prices
- future returns should not be predictably derived from past return patterns alone

### 3.3 Random Walk

A **random walk** means future price changes are not systematically predictable from past price changes.

In a strict weak-form efficient market:

- return signs should look random
- serial correlation should be absent
- variance should scale in a random-walk-like way
- long memory should not exist

### 3.4 Why mixed efficiency matters here

Your findings do not support a simple "efficient" or "inefficient" label.

Instead, KSE-30 appears **mixed**:

- some tests support random-walk behavior over short horizons
- other tests show persistence and serial dependence

That is an academically strong conclusion, especially for an emerging market.

## 4. Time-Series Concepts

### 4.1 Stationarity

**Stationarity** means that the statistical properties of a series, such as mean and variance, stay broadly stable over time.

Why it matters:

- many econometric models assume stationarity
- non-stationary data can produce misleading or spurious results

### 4.2 Differencing

**Differencing** is a transformation used to remove trends and make a series more stationary.

If a variable has a unit root or trend, first differencing is often applied.

### 4.3 Log Return

From the report:

`idx_log_return(t) = ln[ idx_ff_mcap_total(t) / idx_ff_mcap_total(t-1) ]`

Meaning:

- `ln` is the natural logarithm
- the formula measures continuously compounded return

Why log returns are used:

- they are usually more stable than price levels
- they are standard in finance
- they work well in volatility models

### 4.4 ADF Test

**ADF** means **Augmented Dickey-Fuller** test.

Purpose:

- to test for a unit root
- to assess whether a time series is non-stationary

In plain language:

- low p-value generally supports stationarity

### 4.5 KPSS Test

**KPSS** means **Kwiatkowski-Phillips-Schmidt-Shin** test.

It complements ADF.

Why both are useful:

- ADF and KPSS test stationarity from opposite angles
- using both gives stronger evidence for whether transformations are appropriate

## 5. Forecasting Models in the Project

### 5.1 Naive Random Walk Benchmark

The **naive** model assumes the next value is essentially the same as the current value.

Why it matters:

- it is the baseline
- if a more advanced model cannot beat the naive benchmark, it is not very useful

### 5.2 ARIMA

**ARIMA** means **Autoregressive Integrated Moving Average**.

It combines:

- autoregressive behavior from past values
- differencing for non-stationarity
- moving-average behavior from past errors

### 5.3 ARIMAX

**ARIMAX** is ARIMA with **exogenous variables**.

In your project, ARIMAX uses lagged macro-financial information in addition to the flow series itself.

Slide result:

- `ARIMAX(1,0,1)`
- directional accuracy = `75.0%`
- RMSE = `58.56`
- MAE = `37.35`

Why it matters:

- it captures lagged structure better than a simple benchmark
- it works better as a direction/regime model than as an exact value model

### 5.4 VAR

**VAR** means **Vector Autoregression**.

It models multiple time series jointly so each variable can depend on its own lags and the lags of other variables.

Why it fits this project:

- fund flows, macro variables, and index measures may interact jointly
- no single macro variable was strong enough alone, so a system approach is reasonable

Slide result:

- `VAR(1)`
- directional accuracy = `75.0%`
- RMSE = `63.10`
- MAE = `39.41`

### 5.5 Granger Causality

**Granger causality** does not mean true philosophical causation.

It means:

- if past values of variable X improve prediction of variable Y, then X is said to Granger-cause Y

Your report states:

- no single macroeconomic variable passes the 5% significance threshold on its own
- CPI comes close with `p = 0.0639`
- oil, interest rate, and exchange-rate changes are not individually significant

Interpretation:

- isolated one-variable macro signals are weak
- predictive value likely comes from lagged interactions and combined dynamics

### 5.6 Directional Accuracy

This is one of the most important metrics in your flow section.

It asks:

- did the model correctly predict the sign of movement?
- inflow vs outflow

Why it matters more here than exact magnitude:

- monthly flows are noisy and shock-driven
- a model can be poor at exact PKR amounts but still useful at direction

This is why your presentation says:

- the models are stronger as **directional tools** than as **point estimators**

### 5.7 Why R-squared can be negative here

`R²` compares model fit with a simple baseline.

A negative `R²` means:

- the model is worse than the baseline at matching exact magnitudes

That sounds bad, but in this context it does not destroy the usefulness of the model because:

- the real value is in directional classification, not precise PKR forecasting

## 6. Volatility and Risk Models

### 6.1 Volatility Clustering

A classic financial fact is **volatility clustering**:

- high-volatility periods tend to be followed by high-volatility periods
- calm periods tend to be followed by calm periods

This means variance is not constant over time.

### 6.2 GARCH

**GARCH** means **Generalized Autoregressive Conditional Heteroskedasticity**.

In your project the main specification is **GARCH(1,1)**.

Interpretation of symbols:

- `ω` or omega = constant term in the variance equation
- `α` or alpha = impact of recent shocks
- `β` or beta = persistence from previous volatility

From the slide:

- `ω = 0.0749`
- `α = 0.1284`
- `β = 0.8383`
- persistence = `α + β = 0.9667`

### 6.3 Volatility Persistence

Persistence is:

`α + β`

When this is close to 1, shocks decay slowly.

Your result:

- `0.9667`

Meaning:

- volatility remains elevated after shocks
- risk is sticky, not short-lived

### 6.4 EGARCH

**EGARCH** means **Exponential GARCH**.

Why it is useful:

- it models log variance
- it can capture asymmetry in volatility response

### 6.5 Gamma and Leverage Effect

In EGARCH, `γ` or **gamma** measures asymmetry.

Your report says the gamma term is significantly negative.

Interpretation:

- negative news creates a larger jump in future volatility than positive news of equal size

This is the **leverage effect**.

### 6.6 AIC

**AIC** means **Akaike Information Criterion**.

Used for model comparison:

- lower AIC suggests a better balance of fit and parsimony

Your result:

- GARCH AIC = `4243.43`
- EGARCH AIC = `4209.68`

So EGARCH is preferred.

### 6.7 VaR

**VaR** means **Value at Risk**.

At 5% VaR, the model estimates a downside loss threshold such that losses worse than that threshold should happen around 5% of the time.

### 6.8 VaR Backtesting

Backtesting checks whether actual breaches occur about as often as the model predicts.

Your result:

- `58 exceedances out of 1,300`
- breach rate = `4.46%`
- expected rate = `5%`

Interpretation:

- the model is reasonably well calibrated
- it provides a plausible downside-risk envelope

## 7. Market Efficiency Tests Used

### 7.1 Runs Test

The **Runs Test** checks whether positive and negative returns occur in a random sequence.

Your result:

- `Z = -1.91`
- `p = 0.0561`

Interpretation:

- borderline result
- does not strongly reject random ordering of return signs

### 7.2 Variance Ratio Test

The **Variance Ratio (VR) Test** checks whether return variance scales with horizon in a way consistent with a random walk.

Interpretation:

- `VR ≈ 1` supports random walk
- `VR > 1` can suggest momentum
- `VR < 1` can suggest mean reversion

Your slide reports:

- `VR(2) = 1.0069`, `p = 0.8752`

That is broadly consistent with efficiency over that short horizon.

### 7.3 Ljung-Box Q Test

The **Ljung-Box Q Test** checks for serial correlation across multiple lags.

Your result:

- `p = 0.0000`

Interpretation:

- reject no-autocorrelation
- returns contain statistically significant dependence

### 7.4 Hurst Exponent

The **Hurst exponent**, written as `H`, measures long-memory behavior.

Interpretation:

- `H = 0.5` suggests random walk
- `H > 0.5` suggests persistence
- `H < 0.5` suggests mean reversion

Your result:

- `H = 0.6559`

Meaning:

- KSE-30 shows persistent long-memory behavior

### 7.5 Mixed Efficiency

Putting all tests together:

- Runs and variance ratio lean toward short-horizon efficiency
- Ljung-Box and Hurst point toward dependence and persistence

So your academically correct conclusion is:

- **KSE-30 is neither fully efficient nor fully inefficient**

## 8. Portfolio Rebalancing Framework

### 8.1 Rebalancing

**Portfolio rebalancing** means adjusting holdings when index composition or constituent weights change.

In your project, the framework anticipates future KSE-30 changes before the official review.

### 8.2 Two prediction tasks

The framework has two tasks:

1. **Inclusion prediction**
   Will the stock remain in the KSE-30 after the next recomposition?
2. **Weight prediction**
   What will the stock's constituent weight be?

### 8.3 Logistic Regression

Used for inclusion prediction.

It outputs a probability-like score for class membership.

In your context:

- a higher score means a higher retention probability

### 8.4 Retention Probability

**Retention probability** means the estimated probability that a stock stays in the index.

This is one of the most decision-useful outputs in the project.

### 8.5 Exclusion Risk

The report defines:

`Exclusion Risk = 1 - Average Retention Probability`

Interpretation:

- higher exclusion risk means the stock is more vulnerable to removal

### 8.6 Ridge Regression

Used for weight prediction.

Ridge is a linear regression model with **L2 regularization**, which shrinks coefficients and helps when predictors are correlated.

Why it worked well here:

- index weights are very stable
- a simple regularized model is enough to slightly improve on the naive benchmark

### 8.7 Random Forest

**Random Forest** is an ensemble model made of many decision trees.

It is useful for nonlinear relationships, but in your results it is not the preferred final model for the main rebalancing tasks.

### 8.8 AUC

**AUC** means **Area Under the ROC Curve**.

It measures how well the classifier separates retained from excluded stocks across thresholds.

This matters because your report identifies **class imbalance**.

### 8.9 Class Imbalance

Class imbalance means one class is much more common than the other.

Here:

- most stocks are retained

So a model can get high accuracy by simply predicting "retain" most of the time.

That is why:

- accuracy alone is misleading
- AUC is more informative

### 8.10 Rebalancing Results

From the slides/report:

- all inclusion models show `96.55%` accuracy
- Logistic Regression has `AUC = 0.8214`
- Random Forest has `AUC = 0.5446`

Interpretation:

- Logistic Regression separates at-risk stocks much better
- Random Forest likely overfit the majority class

For weight prediction:

- Naive: `RMSE = 0.5595`, `R² = 0.9677`
- Ridge: `RMSE = 0.5590`, `R² = 0.9678`
- Random Forest: `RMSE = 0.6900`, `R² = 0.9509`

Interpretation:

- weights are highly stable
- even the naive baseline is strong
- Ridge is preferred because it slightly improves fit without unnecessary complexity

## 9. Important Variables and Features from the Report

### 9.1 Monthly NAV Return

From the report:

`nav_return_m = ( NAV_end / NAV_start ) - 1`

### 9.2 Rolling Volatility

From the report:

`rolling_vol_30d(t) = std(log returns over past 30 days) × √252`

Interpretation:

- `std` = standard deviation
- `√252` annualizes volatility using roughly 252 trading days

### 9.3 Rebalancing Features

The report lists these as important predictive features:

- `cur_weight` = current constituent weight
- `mom_30`, `mom_60`, `mom_90` = 30/60/90-day momentum
- `vol_30`, `vol_60` = realized volatility
- `avg_volume` = average recent trading volume
- `mkt_cap_proxy` = market-cap estimate
- `price_to_ma20`, `ma50` = price relative to moving averages
- `wt_drift` = weight change since last recomposition
- `wt_range` = historical range of constituent weight

### 9.4 Momentum

Momentum generally means recent price strength or weakness.

For example:

`momentum = (Price_t / Price_{t-N}) - 1`

Negative momentum often signals weakening relative position.

### 9.5 Weight Drift

Weight drift means how much a stock's index weight has moved since the last recomposition.

This can signal whether it is gaining or losing structural importance in the index.

## 10. Statistical and Evaluation Terms

### 10.1 RMSE

**RMSE** = Root Mean Squared Error

It penalizes larger forecast errors more heavily.

Lower is better.

### 10.2 MAE

**MAE** = Mean Absolute Error

It measures average absolute forecast error.

Lower is better.

### 10.3 R-squared

`R²` measures the proportion of variation explained by the model.

Higher is generally better, but interpretation depends on the task and baseline.

### 10.4 p-value

A **p-value** helps assess statistical significance.

Rough practical rule:

- below `0.05` is often treated as statistically significant

### 10.5 Z-statistic

A **Z-statistic** standardizes how far a result is from its null expectation.

In your slides it appears in the Runs Test.

## 11. Symbols and Notation Cheat Sheet

- `t` = current time period
- `t-1` = previous time period
- `ln` = natural logarithm
- `ω` = omega, constant term in GARCH variance equation
- `α` = alpha, shock sensitivity
- `β` = beta, volatility persistence
- `γ` = gamma, asymmetry/leverage parameter in EGARCH
- `H` = Hurst exponent
- `VR` = Variance Ratio
- `Q` = Ljung-Box Q statistic
- `√252` = annualization factor for daily volatility
- `p` = p-value
- `AIC` = Akaike Information Criterion
- `AUC` = Area Under the ROC Curve

## 12. How the Slides Fit Together Conceptually

Slides 13 to 18 follow a clear logic:

1. **Slide 13**
   Fund flows are forecastable in direction.
2. **Slide 14**
   The flow signal is weak for exact PKR values but useful for regimes.
3. **Slide 15**
   Market volatility is persistent and asymmetric.
4. **Slide 16**
   VaR backtesting shows the volatility model is usable for risk control.
5. **Slide 17**
   Market efficiency is mixed, so exploitable structure may exist.
6. **Slide 18**
   Those findings are turned into a stock-level rebalancing framework.

This chain is important in the viva because it shows the project is not a collection of separate techniques. It is one connected argument.

## 13. Safest Viva Language to Use

These are the safest ways to describe the work academically:

- "Our models are more useful as directional tools than exact point estimators."
- "The market shows mixed weak-form efficiency rather than pure efficiency or pure inefficiency."
- "EGARCH is preferred because it captures both persistence and asymmetric downside risk."
- "The rebalancing framework is a probabilistic decision-support system, not a fully validated live trading strategy."
- "AUC is more informative than accuracy in the inclusion task because of class imbalance."

## 14. Final One-Paragraph Summary

The project studies how fund flows, volatility, market efficiency, and index rebalancing interact in the Pakistani market. It derives aggregate KSE-30-related fund flows from NAV and AUM data, uses ARIMAX and VAR to forecast flow direction, applies GARCH and EGARCH to model persistent and asymmetric volatility, validates downside risk through VaR backtesting, and tests weak-form efficiency through Runs, Variance Ratio, Ljung-Box, and Hurst diagnostics. Since the evidence suggests directionally useful flow signals, persistent downside-sensitive risk, and mixed market efficiency, the project then extends these findings into a practical rebalancing framework using Logistic Regression for retention probability and Ridge Regression for constituent weight prediction.
