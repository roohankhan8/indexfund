# Presentation Talk Script (Slide-by-Slide)

**Time:** ~10 minutes (can be shortened)

**Speaker note:** Use natural pacing; emphasize the *directional accuracy* and *exclusion-risk / retention probability* as the most practical outcomes.

---

## Slide 1 — Title
Good [morning/afternoon]. We are Roohan Khan, Aina Batool, Maida Murtaza, and Laiba Sarfaraz. Our project is “Analyzing Fund Flow Patterns and Market Efficiency in Pakistan with Predictive Models.”

Our advisor is Ms. Ubaida Fatima. Today we’ll explain the problem, the methodology pipeline, and the key results—especially how we turn forecasting into a portfolio rebalancing decision-support framework.

---

## Slide 2 — Table of Contents
We’ll cover:
1) Introduction and problem statement
2) Methodology and model building
3) Results
4) Portfolio rebalancing
5) Conclusion and recommendations

---

## Slide 3 — Introduction to Topic / Significance & Motivation
Pakistan’s mutual fund and asset management industry is large—Assets Under Management are roughly **Rs. 4.48 trillion**. Yet, research on how fund flows relate to KSE index behavior—especially for **KSE-30**—is still sparse.

So the main motivation is: can we use observable fund-flow information to forecast what happens in the market and support better rebalancing decisions.

---

## Slide 4 — Fund Flow (What we mean)
Fund flows are the **net movement of money into or out of investment funds**.

- Positive flows typically indicate investor confidence
- Negative flows reflect risk aversion

In practice, fund flows connect to:
- NAV changes
- liquidity conditions
- and eventually portfolio rebalancing—especially for index-tracking funds.

In Pakistan, fund flows are regulated by SECP and influenced by political and macroeconomic factors.

---

## Slide 5 — Market Efficiency (Why we test it)
Market efficiency means stock prices quickly incorporate available information.

If markets are fully efficient, consistently earning abnormal returns becomes extremely difficult.

Because this project is about forecasting and rebalancing, we must test whether there is any exploitable predictability. And emerging markets can be **mixed**—sometimes close to random walk, sometimes showing persistence.

---

## Slide 6 — Problem Identification
We identified four gaps/problems:
1) Pakistan’s financial market has grown, but index fund research remains sparse
2) No integrated framework linking **fund flows + market efficiency + ML forecasting** for KSE-30
3) Most prior literature focuses on developed economies
4) This leaves investors and policymakers without localized, unified evidence

---

## Slide 7 — Scope and Objectives
Our objectives are to:
- analyze fund flow patterns in KSE-30 index funds
- forecast fund flows using econometric and ML models
- test weak-form market efficiency of KSE-30
- model volatility dynamics
- build a portfolio rebalancing application

**Data scope:** KSE-30 constituent stocks and three index-tracking funds: **AKD, NBP, NTI**. The reporting analysis window is August 2025 to June 2026 with modeling based on a longer history starting from 2020.

---

## Slide 8 — Literature Review (Key themes)
From prior research:
- Fund flows can contain predictive information about future behavior
- Market efficiency and index fund flow have a relationship: less efficiency can reduce or change how flows behave
- Pakistan-focused studies show weak-form inefficiency or mixed evidence depending on the test
- Volatility clustering and asymmetric risk responses appear in Pakistan, making GARCH/EGARCH suitable

This motivates our design: forecast flows, model volatility, test efficiency, then use rebalancing as the practical output.

---

## Slide 9 — Research Gap (Missing key factors)
The key missing pieces in many PSX studies are:
- investor sentiment, macro variables, and fund flows are rarely integrated
- developed markets dominate the evidence base
- no unified framework combines price, volatility, and flows with efficiency testing
- KSE-30 is under-studied compared to KSE-100
- insufficient model comparison across econometrics vs ML

---

## Slide 10 — Methodology Pipeline (Big picture)
Our pipeline is built as:
- Data collection & processing
- Exploratory data analysis
- Fund-flow forecasting (ARIMAX, VAR)
- Volatility modeling (GARCH, EGARCH)
- Market efficiency tests (Runs, Variance Ratio, Ljung–Box, Hurst)
- Portfolio rebalancing (Logistic Regression, Ridge, Random Forest)
- Results & recommendations

---

## Slide 11 — Data Collection (What we fed into the models)
We collect:
- **KSE-30 stocks**: daily log returns, volatility proxies, and constituent weights
- **Fund flows**: monthly aggregate fund flow in PKR mn
- **Dependent variable**: monthly aggregate fund flow / sector flow proxy
- **Independent variables**: macro variables—CPI, interest rate, exchange rate, oil price, gold
- **Fund-level inputs**: NAV and AUM from AKD, NBP, NTI (starting around Jan 2021)

---

## Slide 12 — Models & Evaluation
Econometric models:
- ARIMA, ARIMAX, VAR
- GARCH(1,1), EGARCH(1,1)

ML models:
- LSTM and Random Forest (referenced in the work)
- Ridge Regression and Logistic Regression (used in final rebalancing performance comparison)

Evaluation:
- RMSE, MAE, R²
- directional accuracy for flows
- AUC for inclusion / retention

---

## Slide 13 — Exploratory Data Analysis Results
EDA confirms major statistical properties:
- contemporaneous correlation between flows and KSE-30 returns is very small (~0.0095), suggesting the relationship is mainly in **lagged dynamics**
- fund flows are episodic with outliers (peak inflow ~234.86 mn and peak outflow ~-44.01 mn)
- AUM is highly concentrated: AKD dominates the aggregate sector AUM
- KSE-30 mean daily return ~0.0736% and std ~1.3945%
- extreme events: about -10.24% and +9.32%
- Jarque–Bera p-value ~0 → non-normal “fat tails,” justifying GARCH/EGARCH

---

## Slide 14 — Fund Flow Forecasting Results
Key results:
- Naive random walk directional accuracy: **37.5%**
- ARIMAX(1,0,1): **75.0%** directional accuracy, with improved error metrics
- VAR(1): also **75.0%** directional accuracy

The takeaway: fund flows are hard to forecast in exact magnitude, but direction can be predicted materially better than naive.

---

## Slide 15 — Volatility Modeling Results
We fit GARCH(1,1) and EGARCH(1,1).

Findings:
- High volatility persistence: alpha + beta ≈ **0.9667** → shocks decay slowly
- EGARCH preferred by AIC and shows significant negative gamma → **leverage effect**
  (bad news increases volatility more than good news of the same size)

Risk validation:
- VaR backtest at 5% shows 58 exceedances out of 1300 observations = **4.46%**, close to nominal.

---

## Slide 16 — Market Efficiency Results
Efficiency is mixed:
- Runs test: borderline efficient (p ≈ 0.0561)
- Variance ratio VR(2): supports efficiency (p ≈ 0.8752)
- Ljung–Box Q: indicates inefficiency (p ≈ 0.0000)
- Hurst exponent: H = 0.6559 → persistent / long-memory behavior

So, KSE-30 cannot be described as fully efficient or fully inefficient. That mixed result motivates forecasting/tilting.

---

## Slide 17 — Portfolio Rebalancing: Framework
We convert predictive modeling into a rebalancing framework with two prediction tasks:
1) **Inclusion prediction**: will a stock remain after the next recomposition?
2) **Weight prediction**: what will its constituent weight be?

Data: **422 stock-window observations** across **16 semi-annual rebalancing periods**.

Features: current weight, momentum (30/60/90 days), volatility, average volume, market-cap proxy, price-to-MA ratios, weight drift, and range.

---

## Slide 18 — Portfolio Rebalancing Results
Results:
- Weight prediction: **Ridge Regression** performs best vs naive; Random Forest underperforms.
- Inclusion/exclusion prediction: **Logistic Regression** has strong discrimination with **AUC ~0.8214**.

Because most stocks tend to be retained, accuracy alone can be misleading—AUC is the key metric for “at-risk” exclusion ranking.

---

## Slide 19 — Quantitative Investment Strategy
We propose an integrated decision-support strategy:

1) **Tactical rebalancing tilt**
   - overweight stocks with high retention probability
   - underweight or divest high exclusion-risk stocks

2) **Macro-flow alignment**
   - use ARIMAX directional flow signals as a regime filter

3) **Volatility-modulated execution**
   - scale position sizes using EGARCH conditional variance

Importantly: our framework is decision-support and emphasizes probabilistic risk management; it does not include a transaction-cost model or full live-trading backtest.

---

## Slide 20 — Conclusion (Key takeaways)
Overall, we conclude:
- Fund flows are directionally predictable (ARIMAX 75% vs 37.5% naive)
- Volatility is persistent and asymmetric (EGARCH + VaR validated)
- Market efficiency is mixed (long-memory exists)
- Constituents’ retention and weights can be forecast well enough to support rebalancing

---

## Slide 21 — Recommendations
Recommendations:
- improve data transparency to reduce informational friction and move toward weak-form efficiency
- use directional flow signals plus rebalancing forecasts as a decision-support layer
- continue using EGARCH-based VaR for downside risk calibration

---

## Slide 22 — Future Research Directions
Future research:
- expand monthly flow dataset and add stronger predictors (FPI, political indicators, sentiment)
- test LSTM and regime-switching models with larger samples
- run a full net-of-cost backtest including transaction costs and slippage
- extend to sector-level analysis such as banking and energy

---

## Slide 23 — References / Closing
We acknowledge the key literature that motivated the approach.

Thank you for your attention.

