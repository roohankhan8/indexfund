# Modelling Volatility of Pakistan Stock Market Using Family of GARCH Models (2025)

## Overview
This paper models volatility in Pakistan's stock market with a focus on the KSE-100 index, and also studies dependence with exchange rate and crude oil prices. The authors estimate ARMA-GARCH family models and then use copula-GARCH for dependence structure.

## Model Used
- Mean model selection with ARMA (identified as ARMA(1,1) for KSE-100 returns).
- Volatility modeling with GARCH(1,1) under multiple error distributions (normal, skew-normal, Student-t, skewed Student-t, GED, skewed GED).
- Best marginal model for KSE-100 reported as ARMA(1,1)-GARCH(1,1) with skewed Student-t errors based on AIC/BIC.
- Dependence analysis using bivariate copula-GARCH models.
- Implementation references include R (rugarch package) and auto.arima checks.

## Data Used
- Weekly series for:
  - KSE-100 index closing prices (converted to returns).
  - Exchange rate.
  - WTI crude oil prices.
- Study horizon reported as January 2000 to December 2022 (about 22 years).
- Total observations reported around 1200 for each variable.

## Where the Data Came From
- Pakistan Stock Exchange (KSE-100 series context in the paper).
- State Bank of Pakistan (exchange-rate related macro-financial data context).
- Yahoo Finance (explicitly mentioned in data description).

## Key Findings
- ARCH effects are significant, supporting conditional heteroskedasticity modeling.
- ARMA(1,1)-GARCH(1,1) with skewed Student-t is selected as best for KSE-100 volatility.
- High persistence in volatility is reported (ARCH+GARCH coefficients close to 1), implying long-lasting shock effects.
- Volatility is described as highly sensitive to market shocks.

## Limitations
- Sampling frequency is weekly only; authors note daily/monthly alternatives for future work.
- Marginal modeling centers on standard GARCH(1,1), leaving broader asymmetric/alternative structures less explored.
- Scope limited to selected variables (KSE-100, exchange rate, crude oil), so additional market drivers are omitted.

## Research Gap
- Need comparative testing across richer volatility families (e.g., EGARCH, TGARCH, APARCH) and additional GARCH orders.
- Need broader multivariate setups including more explanatory macro/financial variables.
- Potential to evaluate robustness across different time frequencies and structural-regime periods.

## Practical Relevance
- Useful for risk management and volatility-aware forecasting for PSX participants.
- Supports model-based monitoring of persistent volatility shocks in frontier-market settings.
