# Predicting Stock Market Trends Based on the Fluctuations in South Asian Stock Markets: Evidence from Pakistan Stock Exchange (2025)

## Overview
This study examines how regional South Asian stock markets influence Pakistan's stock market performance, using the KSE-100 Index as the main target variable. It combines causal inference via regression with time-series forecasting via ARIMA.

## Model Used
- Multiple Regression Analysis to estimate the effect of regional indices on KSE-100.
- ARIMA forecasting for KSE-100 trend prediction (reported model selection includes ARIMA(0,1,0)).
- Time-series diagnostics discussed, including residual checks and Ljung-Box style validation references.

## Data Used
- Historical time-series data from July 2014 to June 2024.
- Dependent variable:
  - KSE-100 Index (Pakistan Stock Exchange).
- Independent variables:
  - Nifty 50 (India).
  - DSEX (Bangladesh).
  - NEPSE (Nepal).
  - MASIX (Maldives).

## Where the Data Came From
- Pakistan Stock Exchange index series (KSE-100 context in the paper).
- Official benchmark index series from neighboring South Asian stock exchanges:
  - National Stock Exchange of India (Nifty 50),
  - Dhaka Stock Exchange (DSEX),
  - Nepal Stock Exchange (NEPSE),
  - Maldives Stock Exchange index (MASIX).

## Key Findings
- Pakistan's stock market is significantly influenced by regional South Asian markets.
- Nifty 50 has the strongest association with KSE-100 among the regional indices studied.
- DSEX, NEPSE, and MASIX also show positive associations, but with smaller magnitudes.
- ARIMA is reported as suitable for forecasting KSE-100 trend dynamics in this setup.

## Limitations
- Regional scope is limited to four neighboring market indices; other international spillovers are excluded.
- Purely time-series/econometric setup may not fully capture structural breaks, regime shifts, or sudden geopolitical shocks.
- Forecast uncertainty rises over longer horizons; model reliability depends on continuity of historical relationships.

## Research Gap
- Need broader multi-market integration including more global indices and cross-asset channels.
- Need hybrid approaches combining econometric models with ML or nonlinear regime-switching frameworks.
- Need explicit structural-break modeling and rolling re-estimation for higher robustness.

## Practical Relevance
- Provides policymakers and investors a regional-contagion lens for PSX risk monitoring.
- Supports incorporating neighboring market signals into short-term KSE-100 forecasting and portfolio decisions.
