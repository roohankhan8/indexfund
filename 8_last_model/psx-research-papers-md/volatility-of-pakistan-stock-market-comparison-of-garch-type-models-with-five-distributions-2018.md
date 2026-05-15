# Volatility of Pakistan Stock Market: A Comparison of GARCH Type Models with Five Distributions (2018)

## Overview
This study compares symmetric and asymmetric GARCH-family volatility models for Pakistan’s stock market under multiple conditional error distributions.

## Model Used
- Symmetric volatility models:
  - GARCH(1,1)
  - GARCH-M(1,1)
- Asymmetric volatility models:
  - EGARCH
  - TGARCH
- Error distribution comparisons include:
  - Normal
  - Student-t
  - GED
  - Student-t with fixed degrees of freedom
  - GED with fixed parameters

## Data Used
- Pakistan stock market index time series (KSE-100 context in the paper).
- Study period reported as 1 January 2008 to 30 June 2018.

## Where the Data Came From
- KSE-100 Pakistan stock market historical index/price series as described in the paper.
- The article focuses on empirical modeling over the stated Pakistan market period.

## Key Findings
- Significant ARCH/GARCH persistence effects are reported across distributions.
- GARCH-M shows evidence consistent with risk-premium effect under selected distributions.
- EGARCH and TGARCH capture asymmetric/leverage behavior.
- Distribution choice materially affects volatility-model fit and forecasting adequacy.
- EGARCH with Student-t is highlighted as a strong-performing specification in reported comparisons.

## Limitations
- Study scope is bounded to selected GARCH-family forms and specified distributional assumptions.
- Findings are period-specific; model ranking may shift under different regimes.
- Exogenous macro/news variables are not explicitly integrated into the core volatility equations.

## Research Gap
- Need wider model families and hybrid volatility-ML frameworks for robustness.
- Need cross-regime and out-of-sample stability analysis across longer horizons.
- Need integration of external covariates to explain event-driven volatility jumps.

## Practical Relevance
- Helps analysts choose volatility specifications for PSX risk estimation.
- Useful for VaR, hedging, and stress-aware portfolio management in emerging markets.
