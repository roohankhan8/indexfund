# A Hybrid Model of Machine Learning Model and Econometrics' Model to Predict Volatility of KSE-100 Index (2022)

## Paper Details
- Authors: Komal Batool, Mirza Faizan Ahmed, Muhammad Ali Ismail
- Journal: Reviews of Management Sciences
- Volume/Issue: Vol. 4, No. 1
- Publication window: January-June 2022
- DOI: 10.53909/rms.04.01.0125
- Market focus: Pakistan Stock Exchange (KSE-100)

## Objective
The paper forecasts KSE-100 volatility by comparing econometric and machine-learning approaches and by constructing three hybrid variants to test whether combined modeling improves predictive accuracy.

## Methodology Overview
1. Use daily KSE-100 closing prices and transform to log returns.
2. Estimate volatility proxy from ARMA residual process.
3. Train and evaluate five models:
- GARCH
- NNAR
- NNAR-based GARCH
- GARCH-based NNAR
- Linear combination of GARCH and NNAR
4. Compare models by RMSE over multiple forecast horizons.

## Model Used
### Base Models
- ARMA(1,9) for return dynamics / residual extraction
- GARCH(5,4) selected via AIC
- NNAR with non-seasonal lag structure for autoregressive volatility sequence

### Hybrid Models
- Linear combination of GARCH + NNAR outputs
- NNAR-based GARCH
- GARCH-based NNAR

## Data Used
- Type: Daily KSE-100 closing values
- Span: January 2000 to July 2019
- Train set: 2000 to 2018
- Test set: January 2019 to July 2019
- Volatility construction: derived from log returns and ARMA-estimated error series

## Where the Data Came From
- Source explicitly stated in paper: Yahoo Finance

## Key Findings
- Best overall model: Linear combination of GARCH and NNAR (lowest RMSE).
- Reported ranking (best to worse):
1. Linear combination of GARCH+NNAR
2. NNAR
3. GARCH
4. NNAR-based GARCH
5. GARCH-based NNAR
- Reported RMSE table supports superior robustness of the linear hybrid across horizons.

## Limitations
- Single-index scope (KSE-100) restricts generalization.
- Relatively short out-of-sample test segment.
- Volatility proxy is model-derived and not cross-validated against alternative realized-volatility definitions.
- GARCH assumptions and horizon behavior may weaken long-horizon performance under asymmetric shocks.

## Research Gap
The paper addresses a specific gap for PSX literature by implementing and comparing three NNAR-GARCH hybrid constructions for KSE-100 volatility, an area previously underexplored in the cited literature.

## Practical Relevance
For PSX volatility forecasting workflows, the paper supports combining econometric and neural-autoregressive signals, with linear fusion showing stronger practical forecasting consistency than single-model baselines.
