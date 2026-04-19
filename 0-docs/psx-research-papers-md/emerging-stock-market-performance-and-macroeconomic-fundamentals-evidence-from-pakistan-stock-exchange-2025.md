# Emerging Stock Market Performance and Macroeconomic Fundamentals: Evidence from Pakistan Stock Exchange (2025)

## Paper Details
- Authors: Dr. Haseeb Hassan, Abubakar Niaz, Dr. Samina Rooh, Dr. Javeria Andleeb Qureshi
- Publication: International Journal of Research in Business and Social Science context page indicates Volume 3, Issue 4, 2025 (as rendered in extracted PDF)
- DOI: 10.5281/zenodo.15552976
- Published: 30 April 2025
- Market focus: Pakistan Stock Exchange (KSE-100 index)

## Objective
The study evaluates how macroeconomic fundamentals and exchange-rate dynamics influence KSE-100 performance, and uses ARIMA forecasting to project near-term stock-market trends.

## Methodology Overview
1. Build a monthly time-series dataset.
2. Model KSE-100 against key macro variables using multiple regression.
3. Apply ARIMA to forecast future KSE-100 movement.
4. Validate residual behavior with Ljung-Box and ACF/PACF checks.

## Model Used
### Explanatory/Impact Model
- Multiple Regression Analysis
- Dependent variable: KSE-100 Index
- Independent variables: USD/PKR, CNY/PKR, EUR/PKR, FDI, BOT

### Forecasting Model
- ARIMA (0,1,0) selected by Expert Modeler for KSE-100 forecasting
- Diagnostics reported include RMSE, MAPE, MAE, Ljung-Box Q, ACF/PACF residual checks

## Data Used
- Frequency: Monthly
- Sample period: July 2014 to June 2024
- Number of observations: 120
- Variables:
- KSE-100 index
- Exchange rates: USD/PKR, CNY/PKR, EUR/PKR
- FDI (million USD)
- BOT (million USD)

## Where the Data Came From
The extracted paper text clearly states variable definitions and monthly observations but does not provide one consolidated table of source URLs in the extracted sections. Sources are implied as official market and macroeconomic series for Pakistan (PSX/KSE and macro datasets typically from official/statistical repositories used in the cited literature). If needed, the original PDF tables/appendix can be cross-checked for exact source endpoints.

## Key Findings
- Macroeconomic variables are jointly significant for KSE-100 performance:
- Reported model fit: R^2 = 0.48, F = 21.12
- Exchange-rate effects are significant, with USD/PKR and CNY/PKR showing strong influence in reported coefficients.
- BOT shows significant negative effect in the regression output (as coded in the paper's specification).
- FDI relation is discussed as positive in narrative conclusions, though coefficient significance is weak in the displayed regression table (p-value reported above common thresholds).
- ARIMA (0,1,0) provides usable forecast accuracy in the reported setup:
- RMSE: 2314.355
- MAPE: 4.164
- MAE: 1735.165
- Ljung-Box Q(18) p-value: 0.59 (residual autocorrelation not significant)

## Forecast Window in Paper
- Forecast horizon shown: July 2024 to June 2025
- Narrative indicates generally resilient upward trend with widening confidence bounds over time (higher long-horizon uncertainty).

## Limitations
- Moderate explanatory power (R^2 = 0.48) indicates substantial variance remains outside included predictors.
- Forecast uncertainty increases for longer horizons, visible in widening confidence intervals.
- Potential specification inconsistency in interpretation: some narrative statements about variable direction/significance do not perfectly align with the reported coefficient table.
- Monthly aggregation may smooth short-run shocks and intramonth volatility transmission.
- Market behavior in emerging economies can be strongly affected by structural breaks and policy discontinuities not fully modeled in simple linear + ARIMA setup.

## Research Gap
The study targets a practical gap in emerging-market literature: integrated assessment of key macroeconomic drivers (exchange rates, FDI, BOT) with forward forecasting for PSX/KSE-100 in one framework, aimed at actionable insights for investors and policymakers in Pakistan.

## Practical Relevance
For PSX-focused decision-making, the paper supports monitoring exchange-rate dynamics and trade-balance stress as primary market signals, while using ARIMA-based forecasting as a tactical planning layer rather than a certainty tool.
