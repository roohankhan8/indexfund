# Modeling and Forecasting Stock Market Volatility of CPEC Founding Countries: Using Nonlinear Time Series and Machine Learning Models (2022)

## Paper Details
- Authors: Tayyab Raza Fraz, Samreen Fatima, Mudassir Uddin
- Journal: JISR Management and Social Sciences & Economics
- Volume/Issue: 20(1), 2022
- DOI: 10.31384/jisrmsse/2022.20.1.1
- Markets studied: KSE-100 (Pakistan) and SSE-100 (China)

## Objective
The study compares nonlinear econometric and machine-learning models to forecast stock-market volatility for CPEC founding-country markets and identify the most reliable forecasting approach under different error criteria.

## Methodology Overview
1. Compute daily stock returns from closing-price series.
2. Test stationarity and ARCH effects.
3. Fit volatility/forecast models:
- GARCH(1,1)
- CGARCH(1,1)
- Markov Regime Switching (MRS)
- LSTM (machine learning)
4. Select model fit with AIC/BIC.
5. Compare out-of-sample forecast performance using RMSE, MAE, MAPE, and SMAPE.

## Model Used
- Linear/nonlinear volatility models: GARCH, CGARCH, MRS
- Machine-learning model: LSTM (RNN-based)
- Diagnostics/model selection: AIC and BIC
- Forecast metrics: RMSE, MAE, MAPE, SMAPE

## Data Used
- Frequency: Daily closing prices
- Period: 1 December 2014 to 31 December 2021 (includes post-CPEC signing and COVID period)
- Markets: KSE-100 and SSE-100
- Estimation split: data up to 30 April 2021 used for estimation; remaining period used for out-of-sample forecast evaluation

## Where the Data Came From
- Source stated in methodology: Investing.com

## Key Findings
- Based on AIC/BIC, standard GARCH(1,1) is selected as best-fitted model among candidate GARCH-family fits.
- Forecast comparison is metric-dependent:
- For SSE-100: LSTM has best RMSE, while CGARCH performs best on MAE, MAPE, and SMAPE.
- For KSE-100: CGARCH is best on RMSE, while MRS performs best on MAE, MAPE, and SMAPE.
- Practical implication: LSTM forecasting power is close to CGARCH/MRS and can be used as an alternative forecasting tool.

## Limitations
- Study scope is restricted to two stock indices (KSE-100 and SSE-100), limiting external generalization.
- Forecast ranking varies by metric, so model choice is criterion-sensitive.
- Market-regime shocks and structural breaks can influence comparative performance over different windows.

## Research Gap
The paper addresses a gap in CPEC-related finance literature by jointly comparing machine-learning (LSTM), conditional-volatility (GARCH/CGARCH), and regime-switching (MRS) approaches on CPEC founder markets in one consistent forecasting framework.

## Practical Relevance
For investors and policymakers tracking CPEC-linked equity risk, the paper shows that no single model dominates all metrics; model choice should align with target forecast loss function, with LSTM serving as a viable operational alternative.
