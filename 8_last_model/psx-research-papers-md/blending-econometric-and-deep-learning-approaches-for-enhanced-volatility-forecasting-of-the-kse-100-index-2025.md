# Blending Econometric and Deep Learning Approaches for Enhanced Volatility Forecasting of the KSE-100 Index (2025)

## Paper Details
- Authors: Ubaida Fatima, Rimsha Zafar, Syeda Arsala Shah, Abdus Samad
- Venue: International Journal of Computer Applications (0975-8887)
- Issue: Volume 187, Number 35
- Publication date: August 2025
- Market focus: Pakistan (KSE-100), with generalization experiments on KMI-30

## Objective
The paper develops and compares individual and hybrid forecasting approaches for market volatility prediction in Pakistan, aiming to improve risk forecasting quality for investors, portfolio decisions, and risk management workflows.

## Methodology Overview
1. Collect daily index data and construct volatility forecasting setup.
2. Train individual models:
- ARIMA
- GARCH
- LSTM
- Linear Regression
3. Train hybrid models:
- ARIMA + LSTM
- Linear Regression + GARCH
4. Evaluate models with RMSE and MAE.
5. Test generalization from KSE-100 to KMI-30.

## Model Used
### Individual Models
- ARIMA (time-series linear dynamics)
- GARCH (conditional heteroskedasticity/volatility clustering)
- LSTM (deep sequence model for nonlinear temporal dependencies)
- Linear Regression (baseline supervised regression model)

### Hybrid Models
- ARIMA with LSTM
- Linear Regression with GARCH

## Data Used
- Primary dataset: Daily historical KSE-100 index data
- Time span: 2019 to 2023
- Fields noted in table: open/close among 5 columns, 1241 rows (for both KSE-100 and KMI-30 datasets)
- Secondary validation/generalization dataset: KMI-30 index (same methodology)

## Where the Data Came From
- Source listed in paper: Investing.com
- Link cited by authors: https://www.investing.com/
- Paper also references data usage from financial research publications, but operational time-series data source is Investing.com.

## Key Results
### Individual Model Error Matrix (KSE-100)
- LSTM: RMSE 0.001637, MAE 0.001178
- ARIMA: RMSE 0.052549, MAE 0.044685
- GARCH: RMSE 0.928456, MAE 0.853722
- Linear Regression: RMSE 0.001277, MAE 0.000765

Interpretation reported by paper: Linear Regression performed best among individual models, followed by LSTM, while ARIMA and GARCH underperformed on this dataset.

### Hybrid Model Error Matrix (KSE-100)
- ARIMA with LSTM: RMSE 0.01249, MAE 0.000156
- Linear Regression with GARCH: RMSE 0.01191, MAE 0.000141

Interpretation reported by paper: Linear Regression with GARCH is marginally superior to ARIMA with LSTM in this setup.

## Limitations
- Cryptocurrency volatility forecasting was excluded because available crypto data was monthly only, which constrained compatible modeling.
- Scope is mostly index-level volatility (KSE-100/KMI-30), limiting stock-level or cross-asset generalization.
- Results depend on a relatively recent period (2019-2023), so robustness across long historical regimes is not fully established.
- Traditional models were weak on this dataset; this may reflect either model specification or data regime sensitivity.

## Research Gap
The paper states a literature and practice gap in emerging-market forecasting tools for Pakistan, especially in comparative evaluation of econometric, machine learning, and hybrid methods for volatility-risk prediction and decision support. It addresses this by benchmarking individual and hybrid models and extending checks to KMI-30.

## Practical Implications
- For PSX-focused risk systems, the findings suggest prioritizing data-driven models with empirical validation over assumptions about model complexity.
- In this paper's setup, Linear Regression (individual) and Linear Regression + GARCH (hybrid) gave the lowest forecast errors and are practical candidates for operational monitoring.
