# Prediction of Stock Market Movement Using Long Short-Term Memory (LSTM) Artificial Neural Network: Analysis of KSE-100 Index (2024)

## Overview
This paper builds a machine-learning framework to predict KSE-100 movement using a broad feature set that includes economic, social, political, and administrative indicators. The approach integrates ANN backpropagation and LSTM for sequence-aware prediction.

## Model Used
- Artificial Neural Network (ANN) with backpropagation.
- Long Short-Term Memory (LSTM) network for sequential dependence.
- Reported architecture/hyperparameters include:
  - 26 input neurons,
  - 3 hidden layers with 50 neurons each,
  - learning rate 0.0015,
  - maximum epoch 5000.
- Data split reported as training/testing/validation around 80%/10%/10%.

## Data Used
- Monthly data from February 2004 to December 2020.
- Output variable:
  - KSE-100 closing price.
- 26 input indicators covering macroeconomic, social, political, and governance dimensions, including (as listed by the paper):
  - balance of trade, house-building consumer financing, CPI, control of corruption,
  - crude oil, domestic savings, exchange rate, external debt stocks,
  - FDI, foreign exchange reserves, GDP growth, gold price,
  - government effectiveness, household final consumption expenditure,
  - industrial production index, industry value added,
  - labor force participation, money supply, personal remittances growth,
  - political stability/absence of violence, portfolio investment growth,
  - regulatory quality, rule of law, 3-month treasury bill, wholesale price index.

## Where the Data Came From
- World Development Indicators (WDI).
- State Bank of Pakistan (SBP).
- Pakistan Bureau of Statistics (PBS).
- KSE-100 historical index series from Pakistan's stock market context.

## Key Findings
- The paper reports around 99% prediction accuracy for KSE-100 movement.
- Forecast outputs suggest KSE-100 remaining near 40,000 points through September 2023 (as reported in-study forecast horizon).
- Comparison of actual vs predicted values (January 2021 to July 2022) is used as model validation evidence.

## Limitations
- Heavy dependence on historical relationships; structural market regime changes can reduce model reliability.
- Data quality/availability constraints can affect generalization to new conditions.
- Long-horizon forecast stability is uncertain without periodic model re-estimation.

## Research Gap
- Need dynamic/online model updating rather than static training.
- Need stronger robustness workflows (strict out-of-sample tests, repeated cross-validation).
- Need ensemble/hybrid modeling and external real-time signals (e.g., news sentiment, social signals) to improve resilience.

## Practical Relevance
- Demonstrates a broad-factor ML pipeline for PSX trend forecasting.
- Offers a framework investors and analysts can adapt for periodic KSE-100 directional forecasting.
