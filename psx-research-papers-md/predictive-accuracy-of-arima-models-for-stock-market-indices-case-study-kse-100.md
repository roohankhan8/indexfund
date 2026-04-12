# The Predictive Accuracy of ARIMA Models for Stock Market Indices: A Case Study of KSE-100

## Overview
This study analyzes KSE-100 forecasting using ARIMA-based time-series modeling and compares short-horizon and trend-oriented configurations.

## Model Used
- Autoregressive Integrated Moving Average (ARIMA) framework.
- Comparative model configurations emphasized:
  - ARIMA(1,0,1)
  - ARIMA(1,1,1)
- Model comparison is based on forecasting behavior for short-term vs long-term trend capture.

## Data Used
- Daily KSE-100 index series.
- Study window reported as January 2012 to December 2023.

## Where the Data Came From
- Pakistan Stock Exchange context via KSE-100 historical index series used by the paper.
- The article frames data as historical KSE-100 daily observations for the specified period.

## Key Findings
- ARIMA(1,0,1) is reported as relatively better for short-term forecasting.
- ARIMA(1,1,1) is reported as better at capturing longer-run uptrend structure.
- Study confirms ARIMA utility for linear time-series behavior in KSE-100.

## Limitations
- ARIMA struggles with nonlinear and abrupt volatility dynamics.
- Forecast quality can degrade when market behavior deviates from linear/stationary assumptions.
- Structural breaks and event-driven shocks are not fully handled by basic ARIMA setups.

## Research Gap
- Need hybrid pipelines combining ARIMA with machine-learning/deep-learning approaches.
- Need stronger handling of nonlinearities, regime shifts, and volatility clustering.
- Need comparative out-of-sample validation under stressed market periods.

## Practical Relevance
- Gives investors and policy observers a baseline ARIMA reference for KSE-100 forecasting.
- Useful as a benchmark model before moving to richer hybrid forecasting architectures.
