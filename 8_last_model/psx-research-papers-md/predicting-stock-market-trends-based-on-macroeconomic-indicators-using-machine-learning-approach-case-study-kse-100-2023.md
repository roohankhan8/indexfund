# Predicting Stock Market Trends Based on Macroeconomic Indicators Through Machine Learning Approach: A Case Study of KSE-100 (2023)

## Overview
This study predicts KSE-100 index levels using Pakistan's macroeconomic indicators with a machine learning framework. The paper emphasizes ANN backpropagation for forecasting monthly and short-horizon daily trends.

## Model Used
- Artificial Neural Network (ANN) with backpropagation (multi-layer perceptron architecture).
- Supervised learning setup.
- Reported training/validation split: 80% / 20%.
- Reported hyperparameter details include learning rate around 0.0015 and iterative training.

## Data Used
- Monthly dataset from February 2004 to December 2020.
- Dependent variable: KSE-100 closing price.
- Independent variables: 16 macroeconomic indicators, including:
  - IPI, exchange rate, M2, CPI, FDI.
  - 3-month T-bill, KIBOR (1-month average), FX reserves.
  - House financing, balance of trade, crude oil, gold.
  - Labor force participation, GDP growth.
  - Household consumption, domestic savings.
- Additional KSE-100 daily historical data used for short-term daily forecasts.

## Where the Data Came From
- State Bank of Pakistan monthly publications.
- Pakistan Bureau of Statistics (including household/NPISH-related macro series context noted by authors).
- Pakistan Stock Exchange/KSE-100 historical index series.

## Key Findings
- Reported predictive performance is approximately 99% for the ANN model.
- Authors conclude KSE-100 movement is strongly associated with macroeconomic conditions.
- The model is used to generate:
  - Monthly forecast trends (2021 to mid-2023 window in the paper).
  - Daily short-horizon forecast trajectories.

## Limitations
- Single-model emphasis (ANN backpropagation) with limited comparative benchmarking against multiple modern ML/DL alternatives.
- Potential overfitting risk given high reported accuracy and finite historical sample.
- Forecast quality is sensitive to macro-data quality, timing, and revisions.
- External shocks and behavioral/sentiment factors are not deeply integrated into the core model.

## Research Gap
- Need robust head-to-head comparison with alternative models (LSTM/GRU/transformer, ensemble tree methods, hybrid econometric-ML baselines).
- Need stronger out-of-sample validation across structural breaks and crisis regimes.
- Need expanded feature space (news/sentiment, political-risk, global spillovers) and explainability analysis.

## Practical Relevance
- Provides a macro-informed forecasting template for PSX-focused investors and policy analysts.
- Shows how structured macroeconomic inputs can be operationalized in ML-based market forecasting pipelines.
