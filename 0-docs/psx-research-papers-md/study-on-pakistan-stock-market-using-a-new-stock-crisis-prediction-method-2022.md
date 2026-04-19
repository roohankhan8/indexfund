# Study on the Pakistan Stock Market Using a New Stock Crisis Prediction Method (2022)

## Overview
This work proposes a multi-stage pipeline to predict stock crisis events in Pakistan equities, combining feature selection, classification, technical bubble detection, crisis-point marking, and deep learning prediction.

## Model Used
- Hybridized Feature Selection (HFS) for financial-variable reduction.
- Naive Bayes for identifying fundamentally strong stocks.
- Stochastic RSI (StochRSI) to detect bubble-like overbought behavior.
- Moving-average statistics for crisis-point identification.
- Deep learning forecasting models:
  - GRU
  - LSTM
- Evaluation metrics: RMSE, MSE, MAE.

## Data Used
- Pakistan stock datasets with ticker examples reported as HBL, NBP, UBL, BAFL.
- Financial ratios/technical variables used across the staged pipeline.
- Crisis concept defined around sharp drawdowns (paper discusses >10% collapse framing).

## Where the Data Came From
- Yahoo Finance, explicitly reported by the paper.
- Data availability section lists ticker access codes such as HBL.KA, NBP.KA, UBL.KA, BAFL.KA.

## Key Findings
- HFS-based GRU outperformed HFS-based LSTM for crisis prediction in reported experiments.
- The hybrid staged method improves crisis-focused forecasting structure over single-step approaches.
- Pipeline supports practical identification of pre-crisis and crisis phases.

## Limitations
- Performance depends on selected technical/fundamental attributes and data quality.
- Crisis forecasting remains sensitive to exogenous shocks not fully encoded in historical variables.
- Results are tested on a limited Pakistan stock subset.

## Research Gap
- Need inclusion of additional technical and macro event features.
- Need optimizer tuning and deeper architecture refinement (authors explicitly suggest further GRU improvement).
- Need broader cross-market validation and robustness testing over multiple crisis regimes.

## Practical Relevance
- Useful for investors and risk managers needing crisis-aware signals rather than only level forecasts.
- Provides a reproducible multi-step framework for PSX crisis prediction studies.
