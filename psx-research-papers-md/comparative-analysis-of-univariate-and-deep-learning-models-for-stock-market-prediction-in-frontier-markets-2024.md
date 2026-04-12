# Comparative Analysis of Univariate and Deep Learning Models for Stock Market Prediction in Frontier Markets: A Case Study of Pakistan, Bangladesh, and Sri Lanka (2024)

## Paper Details
- Authors: Rabia Sabri, Sobia Iqbal
- Journal: Business Review
- Issue: Volume 19, Issue 2, pages 54-73
- Publication window: July-December 2024
- DOI: 10.54784/1990-6587.1656
- Case markets: Pakistan, Bangladesh, Sri Lanka (frontier markets)

## Objective
The study compares traditional univariate forecasting models and deep learning models for frontier-market stock prediction, and evaluates whether combining interpretability tools with advanced models can improve practical decision-making.

## Methodology Overview
1. Collect historical stock market and macroeconomic data.
2. Train and compare four model families:
- ARIMA
- Theta model
- LSTM
- 1D CNN
3. Evaluate with error metrics (MAE, RMSE, and model-fit measures such as R^2 in methodology discussion).
4. Forecast stock index returns for 2024-2025.
5. Improve interpretability of deep learning outputs using SHAP and LIME.

## Model Used
### Traditional Models
- ARIMA: for linear dependencies and short-term trend behavior.
- Theta: intended to improve seasonal/hidden-structure handling over plain ARIMA.

### Deep Learning Models
- LSTM: captures long-term dependencies and nonlinear temporal relationships.
- 1D CNN: captures local and complex patterns in time-series signals.

### Interpretability Layer
- SHAP (Shapley Additive Explanations): feature-level contribution analysis.
- LIME (Local Interpretable Model-agnostic Explanations): local explanation of prediction behavior.

## Data Used
- Stock market historical data for Pakistan, Bangladesh, Sri Lanka.
- Macroeconomic indicators referenced: GDP growth, inflation, interest rates, exchange rates.
- Time horizon (methodology section): January 2010 to December 2023.
- In-sample and out-of-sample splits referenced in text include:
- Training/estimation up to 2022
- Validation/testing in 2023
- Additional mention of in-sample 2010-2018 and out-of-sample 2019-2023 in discussion of evaluation design.
- Forecast horizon: 2024-2025.

## Where the Data Came From
- Financial databases and sources explicitly listed in methodology:
- International Monetary Fund (IMF)
- Bloomberg
- Yahoo Finance
- Macroeconomic inputs from official/government/central-bank and international institutional sources (including World Bank and IMF).

## Key Findings
- Deep learning approaches (LSTM and 1D CNN) generally outperform ARIMA and Theta in handling nonlinear and volatile frontier-market behavior.
- Traditional models remain easier to interpret and computationally cheaper.
- The paper argues for combining model families and interpretability methods for practical forecasting in unstable markets.

## Reported Prediction Comparison Snapshot (2024-2025, billion dollars)
- ARIMA:
- Pakistan predicted 65.2 vs actual 62.8 (error +2.4)
- Bangladesh predicted 30.5 vs actual 31.2 (error -0.7)
- Sri Lanka predicted 15.8 vs actual 14.5 (error +1.3)
- Theta:
- Pakistan 62.6 vs 60.1 (error +2.5)
- Bangladesh 29.2 vs 30.5 (error -1.3)
- Sri Lanka 14.5 vs 13.8 (error +0.7)
- LSTM:
- Pakistan 67.8 vs 64.5 (error +3.3)
- Bangladesh 31.5 vs 32.2 (error -0.7)
- Sri Lanka 16.2 vs 15.8 (error +0.4)
- 1D CNN:
- Pakistan 64.3 vs 61.8 (error +2.5)
- Bangladesh 30.1 vs 31.6 (error -1.5)
- Sri Lanka 15.6 vs 14.9 (error +0.7)

## Limitations
- Dataset window may not be long enough to fully capture long-run behavior of frontier markets (explicitly noted by authors).
- Deep learning models require higher compute resources and specialized expertise.
- Remaining interpretability challenge despite SHAP/LIME support (black-box concern not fully eliminated).
- Risk of overfitting in deep learning models, even with validation/tuning.
- Models do not fully account for unforeseen external shocks (for example abrupt political events or natural disasters).
- Methodology timing statements are not perfectly uniform across sections (some internal inconsistency in period/split narration).

## Research Gap
The paper positions its contribution around an underexplored area: applying and comparing traditional and deep learning methods in frontier South Asian markets (Pakistan, Bangladesh, Sri Lanka), where volatility and structural instability are stronger than in developed markets. It also targets a second gap: making deep learning outputs more interpretable for practical finance decisions through SHAP/LIME.

## Practical Relevance
- For frontier-market risk and forecasting workflows, the study supports using deep models for higher predictive power while pairing them with interpretability methods.
- For operational use, it suggests hybrid/ensemble strategies to balance:
- interpretability and cost (traditional models)
- nonlinear pattern capture and robustness in volatile regimes (deep learning models)
