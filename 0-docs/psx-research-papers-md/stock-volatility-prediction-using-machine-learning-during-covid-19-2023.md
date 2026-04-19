# Stock Volatility Prediction Using Machine Learning During Covid-19 (2023)

## Overview
The paper studies stock return/price-direction prediction under high-volatility pandemic conditions on PSX and compares SVM kernels for predictive performance.

## Model Used
- Support Vector Machine (SVM) classification framework.
- Kernel comparison:
  - Linear kernel
  - Polynomial kernel
  - RBF kernel
- Objective: forecast directional movement under unstable market conditions.

## Data Used
- Daily stock data during Covid-19 and subsequent recovery horizon (paper reports coverage through roughly 2020 to March 2022).
- Three PSX stocks from different sectors.
- Fifteen predictors grouped into categories such as price-volume, valuation, risk, and scale factors.

## Where the Data Came From
- Secondary financial data sources, with stock price series referenced from Pakistan stock market sources in the paper.
- Data processed and modeled in Python.

## Key Findings
- RBF kernel delivers the strongest validation performance among tested kernels (reported around 62% in headline results).
- Linear and polynomial kernels underperform relative to RBF in the high-volatility environment.
- The study supports SVM-based direction forecasting utility during shock periods.

## Limitations
- Limited stock universe (three stocks), constraining generalization.
- Accuracy levels vary by stock and are moderate, indicating residual prediction uncertainty.
- Crisis-period behavior may not transfer cleanly to normal regimes.

## Research Gap
- Need broader cross-sector and longer-horizon testing beyond a narrow crisis sample.
- Need comparative benchmarking against modern deep-learning and ensemble methods.
- Need dynamic feature engineering that adapts to regime transitions.

## Practical Relevance
- Offers a practical baseline for kernel selection in SVM under stressed market conditions.
- Helps investors and analysts evaluate model robustness during volatility spikes.
