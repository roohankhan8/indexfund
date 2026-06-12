# Predictive Performance of LSTM Networks on Sectoral Stocks in an Emerging Market: A Case Study of the Pakistan Stock Exchange (2025)

## Overview
This paper evaluates an LSTM-based forecasting framework for sectoral stocks on PSX rather than only index-level prediction. It studies how model performance differs across sectors with different liquidity and volatility profiles.

## Model Used
- Deep learning model: Long Short-Term Memory (LSTM) network.
- Implementation context: TensorFlow/Keras.
- Reported architecture includes stacked LSTM layers with dropout regularization.
- Typical split reported in methodology: chronological train-test setup (around 80% train, 20% test).
- Performance metric prominently reported: R-squared.

## Data Used
- Historical daily OHLCV (Open, High, Low, Close, Volume) stock data.
- Multi-year time-series dataset for ten major PSX stocks across different sectors.
- Engineered technical indicators (e.g., moving-average and momentum style features) and selected fundamental variables were used.

## Where the Data Came From
- Publicly available market data (web-scraped historical OHLCV sources, as described by authors).
- Supplemental fundamental values gathered from financial databases, then merged stock-wise.
- Study universe is explicitly Pakistan Stock Exchange sectoral stocks.

## Key Findings
- Strong predictive performance for relatively stable, liquid sectors; several stocks reported with high fit (e.g., R-squared above 0.87 in aggregate reporting, with some near/above 0.89).
- Lower performance in more shock-sensitive or low-liquidity stocks.
- Results support the usefulness of LSTM in emerging-market equities when data quality and liquidity conditions are favorable.

## Limitations
- Dependence on historical patterns; structural regime shifts can reduce reliability.
- Model does not natively ingest real-time exogenous information (news/sentiment/policy shocks).
- Data quality and coverage constraints for some stocks can impact generalization.

## Research Gap
- Need hybrid frameworks for volatile regimes (e.g., LSTM with volatility-aware components).
- Need richer exogenous feature integration (news sentiment, macro indicators, event signals).
- Need stronger robustness checks across sector-specific market microstructure differences.

## Practical Relevance
- Provides a replicable sector-level LSTM workflow for PSX forecasting.
- Useful for investors assessing where deep learning forecasts are more dependable by sector characteristics.
