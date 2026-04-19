# Modeling and Prediction of KSE-100 Index Closing Based on News Sentiments: Applications of Machine Learning Model and ARMA (p, q) Model (2022)

## Paper Details
- Authors: Asma Zaffar, S. M. Aalim Hussain
- Journal: Multimedia Tools and Applications (Springer)
- Publication year: 2022
- DOI: 10.1007/s11042-022-13052-2
- Market focus: KSE-100 / PSX closing-index behavior

## Objective
The study combines machine-learning and time-series modeling to forecast KSE-100 closing values, incorporating news sentiment and historical market features to improve prediction and reduce investor uncertainty.

## Methodology Overview
1. Collect historical KSE-100 market data (daily, 2015-2020) in six-month cycles.
2. Scrape news from multiple Pakistani news sources and process sentiment using NLP.
3. Build machine-learning forecasting pipeline (ANN/LSTM-based workflow described in paper).
4. Fit ARMA(p,q) models to cycle-wise time-series behavior.
5. Evaluate adequacy with AIC/SIC/HQC and diagnostics (DW, RMSE, MAE, MAPE, Theil’s U).

## Model Used
### Machine Learning / NLP Layer
- ANN/LSTM-style prediction workflow (paper discusses ANN and LSTM pipeline usage)
- News-sentiment feature generation using NLP and VADER

### Time-Series Layer
- ARMA(p,q) models selected cycle-wise via information criteria
- Mentioned use of ARIMA perspective in broader discussion, but core empirical selection is ARMA-based per cycle

## Data Used
- KSE-100 daily stock data (Open, High, Low, Close, Volume)
- Time span: 2015 to 2020
- 12 cycles, each of six-month duration
- News data scraped from multiple media outlets and aligned by date with market data
- Train-test strategy in ML section: 80/20 split

## Where the Data Came From
- Historical index data: Investing.com (as reported)
- News sources: Dawn, Daily Times, Business Recorder, The Express Tribune, Pakistan Observer

## Key Findings
- Most cycles are best fit by ARMA(2,1); cycles 2-5 prefer ARMA(3,1), cycle 8 prefers ARMA(1,1), cycle 12 prefers ARMA(4,1).
- Durbin-Watson values (<2, as interpreted by authors) and Theil’s U results indicate strong serial dependence across cycles.
- Forecast diagnostics (RMSE, MAE, MAPE, Theil’s U) suggest cycle-wise ARMA adequacy with strong relation to prior-cycle dynamics.
- ML workflow with news sentiment is reported to produce useful predictive behavior for KSE-100 closing direction/value trends.

## Limitations
- Results are heavily period-specific (2015-2020) and cycle-specific, which may reduce transferability.
- Model adequacy and preferred ARMA order vary by cycle.
- News scraping and sentiment quality depend on source consistency and preprocessing pipeline.
- Ambiguity in paper narrative between ANN and LSTM reporting can limit strict reproducibility without code artifacts.

## Research Gap
The paper targets a local applied gap by integrating news sentiment features with KSE-100 time-series modeling and comparing ML workflow outputs with ARMA diagnostics in a unified practical forecasting setup.

## Practical Relevance
For PSX participants, the study supports combining sentiment-aware machine-learning signals with ARMA diagnostics for short-horizon market monitoring and cycle-wise risk-aware decision support.
