# Forecasting Stock Prices Using Long Short-Term Memory Involving Attention Approach: An Application of Stock Exchange Industry (2025)

## Paper Details
- Authors: Muhammad Idrees, Maqbool Hussain Sial, Najam Ul Hassan
- Journal: PLOS ONE
- Publication date: 18 March 2025
- DOI: 10.1371/journal.pone.0319679
- Market context: Karachi/Pakistan Stock Exchange index data (KSE-100)

## Objective
The paper develops and compares deep-learning architectures for stock-price prediction, with focus on whether adding attention to sequence models improves prediction quality on KSE-100 data.

## Methodology Overview
1. Collect daily KSE-100 market data.
2. Pre-process and standardize features.
3. Build sequence windows from prior 10 days.
4. Train and compare four models:
- ANN
- RNN with Attention
- LSTM with Attention
- GRU with Attention
5. Evaluate with multiple error metrics and R-squared.
6. Interpret the selected model with SHAP feature-contribution analysis.

## Model Used
### Main Proposed Model
- LSTM with Attention architecture
- Stacked LSTM layers plus attention and dense layers

### Baseline/Comparator Models
- Artificial Neural Network (ANN)
- RNN-Attention
- GRU-Attention

### Evaluation Metrics
- RMSE
- MAE
- MAPE
- MSE
- R-squared

## Data Used
- Source dataset: KSE-100 historical market data
- Period: 22 February 2008 to 23 February 2021
- Rows: 3221 business-day observations
- Features used: Open, High, Low, Close, Change, Volume (Date dropped for training)
- Windowing: transformed to sequence shape using previous 10 days (3209 sequences)
- Split:
- Training: 3009 samples (22 Feb 2008 to 06 May 2020)
- Testing: 200 samples (07 May 2020 to 23 Feb 2021)

## Where the Data Came From
- Public source explicitly provided in the article:
- https://www.kaggle.com/datasets/zusmani/pakistan-stock-exchange-kse-100

## Key Findings
- LSTM-Attention is the best overall performer among tested models for this dataset.
- Reported R-squared values for LSTM-Attention:
- Training: 0.9996
- Validation: 0.9980
- Testing: 0.9921
- Testing metrics table reports:
- LSTM-Attention: RMSE 0.0245, MAE 0.0212, MAPE 2.6491, MSE 0.0008, R2 0.9921
- ANN: RMSE 0.0324, MAE 0.0246, MAPE 2.6361, MSE 0.0011, R2 0.9861
- RNN-Attention: RMSE 0.0294, MAE 0.0229, MAPE 2.1249, MSE 0.0008, R2 0.9384
- GRU-Attention: RMSE 0.0244, MAE 0.0187, MAPE 1.7404, MSE 0.0005, R2 0.9587

## SHAP-Based Interpretability Highlights
- Price features (Open, High, Close) show consistently stronger positive contribution.
- Change feature is most variable, with strong negative influence at certain time steps.
- Volume has comparatively small overall contribution in this setup.

## Limitations
- The study uses a single-market dataset (KSE-100), limiting cross-market generalization.
- Input feature space is primarily market OHLCV-style variables; broader macro/political/news variables are not included in the core model.
- The paper notes that some sequential alternatives (for example GRU-attention) can be less consistent across sessions despite favorable point metrics in some runs.

## Research Gap
The study targets a gap in PSX/KSE-focused forecasting literature by integrating attention into LSTM for sequence relevance selection, and by adding SHAP-based interpretability to compare model behavior beyond pure accuracy.

## Practical Relevance
For Pakistan equity forecasting pipelines, the findings support LSTM-attention as a strong candidate architecture for short-horizon predictive systems, particularly when paired with feature-attribution checks for model transparency.
