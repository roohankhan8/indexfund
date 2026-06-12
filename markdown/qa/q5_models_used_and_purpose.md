# Q5: Models used in `6_cursor_model/pipeline.py` and their purpose

## 1) Volatility modeling (Section 4)
1. `GARCH(1,1)`
- Purpose: Model time-varying volatility of KSE-30 daily log returns.
- Used for: estimating conditional volatility and risk persistence (`alpha + beta`), plus VaR backtesting.

2. `EGARCH(1,1)`
- Purpose: Alternative volatility model that can capture asymmetric volatility behavior.
- Used for: model comparison against GARCH via AIC/log-likelihood; best model is selected for reporting/plots.

## 2) Aggregate fund-flow forecasting (Section 5)
3. `ARIMAX(1,0,1)` (with macro exogenous variants)
- Purpose: Forecast aggregate KSE-30 sector fund flow using past flow dynamics + macro inputs.
- Used for: transformed-flow prediction and out-of-sample evaluation.

4. `VAR(1)`
- Purpose: Jointly model flow and macro variables in a system where variables depend on lagged values of each other.
- Used for: comparative forecasting of aggregate sector flow.

5. `Naive (Random-Walk style) baseline`
- Purpose: Benchmark model (next value ~ previous value / carry-forward style baseline).
- Used for: reference performance against ARIMAX and VAR.

## 3) Rebalancing weight prediction (Section 7, regression)
6. `Ridge Regression`
- Purpose: Predict next rebalance constituent weight (`target_weight`) with regularized linear regression.
- Used for: main linear benchmark for weight prediction.

7. `RandomForestRegressor`
- Purpose: Nonlinear weight prediction model.
- Used for: compare with Ridge and Naive; also used in CV and forward predictions.

8. `Naive weight baseline` (use current weight)
- Purpose: Simple benchmark for rebalancing weight task.
- Used for: compare whether ML models improve over “no-change” weight assumption.

## 4) Rebalancing inclusion/retention prediction (Section 7, classification)
9. `LogisticRegression`
- Purpose: Predict binary retention/inclusion probability at next rebalance.
- Used for: interpretable probabilistic classifier baseline.

10. `RandomForestClassifier`
- Purpose: Nonlinear classifier for inclusion/retention.
- Used for: compare accuracy/AUC vs Logistic and Naive.

11. `Naive inclusion baseline` (predict all retained)
- Purpose: Simple benchmark classifier.
- Used for: sanity-check model lift versus trivial rule.

## 5) Statistical tests used for diagnostics (not predictive ML models)
12. `Granger causality tests`
- Purpose: Check whether lagged macro variables add predictive information for fund flow.

13. `Runs test`, `Variance Ratio (VR) test`, `Ljung-Box Q`, `Hurst exponent`
- Purpose: Market-efficiency / dependence diagnostics on index returns.

## Notes
- Main predictive tasks are: volatility estimation, sector flow forecasting, rebalance weight regression, and inclusion classification.
- Multiple Naive baselines are intentionally included to show whether complex models add real value.
