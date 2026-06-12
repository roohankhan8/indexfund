# Equations Used in the Project and Their Sources

This file consolidates the main equations used across the project and maps each one to:

1. The repo file(s) where it is implemented or documented.
2. The original theoretical source normally cited for that equation or method.

## Scope

The dissertation's final methodological focus is the KSE-30 workflow described in:

- `7_codex_model/pipeline.py`
- `6_cursor_model/`
- `report_workspace_2/chapter-03-methodology/chapter-03-methodology.txt`

Older but still project-relevant equations, especially for portfolio optimization, also appear in:

- `4a_claude_model_merged/merged_pipeline.py`
- `4_claude_model/nb4_portfolio_optimisation.py`
- `8_last_model/pipeline.py`

## 1. Data Construction and Return Equations

| Equation | Meaning in project | Repo source | Canonical source |
|---|---|---|---|
| `R_t = P_t / P_{t-1} - 1` | Simple return used in some later variants for fund NAV returns and stock returns | `8_last_model/pipeline.py` lines around 66, 80, 143, 144 | Standard return definition in empirical finance texts |
| `r_t = ln(P_t / P_{t-1})` | Log return used in the main KSE-30 workflow for stocks, funds, oil, USD/PKR, and index returns | `7_codex_model/pipeline.py` lines around 254, 283, 293, 324, 401; `4a_claude_model_merged/merged_pipeline.py` lines around 193, 249, 289, 307 | Standard continuous-compounding return definition; commonly used in time-series finance |
| `sigma_{30,t} = sd(r_{t-29}, ..., r_t) * sqrt(252)` | Annualized rolling 30-day volatility | `7_codex_model/pipeline.py` lines around 258, 325, 404, 1174; `4a_claude_model_merged/merged_pipeline.py` lines around 197, 228, 250 | Standard annualization rule in finance |
| `r^{index}_t = ln(FFMCAP_t / FFMCAP_{t-1})` | Reconstructed KSE-30 index log return from total free-float market capitalization | `7_codex_model/pipeline.py` lines around 395-405 | Standard index-return construction from capitalization levels |
| `R^{month} = sum_{d in month} r_d` | Monthly return built by summing daily log returns | `7_codex_model/pipeline.py` lines around 427-436 | Property of log returns |

## 2. Mutual Fund Flow Equations

| Equation | Meaning in project | Repo source | Canonical source |
|---|---|---|---|
| `DollarFlow_{i,t} = TNA_{i,t} - TNA_{i,t-1}(1 + R_{i,t})` | Literature-standard return-adjusted net flow for fund `i`; this is the academically safer form to report | `report_workspace_2/chapter-03-methodology/chapter-03-methodology.txt` lines 59-63, but should be interpreted using total return `R` rather than raw NAV change if distributions exist | Standard mutual-fund flow decomposition used in the fund-flow literature |
| `Flow_t = AUM_t - AUM_{t-1} * (NAV_t / NAV_{t-1})` | Project implementation proxy for return-adjusted flow; acceptable only if `NAV_t / NAV_{t-1} - 1` is a good proxy for total return | `report_workspace_2/chapter-03-methodology/chapter-03-methodology.txt` lines 59-63 | Implementation approximation, not the best literature form when distributions matter |
| `R_{i,t} = TotalReturn_{i,t}` | Preferred return term inside the flow equation because mutual-fund distributions can reduce NAV mechanically | not explicitly implemented in the current repo | Standard mutual-fund performance convention |
| `NAVReturn_m = NAV_end / NAV_start - 1` | Monthly NAV return used in the project as a practical return proxy | `report_workspace_2/chapter-03-methodology/chapter-03-methodology.txt` lines 63-65 | Standard holding-period return, but weaker than total return for mutual-fund flow estimation |
| `FlowPct_t = Flow_t / AUM_{t-1}` | Flow normalized by lagged AUM or lagged sector AUM | `7_codex_model/pipeline.py` lines around 468-470; `4a_claude_model_merged/merged_pipeline.py` line 269 | Standard normalized fund-flow measure |
| `FlowSpike_t = 1{|Flow_t - mean(Flow)| > 2 * sd(Flow)|}` | Flags unusually large aggregate flow events | `7_codex_model/pipeline.py` lines 472-473 | Standard z-score style outlier rule |

Recommended aggregation for the report:

- `CompositeTNA_t = sum_i TNA_{i,t}`
- `w_{i,t-1} = TNA_{i,t-1} / CompositeTNA_{t-1}`
- `CompositeReturn_t = sum_i w_{i,t-1} R_{i,t}`
- `CompositeNetFlow_t = CompositeTNA_t - CompositeTNA_{t-1}(1 + CompositeReturn_t)`
- `CompositeFlowPct_t = CompositeNetFlow_t / CompositeTNA_{t-1}`

This is stronger than calling `sum_i Flow_i` the market's "total net flow" without qualification, and it matches the minimal-change composite-index fix recommended for the dissertation.

## 3. Forecasting Equations

| Equation | Meaning in project | Repo source | Canonical source |
|---|---|---|---|
| `hat(y)_t = y_{t-1}` | Naive benchmark for monthly flow forecasting and some rebalancing tasks | described in `report_workspace_2/chapter-03-methodology/chapter-03-methodology.txt` line 101; used in `7_codex_model/pipeline.py` | Standard random-walk / persistence benchmark |
| `y_t = beta_0 + phi_1 y_{t-1} + beta' X_t + epsilon_t` | Custom ARIMAX-style monthly fund-flow model; in code this is effectively a 1-lag autoregression with exogenous macro variables estimated by least squares | `7_codex_model/pipeline.py` lines 859-877; methodology text lines 101-103 | Box and Jenkins (1970) for ARIMA/ARIMAX family |
| `Y_t = c + A_1 Y_{t-1} + B X_t + epsilon_t` | Custom VAR(1) system with endogenous block `[flow, interest rate, CPI]` and exogenous oil/USD returns | `7_codex_model/pipeline.py` lines 884-901 | Sims (1980), vector autoregression |
| `F-test: H_0: theta_1 = ... = theta_p = 0` | Granger-causality logic: does lagged `X` improve prediction of `Y` beyond lagged `Y`? | methodology text lines 99-101; Granger output in `7_codex_model/pipeline.py` lines around 841-847 | Granger (1969) |

## 4. Volatility Modelling Equations

| Equation | Meaning in project | Repo source | Canonical source |
|---|---|---|---|
| `sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2` | GARCH(1,1) conditional variance recursion | `7_codex_model/pipeline.py` lines 646-654; `overview/README.md` GARCH section | Bollerslev (1986) |
| `ln(sigma_t^2) = omega + alpha * (|z_{t-1}| - E|z|) + gamma z_{t-1} + beta ln(sigma_{t-1}^2)` where `z_t = r_t / sigma_t` and `E|z| = sqrt(2/pi)` | EGARCH(1,1) used to capture asymmetry / leverage | `7_codex_model/pipeline.py` lines 656-666 | Nelson (1991) |
| `NLL = 0.5 * sum_t [ln(2*pi*sigma_t^2) + r_t^2 / sigma_t^2]` | Gaussian negative log-likelihood minimized to estimate GARCH/EGARCH parameters | `7_codex_model/pipeline.py` lines 654, 666 | Standard Gaussian likelihood for conditional volatility models |
| `Persistence = alpha + beta` | Volatility persistence summary for GARCH(1,1) | reported in `report_workspace_2/chapter-05-results-and-analysis/chapter-05-results-and-analysis.txt` line 116 | Standard GARCH interpretation |
| `AIC = 2k + 2*NLL` | Model comparison metric for GARCH/EGARCH | `7_codex_model/pipeline.py` lines 679, 694 | Akaike (1974) |
| `BIC = k ln(n) + 2*NLL` | Model comparison metric for GARCH/EGARCH | `7_codex_model/pipeline.py` lines 679, 694 | Schwarz (1978) |
| `VaR_{0.05,t} = Phi^{-1}(0.05) * sigma_t` | 5% Value-at-Risk line from conditional volatility | `7_codex_model/pipeline.py` lines 759-760 | Variance-covariance VaR; J.P. Morgan RiskMetrics tradition |

## 5. Market-Efficiency and Stationarity Equations

| Equation | Meaning in project | Repo source | Canonical source |
|---|---|---|---|
| `Delta s_t = c + rho s_{t-1} + u_t` | Simplified ADF-style stationarity regression used in code | `7_codex_model/pipeline.py` lines 793-807 | Dickey and Fuller (1979, 1981) |
| `Z_runs = (R - E[R]) / sqrt(Var[R])` with `E[R] = 2 n_+ n_- / n + 1` and `Var[R] = 2 n_+ n_- (2 n_+ n_- - n) / (n^2 (n-1))` | Runs test for randomness of return signs | `7_codex_model/pipeline.py` lines 973-982 | Wald and Wolfowitz (1940) |
| `VR(q) = Var(sum_{i=0}^{q-1} r_{t-i}) / (q Var(r_t))` | Variance-ratio test for random walk behavior | `7_codex_model/pipeline.py` lines 984-996; `4a_claude_model_merged/merged_pipeline.py` lines 824-835 | Lo and MacKinlay (1988) |
| `Q = n(n+2) sum_{k=1}^{m} rho_k^2 / (n-k)` | Ljung-Box serial-correlation test statistic | `7_codex_model/pipeline.py` lines 998-1005 | Ljung and Box (1978) |
| `R/S` scaling with `H = slope of ln(R/S) on ln(lag)` | Hurst exponent from rescaled-range analysis; used to classify persistence vs mean reversion | `7_codex_model/pipeline.py` lines 1007-1025 | Hurst (1951) |

## 6. Portfolio Optimisation Equations

This block appears in the earlier project workflow and in the merged pipeline. It is part of the project, but not the final dissertation's main KSE-30 rerun.

| Equation | Meaning in project | Repo source | Canonical source |
|---|---|---|---|
| `mu_p = w' mu` | Portfolio expected return | `4a_claude_model_merged/merged_pipeline.py` lines 687-692 | Markowitz (1952) |
| `sigma_p = sqrt(w' Sigma w)` | Portfolio volatility | `4a_claude_model_merged/merged_pipeline.py` lines 687-692 | Markowitz (1952) |
| `Sharpe_p = (mu_p - r_f) / sigma_p` | Risk-adjusted return used to compare equal-weight, market-cap, and optimized portfolios | `4a_claude_model_merged/merged_pipeline.py` lines 687-697 | Sharpe (1966, 1994) |
| `max_w Sharpe_p` subject to `sum_i w_i = 1` | Maximum-Sharpe portfolio solved numerically | `4a_claude_model_merged/merged_pipeline.py` lines 694-715 | Mean-variance portfolio optimization |
| `min_w -Sharpe_p` | Equivalent minimization form used in code | `4a_claude_model_merged/merged_pipeline.py` lines 694-697 | Numerical implementation of maximum-Sharpe optimization |

## 7. Rebalancing and Portfolio-Tilt Equations

| Equation | Meaning in project | Repo source | Canonical source |
|---|---|---|---|
| `Mom_h = sum_{i=0}^{h-1} r_{t-i}` for `h in {30, 60, 90}` | Momentum features for KSE-30 rebalancing windows | `7_codex_model/pipeline.py` lines 1171-1173, 1291-1292 | Standard momentum-feature construction in empirical asset pricing |
| `Vol_h = sd(r_{t-h+1}, ..., r_t) * sqrt(252)` | Short-horizon volatility features | `7_codex_model/pipeline.py` lines 1174-1176, 1293-1295 | Standard annualized realized volatility feature |
| `PriceToMA20_t = P_t / MA20_t - 1` | Distance from 20-day moving average | `7_codex_model/pipeline.py` lines 1179-1182, 1298-1301 | Standard technical-analysis feature |
| `PriceToMA50_t = P_t / MA50_t - 1` | Distance from 50-day moving average | `7_codex_model/pipeline.py` lines 1181-1182, 1300-1301 | Standard technical-analysis feature |
| `WtDrift = w_t - w_{t-h}` | Change in index weight across the lookback window | `7_codex_model/pipeline.py` lines 1183-1184, 1302-1303 | Project-specific feature engineering |
| `WtRange = max(w) - min(w)` | Weight instability feature | `7_codex_model/pipeline.py` lines 1185-1188, 1304-1305 | Project-specific feature engineering |
| `Retained = 1` if stock remains in next KSE-30 review, else `0` | Binary target for retention/inclusion prediction | `7_codex_model/pipeline.py` lines 1187-1188 | Project-specific target definition |
| `hat(w)_{avg} = (hat(w)_{ridge} + hat(w)_{rf}) / 2` | Final forward weight forecast ensemble | `7_codex_model/pipeline.py` lines 1311-1314 | Project-specific ensemble rule |
| `hat(p)_{ret} = (p_{logit} + p_{rf}) / 2` | Average retention probability used in the forward forecast | `7_codex_model/pipeline.py` lines 1315-1318 | Project-specific ensemble rule |
| `ExclusionRisk = 1 - hat(p)_{ret}` | Reported exclusion-risk score | `7_codex_model/pipeline.py` lines 1317-1319 | Project-specific transformation |

## 8. Portfolio-Tilt Equations in `8_last_model`

| Equation | Meaning in project | Repo source | Canonical source |
|---|---|---|---|
| `Score_f = max(hat(Flow)_f, 0)` | Negative predicted flows are truncated to zero before tilt weighting | `8_last_model/pipeline.py` lines 220-221 | Project-specific allocation rule |
| `w^{tilt}_f = Score_f / sum_j Score_j` if `sum_j Score_j > 0`, else `1/F` | Flow-based tilt weights across funds | `8_last_model/pipeline.py` lines 223-227 | Project-specific portfolio-tilt rule |
| `w^{eq}_f = 1/F` | Equal-weight benchmark across funds | `8_last_model/pipeline.py` line 227 | Standard equal-weight benchmark |
| `CumRet_t = prod_{tau <= t} (1 + r_tau)` | Cumulative return construction for the tilt and equal-weight strategies | `8_last_model/pipeline.py` lines 236-237 | Standard cumulative return formula |

## 9. Evaluation Metrics Used in the Project

| Equation | Meaning in project | Repo source | Canonical source |
|---|---|---|---|
| `RMSE = sqrt((1/n) sum_i (y_i - hat(y)_i)^2)` | Main regression error metric | used throughout `7_codex_model/pipeline.py`, `4a_claude_model_merged/merged_pipeline.py` | Standard forecasting metric |
| `MAE = (1/n) sum_i |y_i - hat(y)_i|` | Secondary regression error metric | same as above | Standard forecasting metric |
| `R^2 = 1 - SS_res / SS_tot` | Relative explanatory fit for regression tasks | `7_codex_model/pipeline.py` lines around 174-184, 871, 901; `4a_claude_model_merged/merged_pipeline.py` lines 137-146 | Standard regression metric |
| `DirAcc = (1/n) sum_i 1{sign(y_i) = sign(hat(y)_i)}` | Directional accuracy for flow forecasts | `7_codex_model/pipeline.py` lines around 174-184; `4a_claude_model_merged/merged_pipeline.py` lines 137-147 | Standard financial forecast evaluation |
| `Accuracy = (TP + TN) / (TP + TN + FP + FN)` | Classification accuracy for retention models | `7_codex_model/pipeline.py` lines 1244-1260 | Standard classification metric |
| `AUC = area under ROC curve` | Ranking quality for retention-probability models | `7_codex_model/pipeline.py` lines 1244-1260 | Hanley and McNeil (1982) style ROC/AUC usage |

## 10. Practical Citation List for the Report

If you want short academic citations for the major equations, these are the main ones to cite:

- Markowitz, H. (1952). Portfolio Selection.
- Sharpe, W. F. (1966). Mutual Fund Performance.
- Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity.
- Nelson, D. B. (1991). Conditional Heteroskedasticity in Asset Returns: A New Approach.
- Box, G. E. P., and Jenkins, G. M. (1970). Time Series Analysis: Forecasting and Control.
- Sims, C. A. (1980). Macroeconomics and Reality.
- Granger, C. W. J. (1969). Investigating Causal Relations by Econometric Models and Cross-spectral Methods.
- Dickey, D. A., and Fuller, W. A. (1979, 1981). Unit-root testing papers.
- Lo, A. W., and MacKinlay, A. C. (1988). Stock Market Prices Do Not Follow Random Walks.
- Ljung, G. M., and Box, G. E. P. (1978). On a Measure of Lack of Fit in Time Series Models.
- Hurst, H. E. (1951). Long-term storage capacity of reservoirs.
- Wald, A., and Wolfowitz, J. (1940). Runs test for randomness.

For the fund-flow equation specifically, your report should cite papers from the mutual-fund flow literature that define flow as return-adjusted change in total net assets. In the dissertation text, describe your sector series as a proxy aggregate flow built from AKD, NBP, and NTI rather than an official observed KSE-30 net-flow series.

## 11. Notes on What Was Not Written as a Closed-Form Equation

Some project components were implemented as algorithms rather than one neat closed-form equation:

- Random Forest regression and classification
- Ridge regression
- Logistic regression probabilities were used directly from the model output

These are still part of the project, but the most defensible equations to report are the data-construction, time-series, volatility, efficiency, portfolio, and evaluation equations listed above.
