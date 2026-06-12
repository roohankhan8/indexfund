# Required Changes for `FYP REPORT FINAL.pdf`

This file lists the changes needed to make `docs/reports/root_exports/FYP REPORT FINAL.pdf` align with the current project, the examiner's feedback, and the final production workflow.

Reference run used for alignment:

- command: `python production_pipeline/run_all.py`
- run date: `2026-06-12`
- canonical outputs folder: `production_pipeline/output/analysis/`
- canonical report asset folder: `docs/report_workspace/`

## 1. Core framing changes

### 1.1 Stop presenting the target as a direct market-wide total-fund-flow series

The examiner objected to the report's original total-fund-flow equation because it was not sufficiently backed by the literature review. The report should therefore stop treating that equation as the final study target.

The corrected framing is:

- the study does not use an official published KSE-30 market-wide net-flow series
- the study constructs a three-fund sector proxy index from:
  - `AKD`
  - `NBP`
  - `NTI/NIT`
- that composite index is the final mutual-fund-based proxy for KSE-30-related institutional behavior

Use wording such as:

> The study does not rely on an official market-wide KSE-30 net-flow series. Instead, it constructs a three-fund sector proxy index from AKD, NBP, and NTI/NIT and uses that composite index as the observable mutual-fund representation of KSE-30-related institutional activity.

Avoid wording such as:

- "official KSE-30 fund flow"
- "direct total fund flow"
- "observed market-wide KSE-30 net flow"

This change is needed in:

- Executive Summary
- Chapter 1 introduction and objectives
- Chapter 3 methodology
- Chapter 4 results captions and interpretation
- Chapter 6 discussion
- any appendix still describing the target as direct total flow

### 1.2 Replace the old headline equation with the new index-construction equations

The old equation can remain only as literature background, not as the main dissertation target.

Background-only literature form:

`DollarNetFlow_(i,t) = TNA_(i,t) - TNA_(i,t-1) * (1 + R_(i,t))`

If you want to mention the project's earlier proxy logic, keep it only as a short explanatory note:

`FlowProxy_(i,t) = AUM_(i,t) - AUM_(i,t-1) * (NAV_(i,t) / NAV_(i,t-1))`

The final methodology should instead report the three-fund index construction:

`r_(i,t) = (NAV_(i,t) - NAV_(i,t-1)) / NAV_(i,t-1)`

`IndexLevel_(i,t) = 100 * NAV_(i,t) / NAV_(i,0)`

If the dissertation uses equal weighting, use:

`FundIndex_t = (1/3) * [IndexLevel_(AKD,t) + IndexLevel_(NBP,t) + IndexLevel_(NTI,t)]`

If the dissertation uses fixed weights instead, replace the equal-weight equation with:

`FundIndex_t = sum_(i in {AKD,NBP,NTI}) w_i * IndexLevel_(i,t), where sum_i w_i = 1`

If the predictive target is the composite index return, add:

`FundIndexReturn_t = ln(FundIndex_t / FundIndex_(t-1))`

The report should explicitly say:

- the literature-standard flow equation motivated the initial approach
- the examiner required replacing the unsupported direct total-flow target with an observable three-fund index proxy
- the final modeled target is the composite three-fund index or its return series
- any constituent-level flow proxy is supportive background only

This is especially important in Chapter 3.

## 2. Methodology changes

### 2.1 Add a dedicated subsection for index construction

Chapter 3 should contain a subsection such as:

- `Construction of the Three-Fund Sector Proxy Index`

That subsection should explain:

- why the original direct total-flow equation was not retained
- why `AKD`, `NBP`, and `NTI/NIT` were selected
- whether the index is equal-weighted or fixed-weighted
- the rebasing step to `100`
- whether the forecasting target is the index level, log return, or monthly change

### 2.2 Re-scope the modeling section to match the actual final pipeline

The report TOC shows:

- `3.9.5 Long Short-Term Memory (LSTM) Model`
- `3.9.7 Hybrid Forecasting Framework`

The final executable production pipeline does not use LSTM as a final deployed model. The retained workflow is centered on:

- ARIMAX-style fund-index forecasting
- VAR(1)
- GARCH / EGARCH
- market-efficiency diagnostics
- ridge / logistic / random forest comparisons for rebalancing

Required change:

- either remove the LSTM subsection entirely
- or keep it only as literature/background and explicitly state that it is not part of the final implemented pipeline

Do the same for any "hybrid framework" claims if they imply a final production model that is not actually executed.

### 2.3 Update data-splitting wording

The report currently describes training, validation, and testing in generic terms.

The final run actually uses:

- fund-index forecasting: `34` training months and `25` test months
- rebalancing panel: `364` training observations and `58` test observations

Required change:

- replace vague train/validation/test language with the exact split logic used in the final pipeline
- if a separate validation split was not used in the retained code path, do not claim a formal three-way split

### 2.4 Update the final sample coverage

The final regenerated outputs are:

- `daily_master.csv`: `1300` rows x `12` columns, `2021-01-04` to `2026-04-30`
- `monthly_master.csv`: `59` rows x `38` columns, `2021-03-31` to `2026-01-30`
- cleaned stock panel: `46,560` rows, `59` symbols, `2020-01-01` to `2026-04-30`

Required change:

- revise all methodology and result sections that mention sample size, date span, or variable count so they match these final outputs

## 3. Results section changes

### 3.1 Update fund-index forecasting results

Current final run:

- Naive (RW): `RMSE 83.35`, `MAE 51.36`, `R^2 -1.4196`, `DirAcc 37.5%`
- ARIMAX(1,0,1): `RMSE 58.00`, `MAE 36.52`, `R^2 -0.1716`, `DirAcc 70.8%`
- VAR(1): `RMSE 62.54`, `MAE 38.80`, `R^2 -0.3622`, `DirAcc 75.0%`

Required change:

- replace older table values in Chapter 4 and any appendix summary table with these rerun values
- rewrite the text so these models are described as forecasting the three-fund index target rather than a directly observed market-wide total-flow series
- note that ARIMAX is best on RMSE/MAE, while VAR is strongest on directional accuracy
- keep the interpretation that directional utility matters more than point fit alone because out-of-sample `R^2` remains negative

### 3.2 Update stationarity and Granger discussion

Current run summary:

- `total_fund_flow`: stationary, `p = 0.0000`
- `interest_rate_end`: non-stationary, `p = 0.2420`
- `cpi_yoy_end`: non-stationary, `p = 0.4012`
- `oil_return_monthly`: stationary, `p = 0.0000`
- `usdpkr_return_monthly`: stationary, `p = 0.0000`

Current Granger results:

- IR -> target: `p = 0.6405`
- CPI -> target: `p = 0.0532`
- Oil -> target: `p = 0.7365`
- USD/PKR -> target: `p = 0.6365`

Required change:

- update the text to say CPI is borderline at the 10% level, not conventionally significant at 5%
- avoid implying that any macro variable strongly Granger-causes the target series
- rename the target consistently if these tests are now interpreted against the three-fund composite target

### 3.3 Update volatility modeling results

Current final run:

- preferred model: `EGARCH(1,1)`
- EGARCH AIC: `4209.7`
- GARCH persistence: `0.9667`

Required change:

- ensure Chapter 4 says EGARCH is the preferred final specification by AIC
- keep leverage/asymmetry discussion only if tied to the EGARCH outcome

### 3.4 Update market-efficiency results

Current final run:

- Runs test Z: `-1.9106`
- Runs p-value: `0.0561`
- Variance Ratio VR(2): `1.0069`
- Variance Ratio p-value: `0.8752`
- Ljung-Box p-value: `0.0000`
- Hurst exponent: `0.6559`

Required change:

- present the conclusion as mixed or borderline evidence
- do not call the market fully efficient
- do not call it fully inefficient either
- explain that runs and variance-ratio evidence is closer to weak-form efficiency, while Ljung-Box and Hurst imply persistence or serial dependence

### 3.5 Update rebalancing results

Current final run:

- detected rebalancing dates: `17`
- training panel: `422` rows
- effective windows: `16`
- symbols in panel: `47`
- retained rate: `91.9%`

Weight prediction:

- Naive: `RMSE 0.5595`, `MAE 0.2541`, `R^2 0.9677`
- Ridge: `RMSE 0.5590`, `MAE 0.2963`, `R^2 0.9678`
- Random Forest: `RMSE 0.6905`, `MAE 0.3607`, `R^2 0.9509`

Inclusion prediction:

- Naive accuracy: `0.9655`
- Logistic accuracy: `0.9655`, `AUC 0.8214`
- Random Forest accuracy: `0.9655`, `AUC 0.5893`

Required change:

- update all Chapter 5 and appendix tables with these values
- if the report currently implies Random Forest is the final winning rebalancing model, revise that
- the final evidence supports:
  - ridge or naive strength for weight prediction
  - logistic regression as the more credible classifier by AUC

### 3.6 Update forward-looking rebalancing language

Current forecast section states the next rebalancing is approximately:

- `2026-09-16`

The highest-risk names in the current forecast include:

- `SSGC`
- `GHNI`
- `GAL`
- `PAEL`

Required change:

- update the narrative in Chapter 5 and Chapter 6 so all forecast examples use the regenerated forecast table
- do not leave older stock examples if they differ from the latest output

## 4. Figure, table, and chapter-placement changes

### 4.1 Replace stale figures with regenerated current figures

The report should use figures copied or generated from:

- `production_pipeline/output/analysis/figures/`
- `docs/report_workspace/chapter-03-methodology/images/`
- `docs/report_workspace/chapter-04-data-collection-and-processing/images/`
- `docs/report_workspace/chapter-05-results-and-analysis/images/`
- `docs/report_workspace/chapter-06-portfolio-tilt-and-rebalancing-application/images/`
- `docs/report_workspace/chapter-07-discussion/images/`
- `docs/report_workspace/chapter-08-conclusion-and-recommendations/images/`

Required change:

- replace any figure copied from an older run if its numbers, labels, or interpretation no longer match the current result tables

### 4.2 Move important appendix material into the main chapters

The report is currently too appendix-heavy. Important evidence should not be buried outside the main argument.

Move these items into the main chapters:

- the final three-fund index equations into Chapter 3
- the final forecasting comparison table into Chapter 4
- the final stationarity summary and Granger summary into Chapter 4
- the final efficiency summary table into Chapter 4
- the final rebalancing comparison table into Chapter 5
- the final forward-risk or forecast table into Chapter 5

Keep appendices only for:

- extended raw tables not discussed in detail
- supplementary figures not essential to the core argument
- technical implementation material only if required by formatting rules

### 4.3 Reduce appendix size

Recommended appendix cuts:

- remove screenshot-heavy repository appendices if they do not add analytical value
- remove repeated figure versions when one final figure is already used in the chapter
- compress long appendix prose into short tables where possible
- delete appendix content that is never referenced in the discussion or conclusion

## 5. Terminology and wording fixes

### 5.1 Use one consistent fund name convention

The report currently mixes:

- `NTI`
- `NIT`
- `NTI/NIT`

Required change:

- pick one convention and use it consistently
- if the actual fund name and the project variable name differ, explain that once and then stay consistent

### 5.2 Fix over-claiming about machine learning

The Executive Summary currently overstates the project as if the whole workflow is machine-learning-driven.

Required change:

- distinguish:
  - econometric forecasting: ARIMAX / VAR
  - volatility modeling: GARCH / EGARCH
  - rebalancing prediction: ridge / logistic / random forest comparison
- do not imply that LSTM is part of the final implemented forecasting workflow unless you explicitly label it as background only

### 5.3 Replace generic "correlation and regression" wording

The final project is more specific than generic correlation/regression language.

Required change:

- explicitly mention:
  - stationarity testing
  - Granger causality
  - ARIMAX
  - VAR
  - GARCH / EGARCH
  - market-efficiency diagnostics
  - rebalancing prediction

## 6. Structural and formatting corrections

### 6.1 Fix appendix lettering

If the report still jumps from Appendix B to D or E:

- either add the missing Appendix C
- or relabel later appendices so the sequence is continuous

### 6.2 Clean the source formatting

Fix any editable-source issues such as:

- broken spacing
- split words
- inconsistent equation formatting
- inconsistent superscript and subscript notation

### 6.3 Rebalance chapter content versus appendix content

Required structural change:

- Chapter 3 should contain the final target-equation block and variable definitions
- Chapter 4 should contain the main forecasting, volatility, and efficiency result tables
- Chapter 5 should contain the main rebalancing result tables and the forward-risk table
- appendices should hold only overflow material

## 7. Recommended chapter-by-chapter corrections

### Executive Summary

- reframe the target as a three-fund sector proxy index
- state that this change was made after examiner feedback objected to the unsupported total-flow equation
- remove any implication that the final system directly uses LSTM in production
- distinguish econometric forecasting from ML-based rebalancing support

### Chapter 3: Methodology

- replace the old total-flow headline equation with the three-fund index-construction equations
- keep the literature flow equation only as background context
- state whether the final index is equal-weighted or fixed-weighted
- update sample coverage and split logic
- align variables with the actual final master datasets

### Chapter 4: Results and Analysis

- replace all stale numbers with current rerun values
- rewrite tables and captions so the predicted target is the three-fund composite target
- update stationarity and Granger text
- state EGARCH is preferred by AIC
- keep market-efficiency interpretation mixed or borderline

### Chapter 5: Portfolio Tilt and Rebalancing Application

- refresh panel counts and test metrics
- update model comparison narrative so it matches ridge and logistic performance
- refresh the forward forecast examples and high-risk names
- move the final forward-risk table here if it is currently buried in an appendix

### Chapter 6: Discussion

- align discussion claims with current results rather than earlier runs
- avoid overstating forecast strength where `R^2` is negative but directional accuracy is useful
- explain why the three-fund index is a defensible compromise between literature ideals and actual data availability

### Chapter 7: Conclusion and Recommendations

- make sure the conclusion reflects a decision-support system, not a live deployable trading system
- retain transaction-cost and implementation-limitations disclaimers

### Appendices

- cut appendices down to supplementary material only
- move any equation, key summary table, or key result figure that is cited in the argument into the main chapters
- regenerate any remaining appendix tables from current CSV outputs

## 8. Files to use when revising the report

Use these outputs as the source of truth:

- `production_pipeline/output/analysis/daily_master.csv`
- `production_pipeline/output/analysis/monthly_master.csv`
- `production_pipeline/output/analysis/results_fund_flow.csv`
- `production_pipeline/output/analysis/results_garch.csv`
- `production_pipeline/output/analysis/results_efficiency.csv`
- `production_pipeline/output/analysis/results_rebalancing.csv`
- `production_pipeline/output/analysis/results_rebalancing_forecast.csv`
- `docs/report_workspace/chapter-03-methodology/images/`
- `docs/report_workspace/chapter-05-results-and-analysis/images/`
- `docs/report_workspace/chapter-06-portfolio-tilt-and-rebalancing-application/images/`
- `docs/report_workspace/chapter-07-discussion/images/`
- `docs/report_workspace/graph_explanations/`

## 9. Bottom line

The report is not yet perfectly aligned with the current project in four places:

- the examiner-driven shift from an unsupported total-flow equation to a three-fund index target
- methodology scope, especially any LSTM or hybrid wording versus the actual final pipeline
- stale numeric results and appendix tables from older runs
- too much important material still sitting in appendices instead of the main chapters

If those four areas are corrected, the report will align much more closely with the current project.
