# Required Changes for `FYP REPORT FINAL.pdf`

This file lists the changes needed to make `docs/reports/root_exports/FYP REPORT FINAL.pdf` fully align with the current project implementation and the regenerated outputs from the production pipeline.

Reference run used for alignment:

- command: `python production_pipeline/run_all.py`
- run date: 2026-06-12
- canonical outputs folder: `production_pipeline/output/analysis/`
- canonical report asset folder: `docs/report_workspace/`

## 1. Core framing changes

### 1.1 Stop presenting the target as an official observed KSE-30 fund-flow series

The report should consistently say that the flow target is a **proxy aggregate sector-flow series** constructed from three KSE-30-related funds:

- `AKD`
- `NBP`
- `NTI`

Use wording such as:

> The study does not use an official published market-wide KSE-30 net-flow series. Instead, it constructs a proxy aggregate sector-flow measure from return-adjusted changes in AUM/NAV for AKD, NBP, and NTI.

Avoid wording such as:

- "official KSE-30 fund flow"
- "observed market-wide KSE-30 net flow"
- "direct KSE-30 flow data"

This change is needed in:

- Executive Summary
- Chapter 1 introduction and objectives
- Chapter 3 methodology
- Chapter 4 results captions and interpretation
- Chapter 6 discussion
- Appendix A notes

### 1.2 Clarify the exact fund-flow equation

The academically stronger form to report is:

`DollarNetFlow_(i,t) = TNA_(i,t) - TNA_(i,t-1) * (1 + R_(i,t))`

The implemented project approximation is:

`flow_t = AUM_t - AUM_(t-1) * (NAV_t / NAV_(t-1))`

The report should explicitly say that:

- the literature-standard form is return-adjusted asset growth
- the implemented form uses monthly NAV-based return as a practical proxy
- this is acceptable as a proxy but should not be described as the strongest possible mutual-fund flow specification

This is especially important in Chapter 3.

## 2. Methodology changes

### 2.1 Re-scope the modeling section to match the actual final pipeline

The report TOC shows:

- `3.9.5 Long Short-Term Memory (LSTM) Model`
- `3.9.7 Hybrid Forecasting Framework`

The current final executable pipeline does **not** run LSTM in the production workflow. The final retained production workflow is centered on:

- ARIMAX-style fund-flow forecasting
- VAR(1)
- GARCH / EGARCH
- market-efficiency tests
- ridge / logistic / random forest comparisons for rebalancing

Required change:

- either remove the LSTM subsection entirely
- or keep it only as literature/background and explicitly state that it is **not part of the final implemented production pipeline**

Do the same for any "hybrid framework" claims if they imply a deployed combined model that the final pipeline does not actually execute.

### 2.2 Update data-splitting wording

The report currently says the data was divided into training, validation, and testing subsets in general terms.

The final run actually uses:

- fund-flow forecasting: `34` training months and `25` test months
- rebalancing panel: `364` training observations and `58` test observations

Required change:

- replace vague train/validation/test language with the exact split logic used in the final pipeline
- if validation was not actually used as a separate final holdout in the retained code path, do not claim a formal three-way split

### 2.3 Update the final sample coverage

The final regenerated outputs are:

- `daily_master.csv`: `1300` rows × `12` columns, `2021-01-04` to `2026-04-30`
- `monthly_master.csv`: `59` rows × `38` columns, `2021-03-31` to `2026-01-30`
- cleaned stock panel: `46,560` rows, `59` symbols, `2020-01-01` to `2026-04-30`

Required change:

- revise all methodology/result sections that mention sample size, date span, or variable count so they match these final outputs

## 3. Results section changes

All numeric tables in the report should be refreshed from the latest generated outputs, not left at older values.

### 3.1 Update fund-flow forecasting results

Current final run:

- Naive (RW): `RMSE 83.35`, `MAE 51.36`, `R² -1.4196`, `DirAcc 37.5%`
- ARIMAX(1,0,1): `RMSE 58.00`, `MAE 36.52`, `R² -0.1716`, `DirAcc 70.8%`
- VAR(1): `RMSE 62.54`, `MAE 38.80`, `R² -0.3622`, `DirAcc 75.0%`

Required change:

- replace older table values in Chapter 4 and Appendix A with these rerun values
- note that ARIMAX is best on RMSE/MAE, while VAR ties or exceeds on directional accuracy
- keep the interpretation that **directional utility is more meaningful than point-fit alone** because R² remains negative out-of-sample

### 3.2 Update stationarity and Granger discussion

Current run summary:

- `total_fund_flow`: stationary, `p = 0.0000`
- `interest_rate_end`: non-stationary, `p = 0.2420`
- `cpi_yoy_end`: non-stationary, `p = 0.4012`
- `oil_return_monthly`: stationary, `p = 0.0000`
- `usdpkr_return_monthly`: stationary, `p = 0.0000`

Current Granger results:

- IR -> flow: `p = 0.6405`
- CPI -> flow: `p = 0.0532`
- Oil -> flow: `p = 0.7365`
- USD/PKR -> flow: `p = 0.6365`

Required change:

- update the text to say CPI is borderline at the 10% level, not conventionally significant at 5%
- avoid implying that any macro driver strongly Granger-causes the flow series

### 3.3 Update volatility modeling results

Current final run:

- preferred model: `EGARCH(1,1)`
- EGARCH AIC: `4209.7`
- GARCH persistence: `0.9667`

Required change:

- ensure Chapter 4 says EGARCH is the preferred final specification by AIC
- keep the leverage/asymmetry discussion only if tied to the EGARCH outcome

### 3.4 Update market-efficiency results

Current final run:

- Runs test Z: `-1.9106`
- Runs p-value: `0.0561`
- Variance Ratio VR(2): `1.0069`
- Variance Ratio p-value: `0.8752`
- Ljung-Box p-value: `0.0000`
- Hurst exponent: `0.6559`

Required change:

- present the conclusion as **mixed / borderline evidence**
- do not call the market fully efficient
- do not call it fully inefficient either
- explicitly explain that runs and variance-ratio results are closer to efficiency, while Ljung-Box and Hurst imply persistence / serial dependence

### 3.5 Update rebalancing results

Current final run:

- detected rebalancing dates: `17`
- training panel: `422` rows
- effective windows: `16`
- symbols in panel: `47`
- retained rate: `91.9%`

Weight prediction:

- Naive: `RMSE 0.5595`, `MAE 0.2541`, `R² 0.9677`
- Ridge: `RMSE 0.5590`, `MAE 0.2963`, `R² 0.9678`
- Random Forest: `RMSE 0.6905`, `MAE 0.3607`, `R² 0.9509`

Inclusion prediction:

- Naive accuracy: `0.9655`
- Logistic accuracy: `0.9655`, `AUC 0.8214`
- Random Forest accuracy: `0.9655`, `AUC 0.5893`

Required change:

- update all Chapter 5 and Appendix tables with these values
- if the report currently implies Random Forest is the final winning rebalancing model, revise that
- the final evidence supports:
  - ridge / naive strength for weight prediction
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

- update the narrative in Chapter 5 and Chapter 6 so that all forecast examples use the regenerated current forecast table
- do not leave older stock examples if they differ from the latest forecast output

## 4. Figure and table alignment changes

### 4.1 Replace stale figures with regenerated current figures

The report should use figures copied/generated from:

- `production_pipeline/output/analysis/figures/`
- `docs/report_workspace/chapter-03-methodology/images/`
- `docs/report_workspace/chapter-04-data-collection-and-processing/images/`
- `docs/report_workspace/chapter-05-results-and-analysis/images/`
- `docs/report_workspace/chapter-06-portfolio-tilt-and-rebalancing-application/images/`
- `docs/report_workspace/chapter-07-discussion/images/`
- `docs/report_workspace/chapter-08-conclusion-and-recommendations/images/`

Required change:

- replace any report figure that was copied from an older run if its numbers no longer match the current result tables

### 4.2 Fix Appendix A table values

The extracted report currently contains older Appendix A values for the fund-flow models.

Required change:

- refresh Appendix A.1 from `production_pipeline/output/analysis/results_fund_flow.csv`
- refresh volatility, efficiency, and rebalancing appendices from the latest CSV outputs

### 4.3 Check figure numbering against actual files

The report’s numbered figures should map to the regenerated assets as follows:

- Chapter 3 methodology figures: `C3_*`
- Chapter 4/5 result figures: `E*`, `FF*`, `G*`, `EF*`
- Chapter 5 rebalancing framework/results: `C6_*`, `R*`
- Chapter 6/7 discussion figures: `C7_*`
- Chapter 7/8 conclusion figure: `C8_*`

Required change:

- ensure no stale caption remains attached to the wrong regenerated image

## 5. Terminology and wording fixes

### 5.1 Use one consistent fund name convention

The report currently mixes:

- `NTI`
- `NIT`
- `NTI/NIT`

Required change:

- pick one convention and use it consistently everywhere
- if the actual fund entity is NIT but the project variable name is NTI, explain that once and then stay consistent

### 5.2 Fix over-claiming about machine learning

The Executive Summary currently says machine learning models forecast index fund flows and determine optimal stock weight allocations.

Required change:

- revise that statement so it distinguishes:
  - econometric fund-flow forecasting: ARIMAX / VAR
  - volatility modeling: GARCH / EGARCH
  - rebalancing prediction: ridge / logistic / random forest comparison

This matters because the final pipeline is not purely "machine learning driven."

### 5.3 Replace "correlation and regression" if too generic

The final project is much more specific than generic correlation/regression language.

Required change:

- update broad summary wording so it explicitly mentions:
  - stationarity testing
  - Granger causality
  - ARIMAX
  - VAR
  - GARCH / EGARCH
  - market-efficiency diagnostics
  - rebalancing prediction

## 6. Structural and formatting corrections

### 6.1 Fix appendix lettering

The TOC shows:

- Appendix A
- Appendix B
- Appendix D
- Appendix E

Required change:

- either add the missing Appendix C
- or relabel later appendices so the sequence is continuous

### 6.2 Check abbreviation table consistency

Required change:

- fix duplicate or inconsistent abbreviations for `NIT` / `NTI`
- remove abbreviations that are not actually used in the final implemented pipeline if they are only leftovers from literature discussion

### 6.3 Fix OCR/formatting artifacts in the editable source

The extracted text shows spacing and word-break artifacts such as:

- `machine lear ning`
- `fin ancial`
- `provi ding`
- `th eoretical`

Required change:

- clean these formatting issues in the editable DOCX/source file before exporting the final PDF

## 7. Recommended chapter-by-chapter corrections

### Executive Summary

- reframe the target as a proxy aggregate sector-flow series
- remove any implication that the final system directly uses LSTM in production
- distinguish econometric forecasting from ML-based rebalancing support

### Chapter 3: Methodology

- correct the fund-flow equation and explain the NAV-based approximation
- remove or downgrade LSTM/hybrid claims unless they are explicitly labeled as non-final
- update sample coverage and split logic
- align variables with the actual final master datasets

### Chapter 4: Results and Analysis

- replace all stale numbers with current rerun values
- update stationarity and Granger text
- state EGARCH is preferred by AIC
- keep market-efficiency interpretation mixed/borderline

### Chapter 5: Portfolio Tilt and Rebalancing Application

- refresh panel counts and test metrics
- update model comparison narrative so it matches ridge/logistic performance
- refresh the forward forecast examples and high-risk names

### Chapter 6: Discussion

- align discussion claims with current results rather than earlier runs
- avoid overstating forecast strength where R² is negative but direction is useful

### Chapter 7: Conclusion and Recommendations

- make sure the conclusion reflects a decision-support system, not a live deployable trading system
- retain the transaction-cost and implementation-limitations disclaimer

### Appendices

- regenerate Appendix A tables from current CSV outputs
- ensure the figure appendix matches the regenerated figure copies in `docs/report_workspace/`

## 8. Files to use when revising the report

Use these outputs as the source of truth during report correction:

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

The report is structurally strong, but it is not yet perfectly aligned with the current project in three places:

- terminology around what the fund-flow target really is
- methodology scope, especially LSTM/hybrid wording versus the actual final pipeline
- stale numeric results and appendix tables from older runs

If those three areas are corrected, the report will match the current project much more closely.
