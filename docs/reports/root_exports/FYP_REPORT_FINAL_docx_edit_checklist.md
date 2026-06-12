# DOCX Edit Checklist for `FYP REPORT FINAL (1).docx`

This checklist is the practical follow-up to `FYP_REPORT_FINAL_alignment_changes.md`.

It is designed for editing:

- `docs/reports/root_exports/FYP REPORT FINAL (1).docx`

Inspection basis:

- extracted text from the PDF
- DOCX paragraph map in `docs/reports/root_exports/_docx_inspect/document_paragraphs.txt`
- DOCX embedded-image manifest in `docs/reports/root_exports/_docx_inspect/image_manifest.txt`
- current generated figures in `production_pipeline/output/analysis/` and `docs/report_workspace/`

## 1. Highest-priority text edits

### 1.1 Executive Summary

Current problem:

- it overstates the project as predicting direct "index fund flows"
- it describes the workflow too generically
- it implies a stronger ML-centered forecasting stack than the final pipeline actually uses

Replace the framing so it says:

- the project constructs a three-fund sector proxy index
- the index is based on `AKD`, `NBP`, and `NTI/NIT`
- the index was adopted after examiner feedback objected to the unsupported direct total-flow equation
- the final pipeline combines:
  - ARIMAX / VAR for monthly target forecasting
  - GARCH / EGARCH for volatility
  - market-efficiency tests
  - logistic / ridge / random forest comparisons for rebalancing

### 1.2 Methodology chapter

Edit these points:

- remove or downgrade `LSTM` as a final implemented model
- remove or downgrade any "hybrid framework" language that implies it is in the retained production workflow
- replace generic training/validation/testing wording with the actual final split logic
- replace the old total-flow headline equation with the new three-fund index equations

Add a subsection heading if needed:

- `Construction of Three-Fund Sector Proxy Index`

### 1.3 Results chapter

Refresh stale metrics using:

- `production_pipeline/output/analysis/results_fund_flow.csv`
- `production_pipeline/output/analysis/results_garch.csv`
- `production_pipeline/output/analysis/results_efficiency.csv`
- `production_pipeline/output/analysis/results_rebalancing.csv`
- `production_pipeline/output/analysis/results_rebalancing_forecast.csv`

### 1.4 Terminology cleanup

Standardize:

- use "three-fund sector proxy index" as the main label
- use "composite fund index" or "sector proxy index" as acceptable short forms
- do not use "official KSE-30 flow" or "direct total fund flow" as the final target label
- fix `NIT` / `NTI` inconsistency
- keep market-efficiency interpretation as mixed or borderline, not absolute

## 2. Equations that must be changed in the DOCX

This is the most important examiner-driven technical edit.

### 2.1 Demote the old flow equation to literature background only

If the DOCX currently presents this or a similar form as the main study target:

`DollarNetFlow_(i,t) = TNA_(i,t) - TNA_(i,t-1) * (1 + R_(i,t))`

or

`FlowProxy_(i,t) = AUM_(i,t) - AUM_(i,t-1) * (NAV_(i,t) / NAV_(i,t-1))`

change the surrounding text so these appear only as:

- literature motivation
- constituent-level background
- an initial benchmark concept that was not retained as the final target

### 2.2 Insert the constituent return equation

Use:

`r_(i,t) = (NAV_(i,t) - NAV_(i,t-1)) / NAV_(i,t-1)`

Purpose:

- defines the monthly return for each constituent fund
- is needed before the normalized index construction

### 2.3 Insert the normalized fund-level index equation

Use:

`IndexLevel_(i,t) = 100 * NAV_(i,t) / NAV_(i,0)`

Purpose:

- rebases the three funds to a common starting level
- makes them directly aggregable into one composite index

### 2.4 Insert the composite index equation

If the dissertation uses equal weighting, use:

`FundIndex_t = (1/3) * [IndexLevel_(AKD,t) + IndexLevel_(NBP,t) + IndexLevel_(NTI,t)]`

If the dissertation uses fixed weights instead, use:

`FundIndex_t = sum_(i in {AKD,NBP,NTI}) w_i * IndexLevel_(i,t), where sum_i w_i = 1`

Important:

- only one of the above should remain in the final DOCX
- do not leave both unless one is explicitly labeled as an alternative specification

### 2.5 Insert the modeled target equation if forecasting uses index return

If the forecasting chapter models the composite return, add:

`FundIndexReturn_t = ln(FundIndex_t / FundIndex_(t-1))`

If the model instead uses simple percentage change, replace the log-return line with the exact implemented transformation and keep the variable name consistent everywhere.

### 2.6 Keep any constituent flow proxy as optional background only

If you want to preserve the AUM/NAV flow intuition, keep it in one short background paragraph only:

`FlowProxy_(i,t) = AUM_(i,t) - AUM_(i,t-1) * (NAV_(i,t) / NAV_(i,t-1))`

The text must state:

- this is a constituent-level proxy concept
- it is not the final dissertation target equation
- the final target is the composite three-fund index or its return

## 3. Main figure replacements in the DOCX

The DOCX embed map shows that the main analytical figures are already embedded as images inside the Word file. Replace those embedded images with the current generated files below.

### Chapter 3 figures

Existing DOCX image references:

- Figure 3.1 -> `media/image5.png`
- Figure 3.2 -> `media/image6.png`
- Figure 3.3 -> `media/image7.png`
- Figure 3.4 -> `media/image8.png`
- Figure 3.5 -> `media/image9.png`

Replace them with:

- Figure 3.1
  Use: `production_pipeline/output/analysis/figures/eda/E07_top_weights.png`

- Figure 3.2
  Use: `production_pipeline/output/analysis/figures/eda/E05_monthly_correlation.png`

- Figure 3.3
  Use: `docs/report_workspace/chapter-03-methodology/images/C3_EDA_04_return_distribution_and_qq.png`

- Figure 3.4
  Use: `docs/report_workspace/chapter-03-methodology/images/C3_EDA_ST_03_level_vs_return_examples.png`

- Figure 3.5
  Use: `docs/report_workspace/chapter-03-methodology/images/C3_EDA_ST_01_pvalue_heatmap.png`

### Chapter 4 figures

Current DOCX image references:

- Figure 4.1 -> `media/image10.png`
- Figure 4.2 -> `media/image11.png`
- Figure 4.3 -> `media/image12.png`
- Figure 4.4 -> `media/image13.png`
- Figure 4.5 -> `media/image14.png`
- Figure 4.6 -> `media/image15.png`
- Figure 4.7 -> `media/image16.png`
- Figure 4.8 -> `media/image17.png`

Replace them with:

- Figure 4.1
  Use: `docs/report_workspace/chapter-05-results-and-analysis/images/E06_index_cumulative_return.png`

- Figure 4.2
  Use: `docs/report_workspace/chapter-05-results-and-analysis/images/E01_aum_trend.png`

- Figure 4.3
  Use: `docs/report_workspace/chapter-05-results-and-analysis/images/E03_fund_flows.png`

- Figure 4.4
  Use: `docs/report_workspace/chapter-05-results-and-analysis/images/FF02_granger.png`

- Figure 4.5
  Use: `docs/report_workspace/chapter-05-results-and-analysis/images/FF01_total_flow_predictions.png`

- Figure 4.6
  Use: `docs/report_workspace/chapter-05-results-and-analysis/images/G01_returns_and_vol.png`

- Figure 4.7
  Use: `docs/report_workspace/chapter-05-results-and-analysis/images/G02_var_backtest.png`

- Figure 4.8
  Use: `docs/report_workspace/chapter-05-results-and-analysis/images/EF01_acf.png`

### Chapter 5 figures

Current DOCX image references:

- Figure 5.1 -> `media/image18.png`
- Figure 5.2 -> `media/image19.png`
- Figure 5.3 -> likely `media/image20.png`
- Figure 5.4 -> `media/image21.png`
- Figure 5.5 -> `media/image22.png`

Replace them with:

- Figure 5.1
  Use: `docs/report_workspace/chapter-06-portfolio-tilt-and-rebalancing-application/images/C6_01_rebalancing_framework.png`

- Figure 5.2
  Use: `docs/report_workspace/chapter-06-portfolio-tilt-and-rebalancing-application/images/R01_retention_probability.png`

- Figure 5.3
  Use: `docs/report_workspace/chapter-06-portfolio-tilt-and-rebalancing-application/images/R02_feature_importances.png`

- Figure 5.4
  Use: `docs/report_workspace/chapter-06-portfolio-tilt-and-rebalancing-application/images/R03_weight_scatter.png`

- Figure 5.5
  Use: `docs/report_workspace/chapter-06-portfolio-tilt-and-rebalancing-application/images/R04_weight_changes.png`

### Chapter 6 figures

Current DOCX image references:

- Figure 6.1 -> `media/image23.png`
- Figure 6.2 -> `media/image24.png`
- Figure 6.3 -> `media/image25.png`
- Figure 6.4 -> `media/image26.png`
- Figure 6.5 -> `media/image27.png`

Replace them with:

- Figure 6.1
  Use: `docs/report_workspace/chapter-07-discussion/images/C7_01_flow_model_scorecard.png`

- Figure 6.2
  Use: `docs/report_workspace/chapter-07-discussion/images/C7_02_efficiency_evidence_summary.png`

- Figure 6.3
  Use: `docs/report_workspace/chapter-07-discussion/images/C7_03_realized_volatility_regimes.png`

- Figure 6.4
  Use: `docs/report_workspace/chapter-07-discussion/images/C7_04_model_family_comparison.png`

- Figure 6.5
  Use: `docs/report_workspace/chapter-07-discussion/images/C7_05_rebalancing_risk_map_inbubble_labels.png`

## 4. Appendix figure replacements

Appendix figures identified from the DOCX:

- `media/image28.png` -> Figure B.2
- `media/image29.png` -> Figure B.8
- `media/image30.png` -> Figure B.9
- `media/image31.png` -> Figure B.12

Recommended replacements:

- Figure B.2
  Replace with: `docs/report_workspace/chapter-05-results-and-analysis/images/E02_nav_return_dist.png`

- Figure B.8
  Replace with: `production_pipeline/output/eda_raw/18_macro_cpi_levels.png`

- Figure B.9
  Replace with: `production_pipeline/output/eda_raw/19_macro_log_returns.png`

- Figure B.12
  Replace with: `docs/report_workspace/chapter-05-results-and-analysis/images/EF02_variance_ratio.png`

## 5. Appendix E must be revised heavily

The DOCX contains a large block of embedded JPEG screenshots:

- `media/image32.jpeg` through `media/image52.jpeg`

These appear around the old repository-structure appendix. That appendix is now stale because the repo has been restructured into:

- `production_pipeline/`
- `docs/`
- `markdown/`

Required action:

- remove or replace the old Appendix E repository screenshots
- rewrite the appendix so it reflects the current production-ready structure
- cut Appendix E aggressively if the department does not require screenshot-heavy implementation appendices

Recommended replacement content:

- one clean repository tree image showing:
  - `production_pipeline/`
  - `docs/`
  - `markdown/`
  - `README.md`
  - `PROJECT_REFERENCE.md`
- one workflow diagram showing:
  - `production_pipeline/data/raw`
  - `prepare_kse30_basic.py`
  - `pipeline.py`
  - `output/analysis`
  - `docs/report_workspace`
- one short table listing the `run_all.py` stages:
  - `prepare`
  - `pipeline`
  - `eda_raw`
  - `eda_master`
  - `stationarity`
  - `chapter3`
  - `report`
  - `risk_map`

If you do not want to rebuild all screenshots:

- replace the screenshot-heavy appendix with a concise text or table appendix describing the final production layout

## 6. Appendix reduction and chapter-promotion plan

The report should reduce appendix dependence and move core evidence into the main body.

### 6.1 Move these items into the main chapters

- move the final three-fund index equations into Chapter 3
- move the final forecasting comparison table into Chapter 4
- move the final stationarity summary and Granger summary into Chapter 4
- move the final efficiency summary table into Chapter 4
- move the final rebalancing comparison table into Chapter 5
- move the final forward-risk or forecast table into Chapter 5

### 6.2 Keep these in appendices only if space requires it

- raw output dumps
- alternate model specifications not discussed in the text
- supplementary figure panels
- implementation screenshots

### 6.3 Recommended appendix cuts

- remove repeated repository screenshots
- remove duplicated figure versions when one final figure is already used in the chapter
- compress long appendix narratives into short tables
- delete appendix content that is never referenced in the discussion or conclusion

## 7. Tables that must be manually refreshed in the DOCX

These require editing the Word table values directly.

### Table 4.1

Use current values:

- Naive (RW): `RMSE 83.35`, `MAE 51.36`, `R^2 -1.4196`, `DirAcc 37.5%`
- ARIMAX(1,0,1): `RMSE 58.00`, `MAE 36.52`, `R^2 -0.1716`, `DirAcc 70.8%`
- VAR(1): `RMSE 62.54`, `MAE 38.80`, `R^2 -0.3622`, `DirAcc 75.0%`

### Table 4.2

Update to the current GARCH/EGARCH values from:

- `production_pipeline/output/analysis/results_garch.csv`

Key interpretation:

- EGARCH is preferred by lower AIC
- GARCH persistence is `0.9667`

### Table 4.3

Update to current efficiency values:

- Runs Z `-1.9106`
- Runs p `0.0561`
- VR(2) `1.0069`
- VR p `0.8752`
- Ljung-Box p `0.0000`
- Hurst `0.6559`

### Rebalancing chapter tables and text

Update to:

- panel size `422`
- windows `16`
- symbols `47`
- retained rate `91.9%`
- logistic AUC `0.8214`
- ridge weight prediction `R^2 0.9678`

## 8. Paragraphs in the DOCX that deserve direct rewriting

Use the paragraph map file to locate these areas quickly.

### Paragraphs 101-104

Rewrite the Executive Summary to remove overclaiming and align it with the final workflow and the examiner-driven index framing.

### Paragraph 606

Current issue:

- it uses generic split language
- it likely sits near the methodology area where the old target equation appears

Change:

- replace with the actual retained split approach
- insert the three-fund index-construction narrative around this methodology area

### Paragraphs 739-744

Current issue:

- stale Granger values and stale ARIMAX text

Change:

- replace with current output values
- update the CPI significance wording to borderline rather than strongly significant
- make sure the paragraph says ARIMAX and VAR are forecasting the composite three-fund target

### Paragraphs 824-845

Current issue:

- rebalancing prose must be checked against current forecast outputs and current model ranking

Change:

- keep logistic as stronger classification evidence
- keep ridge or naive as the strongest weight-prediction story
- refresh any example stock names if the old text uses names not present in the current forecast

### Paragraphs 851-857

Current issue:

- discussion chapter may still use generic summary language and may not reflect the rerun values exactly

Change:

- keep emphasis on directional utility over point fit
- preserve mixed-efficiency interpretation
- add one sentence explaining that the three-fund index was adopted to improve methodological defensibility after examiner feedback

### Paragraph 866

Current issue:

- says ML is used because the richer rebalancing dataset supports it

This is acceptable, but make sure it does not imply LSTM was part of the final flow-forecasting implementation.

## 9. Quick replacement order for the editor

1. Fix the Executive Summary wording.
2. Replace the unsupported total-flow equation with the three-fund index equations.
3. Move the key equation block and key result tables into the main chapters.
4. Fix Chapter 3 methodology scope and remove or downgrade LSTM wording.
5. Replace Figures 3.1-6.5 using the map above.
6. Refresh Tables 4.1-4.3 and the rebalancing metrics.
7. Refresh Appendix B figures.
8. Rewrite, replace, or heavily cut Appendix E.
9. Export a new PDF and cross-check it against:
   - `FYP_REPORT_FINAL_alignment_changes.md`
   - `production_pipeline/output/analysis/`

## 10. Bottom line

If you only have time for the most important DOCX fixes, do these first:

- replace the unsupported total-flow equation with the final three-fund index equations
- correct the target-series framing everywhere
- move the important equations and key summary tables into the main chapters
- remove LSTM as a final implemented model unless explicitly labeled as background only
- refresh Tables 4.1 to 4.3
- replace Figures 4.4 to 6.5
- replace or rewrite Appendix E because it now documents an obsolete repository structure
