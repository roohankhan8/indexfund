# DOCX Edit Checklist for `FYP REPORT FINAL (1).docx`

This checklist is the practical follow-up to `FYP_REPORT_FINAL_alignment_changes.md`.

It is designed for editing the source file:

- `docs/reports/root_exports/FYP REPORT FINAL (1).docx`

Inspection basis:

- extracted text from the PDF
- DOCX paragraph map in `docs/reports/root_exports/_docx_inspect/document_paragraphs.txt`
- DOCX embedded-image manifest in `docs/reports/root_exports/_docx_inspect/image_manifest.txt`
- current generated figures in `production_pipeline/output/analysis/` and `docs/report_workspace/`

## 1. Highest-priority text edits

These should be fixed before touching formatting.

### 1.1 Executive Summary

Current problem:

- it overstates the project as predicting "index fund flows" in a direct sense
- it describes the final workflow too generically
- it implies a stronger machine-learning-centered flow model than the current pipeline actually uses

Replace the current framing with wording that says:

- the project constructs a **proxy aggregate sector-flow series**
- the flow series is based on `AKD`, `NBP`, and `NTI`
- the final implemented pipeline combines:
  - ARIMAX / VAR for monthly flow prediction
  - GARCH / EGARCH for volatility
  - market-efficiency tests
  - logistic / ridge / random forest comparisons for rebalancing

### 1.2 Methodology chapter

Edit these points:

- remove or downgrade `LSTM` as a final implemented model
- remove or downgrade any "hybrid framework" language that implies it is in the final retained production workflow
- replace generic "training, validation, and testing" wording with the actual final split logic
- insert the corrected fund-flow equation language from `FYP_REPORT_FINAL_alignment_changes.md`

### 1.3 Results chapter

Refresh all stale metrics using:

- `production_pipeline/output/analysis/results_fund_flow.csv`
- `production_pipeline/output/analysis/results_garch.csv`
- `production_pipeline/output/analysis/results_efficiency.csv`
- `production_pipeline/output/analysis/results_rebalancing.csv`
- `production_pipeline/output/analysis/results_rebalancing_forecast.csv`

### 1.4 Terminology cleanup

Standardize:

- use "proxy aggregate sector-flow series" instead of "official KSE-30 flow"
- fix `NIT` / `NTI` inconsistency
- keep market-efficiency interpretation as "mixed" or "borderline", not absolute

## 2. Main figure replacements in the DOCX

The DOCX embed map shows that the main analytical figures are already embedded as images inside the Word file. Replace those embedded images with the current generated files below.

### Chapter 3 figures

These are the existing DOCX image references:

- Figure 3.1 -> `media/image5.png`
- Figure 3.2 -> `media/image6.png`
- Figure 3.3 -> `media/image7.png`
- Figure 3.4 -> `media/image8.png`
- Figure 3.5 -> `media/image9.png`

Replace them with:

- Figure 3.1
  Use: `production_pipeline/output/analysis/figures/eda/E07_top_weights.png`
  Reason: this is the current top-weight figure and matches the "top weighted KSE-30 stocks" intent better than the old embedded chart.

- Figure 3.2
  Use: `production_pipeline/output/analysis/figures/eda/E05_monthly_correlation.png`
  Reason: this is the current monthly-correlation visualization aligned with the final pipeline.

- Figure 3.3
  Use: `docs/report_workspace/chapter-03-methodology/images/C3_EDA_04_return_distribution_and_qq.png`
  Reason: this is the current report-ready return distribution and Q-Q figure.

- Figure 3.4
  Use: `docs/report_workspace/chapter-03-methodology/images/C3_EDA_ST_03_level_vs_return_examples.png`
  Reason: this is the current stationarity transformation illustration.

- Figure 3.5
  Use: `docs/report_workspace/chapter-03-methodology/images/C3_EDA_ST_01_pvalue_heatmap.png`
  Reason: this is the current ADF p-value heatmap.

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
  Reason: this is the cleaner annotated version and is better than the plain unlabeled map.

## 3. Appendix figure replacements

The DOCX also embeds appendix figures that should be checked against the current output set.

### Appendix B images identified from the DOCX

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

## 4. Appendix E must be revised heavily

This is the biggest DOCX-specific issue revealed by the image inspection.

The DOCX contains a large block of embedded JPEG screenshots:

- `media/image32.jpeg` through `media/image52.jpeg`

These appear in the section around:

- paragraph `1412`
- `Table E.2: Internal structure of final pipeline — eight analytical sections`
- `E.4 Report Workspace Role`

Why this is a problem:

- these screenshots document the **old repository structure**
- the repo has now been restructured into:
  - `production_pipeline/`
  - `docs/`
  - `markdown/`
- therefore Appendix E is now stale even if the analytical chapters are corrected

Required action:

- remove or replace all old Appendix E repository screenshots
- rewrite the appendix so it reflects the **current production-ready structure**

Recommended replacement content:

- one clean repository tree screenshot showing:
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

If you do not want to rebuild all those appendix screenshots, the safer option is:

- replace the screenshot-heavy appendix with a concise text/table appendix describing the final production layout

## 5. Tables that must be manually refreshed in the DOCX

These are not image swaps; they require editing the Word table values directly.

### Table 4.1

Use current values:

- Naive (RW): `RMSE 83.35`, `MAE 51.36`, `R² -1.4196`, `DirAcc 37.5%`
- ARIMAX(1,0,1): `RMSE 58.00`, `MAE 36.52`, `R² -0.1716`, `DirAcc 70.8%`
- VAR(1): `RMSE 62.54`, `MAE 38.80`, `R² -0.3622`, `DirAcc 75.0%`

### Table 4.2

Update to the current GARCH/EGARCH values from:

- `production_pipeline/output/analysis/results_garch.csv`

Key interpretation to preserve:

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

### Rebalancing chapter tables/text

Update to:

- panel size `422`
- windows `16`
- symbols `47`
- retained rate `91.9%`
- logistic AUC `0.8214`
- ridge weight prediction `R² 0.9678`

## 6. Paragraphs in the DOCX that deserve direct rewriting

Use the paragraph map file to locate these areas quickly.

### Paragraphs 101–104

Rewrite the Executive Summary to remove overclaiming and align with the final workflow.

### Paragraph 606

Current issue:

- claims train/validation/test in generic terms

Change:

- replace with the actual final split approach used by the retained pipeline

### Paragraphs 739–744

Current issue:

- stale Granger values and stale ARIMAX text

Change:

- replace with current output values and update the CPI significance wording to borderline rather than strongly significant

### Paragraphs 824–845

Current issue:

- rebalancing prose must be checked against current forecast outputs and current model ranking

Change:

- keep logistic as stronger classification evidence
- keep ridge/naive as strongest weight-prediction story
- refresh any example stock names if the old text uses names not present in the current top-risk list

### Paragraphs 851–857

Current issue:

- discussion chapter still uses a generic summary tone and may not reflect the rerun values exactly

Change:

- keep emphasis on directional utility over point-fit
- preserve mixed-efficiency interpretation

### Paragraph 866

Current issue:

- says ML is used because the richer rebalancing dataset supports it

This is acceptable, but make sure it does not imply LSTM was part of the final flow-forecasting implementation.

## 7. Quick replacement order for the editor

Recommended edit sequence inside Word:

1. Fix Executive Summary wording.
2. Fix Chapter 3 methodology scope and fund-flow equation.
3. Replace Figures 3.1–6.5 using the replacement map above.
4. Refresh Tables 4.1–4.3 and rebalancing metrics.
5. Refresh Appendix B figures.
6. Rewrite or replace Appendix E screenshots.
7. Export a new PDF and cross-check it against:
   - `FYP_REPORT_FINAL_alignment_changes.md`
   - `production_pipeline/output/analysis/`

## 8. Bottom line

If you only have time for the most important DOCX fixes, do these first:

- correct the flow-series framing
- remove LSTM as a final implemented model unless explicitly labeled as background only
- refresh Tables 4.1 to 4.3
- replace Figures 4.4 to 6.5
- replace or rewrite Appendix E because it now documents an obsolete repository structure
