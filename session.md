# Session Handoff

This file summarizes the current report-writing direction, the canonical pipeline choice, and the most important repo paths so a future chat can continue without rebuilding context from scratch.

## Current reporting decisions

- Ignore `2_midyear_model/` and `3_final_model/` in the final dissertation methodology.
- Use folders `4_claude_model/`, `5_claude_pipeline/`, and `6_cursor_model/` as the final methodological lineage.
- Treat `6_cursor_model/` as the **primary final pipeline** because it is the most KSE-30 specific.
- Treat `5_claude_pipeline/` as the upstream consolidated pipeline that prepares the cleaned inputs and provides a full end-to-end baseline.
- Treat `4_claude_model/` as the modular predecessor that explains how the methodology was separated into preprocessing, flow prediction, GARCH, efficiency, and rebalancing components.
- Always use `report_workspace/table-of-contents.txt` as the source of truth when writing report chapters.
- Chapter drafts must follow the TOC headings and numbering exactly.
- A dedicated root-level workspace now exists at `report_workspace/` for report writing assets.
- Chapter drafts should now be edited from `report_workspace/`, not from `0-docs/reports/`.

## Files created or updated in this session

- `report_workspace/chapter-03-methodology/chapter-03-methodology.txt`
  - Rewritten Chapter 3.
  - Now ignores folders 2 and 3.
  - Focuses on the merged workflow from folders 4 to 6.
  - Gives strongest emphasis to `6_cursor_model/`.

- `report_workspace/chapter-04-data-collection-and-processing/chapter-04-data-collection-and-processing.txt`
  - Drafted Chapter 4 using the exact TOC headings from `4.1` to `4.9`.
  - Focuses on the final data pipeline used by `5_claude_pipeline/` and `6_cursor_model/`.
  - Includes concrete dataset dimensions from the final KSE-30 workflow.

- `report_workspace/table-of-contents.txt`
  - Table of contents tailored to this project using `FYDP.pdf` as structure reference.
  - Contains placeholder page numbers.

- `0-docs/reports/chapter-05-results-and-analysis.txt`
  - Drafted Chapter 5 using the exact TOC headings from `5.1` to `5.6`.
  - Based mainly on `6_cursor_model/` results, with limited upstream support where descriptive context was needed.
  - Includes actual KSE-30 forecasting, GARCH/EGARCH, VaR, efficiency, and EDA findings.

- `0-docs/reports/chapter-06-portfolio-tilt-and-rebalancing-application.txt`
  - Drafted Chapter 6 using the exact TOC headings from `6.1` to `6.9`.
  - Based on `6_cursor_model/results_rebalancing.csv` and `results_rebalancing_forecast.csv`.
  - Includes the final forward rebalancing interpretation for KSE-30 constituents.

- `report_workspace/chapter-07-discussion/chapter-07-discussion.txt`
  - Drafted Chapter 7 using the exact TOC headings from `7.1` to `7.7`.
  - Interprets the final forecasting, efficiency, volatility, and rebalancing findings.

- `report_workspace/chapter-08-conclusion-and-recommendations/chapter-08-conclusion-and-recommendations.txt`
  - Drafted Chapter 8 using the exact TOC headings from `8.1` to `8.4`.
  - Summarizes the final study, contributions, recommendations, and future research directions.

- `report_workspace/`
  - New root-level report workspace.
  - Contains chapter folders, chapter-specific image folders, the moved TOC, and a copied `session.md`.

## Current Chapter 3 stance

The final Chapter 3 should say:

1. The final methodology is a merged framework from `4_claude_model` -> `5_claude_pipeline` -> `6_cursor_model`.
2. Folder 6 is the main empirical methodology for the dissertation.
3. Folder 5 is the upstream consolidated pipeline and folder 4 is the modular methodological base.
4. The study focuses on:
   - aggregate KSE-30 related fund flows,
   - KSE-30 volatility using GARCH and EGARCH,
   - KSE-30 market efficiency using runs, variance ratio, Ljung-Box, and Hurst,
   - KSE-30 rebalancing prediction using Ridge, Logistic, and Random Forest models.

## Canonical workflow to describe in the report

### Stage 1: Modular logic in `4_claude_model/`

- `nb0_preprocessing.py`
  - builds `daily_master.csv`, `monthly_master.csv`, and cleaned stock outputs
- `nb2_fund_flow_prediction.py`
  - aggregate and fund-level flow forecasting
- `nb3_garch_volatility.py`
  - GARCH and EGARCH estimation
- `nb5_market_efficiency.py`
  - runs test, variance ratio, Ljung-Box, Hurst
- `nb4b_rebalancing_prediction.py`
  - KSE-30 weight and inclusion prediction

### Stage 2: Consolidated workflow in `5_claude_pipeline/`

- `pipeline.py`
  - one-file end-to-end baseline
  - creates:
    - `daily_master.csv`
    - `monthly_master.csv`
    - `kse30_stocks_clean.csv`
    - `results_fund_flow.csv`
    - `results_garch.csv`
    - `results_efficiency.csv`
    - `results_rebalancing.csv`
    - `results_rebalancing_forecast.csv`

### Stage 3: Final KSE-30 specific workflow in `6_cursor_model/`

- `pipeline.py`
  - reads upstream cleaned data from `5_claude_pipeline/`
  - focuses on aggregate KSE-30 sector flow
  - estimates KSE-30 volatility and efficiency directly at index level
  - predicts KSE-30 rebalancing outcomes

## Most important paths for the report

### Chapter drafts

- `report_workspace/chapter-03-methodology/chapter-03-methodology.txt`
- `report_workspace/chapter-04-data-collection-and-processing/chapter-04-data-collection-and-processing.txt`
- `report_workspace/chapter-05-results-and-analysis/chapter-05-results-and-analysis.txt`
- `report_workspace/chapter-06-portfolio-tilt-and-rebalancing-application/chapter-06-portfolio-tilt-and-rebalancing-application.txt`
- `report_workspace/chapter-07-discussion/chapter-07-discussion.txt`
- `report_workspace/chapter-08-conclusion-and-recommendations/chapter-08-conclusion-and-recommendations.txt`
- `report_workspace/table-of-contents.txt`
- `report_workspace/session.md`

### Reference PDFs

- `0-docs/reports/FYDP.pdf`
- `0-docs/reports/FYP Report (Analyzing Mutual Funds).pdf`

### Folder summaries

- `mds/4_claude_model.md`
- `mds/5_claude_pipeline.md`
- `mds/6_cursor_model.md`

### Main results source for KSE-30 focused writing

- `6_cursor_model/results_fund_flow.csv`
- `6_cursor_model/results_garch.csv`
- `6_cursor_model/results_efficiency.csv`
- `6_cursor_model/results_rebalancing.csv`
- `6_cursor_model/results_rebalancing_forecast.csv`

### Useful figures from folder 6

- `6_cursor_model/figures/eda/E05_monthly_correlation.png`
- `6_cursor_model/figures/fund_flow/FF01_total_flow_predictions.png`
- `6_cursor_model/figures/fund_flow/FF02_granger.png`
- `6_cursor_model/figures/garch/G01_returns_and_vol.png`
- `6_cursor_model/figures/garch/G02_var_backtest.png`
- `6_cursor_model/figures/efficiency/EF01_acf.png`
- `6_cursor_model/figures/efficiency/EF02_variance_ratio.png`
- `6_cursor_model/figures/rebalancing/R01_retention_probability.png`
- `6_cursor_model/figures/rebalancing/R02_feature_importances.png`
- `6_cursor_model/figures/rebalancing/R03_weight_scatter.png`
- `6_cursor_model/figures/rebalancing/R04_weight_changes.png`
- `6_cursor_model/figures/summary/SUMMARY_dashboard.png`

## Model summary to keep consistent in future chats

### Fund-flow forecasting

- Naive benchmark
- custom ARIMAX(1,0,1)-style model
- custom VAR(1)
- Granger causality tests

### Volatility

- GARCH(1,1)
- EGARCH(1,1)
- VaR backtesting

### Market efficiency

- Runs test
- Variance ratio test
- Ljung-Box style autocorrelation test
- Hurst exponent

### Rebalancing

- Ridge regression for weight prediction
- RandomForestRegressor for weight prediction
- LogisticRegression for inclusion prediction
- RandomForestClassifier for inclusion prediction

## Important writing guidance

- Keep the report centered on **KSE-30**, not generic mutual funds.
- Do not describe folders 2 and 3 as part of the final methodology.
- If older experiments are ever mentioned, label them only as historical or exploratory and not part of the adopted final method.
- In the methodology chapter, prefer folder 6 figures over folder 4 or 5 figures when the same concept exists in multiple places.
- Treat `results_*.csv` files as the authoritative numeric source for report tables and claims.
- Before writing any chapter, first check `report_workspace/table-of-contents.txt` and mirror its exact section names.
- If a chapter draft already exists but its headings differ from the TOC, update the draft to match the TOC before adding more content.
- Use the chapter-specific `images/` folders under `report_workspace/` when assembling the final document.

## Recommended next tasks in a future chat

1. Update the table of contents if the chapter order changes.
2. Review Chapters 7 and 8 against department style and clean wording if needed.
3. Extract exact values from `6_cursor_model/results_*.csv` into final report tables.
4. Draft figure captions for the copied chapter image folders.
5. Assemble the final Word/PDF chapter order using the files inside `report_workspace/`.

## Suggested prompt for the next chat

Use `session.md`, `report_workspace/table-of-contents.txt`, the chapter drafts inside `report_workspace/`, and `0-docs/reports/FYDP.pdf` as reference. Continue improving the report using the final workflow from `4_claude_model`, `5_claude_pipeline`, and mainly `6_cursor_model`.
