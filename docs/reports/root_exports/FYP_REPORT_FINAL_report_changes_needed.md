# FYP Report Review: Changes Still Needed After Latest Update

## 1. Executive Summary is still not updated

The Executive Summary is still the old proposal-style version.

It still says things like:

- "The data will be cleaned..."
- "Machine learning models will then be implemented..."
- "The project will generate insights..."

It also still overstates the workflow as broadly machine-learning-driven.

Changes needed:

- rewrite the Executive Summary in past tense
- describe the work as completed, not planned
- explain the final implemented workflow clearly:
  - aggregate fund-flow forecasting with ARIMAX and VAR
  - volatility modelling with GARCH and EGARCH
  - market-efficiency diagnostics
  - stock-level rebalancing prediction with Ridge, Logistic Regression, and Random Forest comparisons
- avoid framing the whole dissertation as if ML was the main final engine

## 2. Methodology improved, but it is still internally inconsistent

The new methodology subsections are a real improvement, but several problems remain.

### 2.1 Numbering is broken

The TOC now shows:

- `3.9.3 ARIMA Model`
- `3.9.4 GARCH Family Models`
- `3.9.6 Random Forest Model`

There is no visible `3.9.5`, which makes the numbering look unfinished.

Changes needed:

- renumber the subsection sequence cleanly
- regenerate the TOC after fixing the headings

### 2.2 LSTM was removed from the methodology section, but not from the rest of the report

The update removed `3.9.5 LSTM Model` from the methodology TOC, which is good.

But the report still keeps `LSTM` in:

- abbreviations
- literature review
- references
- future research wording
- front framing that still implies broad ML implementation

That is acceptable in the literature review and future research, but not if the report still gives the impression that LSTM was part of the final implemented pipeline.

Changes needed:

- keep LSTM only as literature/background or future-work material
- make sure no chapter implies LSTM was one of the retained implemented final models

### 2.3 The target-series framing is still too loose

The updated methodology now explicitly discusses fund-level flow construction and composite fund flow, which is better.

But the report should still tighten the wording:

- make clear that `total_fund_flow` is the implemented three-fund aggregate proxy
- avoid wording that sounds like an official directly observed market-wide KSE-30 net-flow series

Use wording like:

- "aggregate three-fund KSE-30-related proxy"
- "composite flow series constructed from AKD, NBP, and NTI"

## 3. Main Chapter 4 is only partially updated

The biggest remaining problem is that some tables are current, but the nearby prose is still old.

### 3.1 Table 4.1 is current, but the explanatory paragraph below it is stale

The main table now shows current values:

- Naive: `RMSE 83.35`, `MAE 51.36`, `R2 -1.4196`, `DirAcc 37.5%`
- ARIMAX: `RMSE 58.00`, `MAE 36.52`, `R2 -0.1716`, `DirAcc 70.8%`
- VAR: `RMSE 62.54`, `MAE 38.80`, `R2 -0.3622`, `DirAcc 75.0%`

But the paragraph under `4.3.2 ARIMAX and VAR Results` still says older numbers:

- ARIMAX directional accuracy `75.0%`
- ARIMAX MAE `37.35`
- ARIMAX R-squared `-0.0935`
- VAR MAE `39.41`

These are no longer aligned with the current pipeline output.

Changes needed:

- rewrite the prose under `4.3.2`
- make the text match the current table exactly
- note that:
  - ARIMAX is best on RMSE and MAE
  - VAR is best on directional accuracy

### 3.2 The sample-size sentence is still wrong

The updated report still says:

- daily sample: `1,300`
- monthly sample: `60 observations`

The current live pipeline output is:

- `daily_master.csv`: `1300` rows
- `monthly_master.csv`: `59` rows

Changes needed:

- replace `60 observations` with `59 observations`

### 3.3 The Granger paragraph is still using older numbers

The report currently states lag-1 Granger values like:

- interest rate `p = 0.7423`
- CPI `p = 0.0639`
- oil `p = 0.6776`
- USD/PKR `p = 0.5866`

The current canonical pipeline outputs are:

- `IR -> flow`: `p = 0.6405`
- `CPI -> flow`: `p = 0.0532`
- `Oil -> flow`: `p = 0.7365`
- `USD/PKR -> flow`: `p = 0.6365`

Changes needed:

- refresh the Granger paragraph under `4.3.1`
- keep the interpretation modest:
  - CPI is borderline at the 10% level
  - no macro variable is strongly significant at 5%

### 3.4 Table 4.2 title still overstates what is actually shown

The report still labels Table 4.2 as:

- `KSE-30 GARCH and EGARCH Results`

But the table content shown in the report still only contains the GARCH row, while EGARCH is discussed only in prose.

Changes needed:

- either add the EGARCH row into the table
- or rename the table to reflect that it currently reports only the GARCH row plus discussion of EGARCH preference

## 4. Chapter 5 rebalancing content is still partly stale

### 4.1 Random Forest AUC is still old

The updated report still uses:

- Random Forest AUC `0.5446`

The current live output in `production_pipeline/output/analysis/results_rebalancing.csv` is:

- RandomForest AUC `0.5982`

Changes needed:

- update the retention/inclusion section in the main body
- update any appendix tables that still use `0.5446`

### 4.2 Main Chapter 5 may still rely on older interpretation text

The top-line interpretation remains broadly fine, but recheck all model-comparison prose against the live table:

- Weight prediction:
  - Naive `0.5595 / 0.2541 / 0.9677`
  - Ridge `0.5590 / 0.2963 / 0.9678`
  - RandomForest `0.6715 / 0.3516 / 0.9535`
- Inclusion prediction:
  - Logistic AUC `0.8214`
  - RandomForest AUC `0.5982`

The current report is still mixing old and new values in different places.

## 5. Appendix A is still stale even though the main chapter table was fixed

This is currently the clearest inconsistency in the report.

### 5.1 Table A.1 still contains the old forecasting values

Appendix A still shows the older set:

- `83.10 / 51.54`
- `58.56 / 37.35`
- `63.10 / 39.41`

while Chapter 4 now shows the current values.

Changes needed:

- update Appendix A Table A.1 to match the live output
- update the explanatory paragraph below it

### 5.2 Appendix A still carries old interpretation text

Appendix A still claims things like:

- ARIMAX improved directional accuracy to `75.0%`
- RMSE improved from `83.10` to `58.56`

Those statements no longer match the live current table.

Changes needed:

- rewrite Appendix A discussion so it matches the latest numbers

### 5.3 Rebalancing appendix values must be rechecked

Appendix A also still uses:

- Random Forest AUC `0.5446`

This is stale versus the current output `0.5982`.

Changes needed:

- refresh all appendix rebalancing tables and commentary from the current CSVs

## 6. TOC and appendix structure are still inconsistent

The update added Appendix C to the body, but the front TOC still lists only:

- Appendix A
- Appendix B
- Appendix D
- Appendix E

Appendix C is still missing from the front TOC.

Changes needed:

- regenerate the TOC
- ensure Appendix C appears in the TOC with the correct page number

## 7. Appendix D and Appendix E still describe the old pipeline lineage

These appendices remain outdated relative to the actual current repo.

They still describe:

- implementation across "three generations of pipeline scripts"
- "Consolidated baseline pipeline"
- "Final KSE-30 narrative pipeline"
- a wrapper-style final run structure

That does not match the current canonical project layout, which is centered on:

- `production_pipeline/`
- `docs/`
- `markdown/`

and the canonical runner:

- `python production_pipeline/run_all.py`

with stage order:

1. `prepare`
2. `pipeline`
3. `eda_raw`
4. `eda_master`
5. `stationarity`
6. `chapter3`
7. `report`
8. `risk_map`

Changes needed:

- rewrite Appendix D and Appendix E around `production_pipeline/`
- remove the old pipeline-lineage narrative as the main production description
- replace it with the actual current project structure and stage order

## 8. Front matter and consistency problems still remain

### 8.1 Advisor title is still inconsistent

Current report still uses:

- title page: `Assistant Professor`
- acknowledgments: `lecturer`

Changes needed:

- choose one correct institutional title and use it consistently

### 8.2 SDG wording is still inconsistent

The Executive Summary still says the project promotes:

- `Decent Work and Economic Growth`
- `Industry, Innovation, and Infrastructure`

But the SDG checklist should be checked against what is actually marked in the final document.

Changes needed:

- make the narrative and the checked SDG items match

### 8.3 NTI/NIT inconsistency remains

The report still uses:

- `NIT`
- `NTI`
- `NTI/NIT`

Changes needed:

- pick one naming convention
- explain it once if necessary
- then stay consistent throughout the report

### 8.4 Abbreviations list still has duplicate or untidy entries

The updated report still shows:

- duplicate `MAE`
- both `NIT` and `NTI`

Changes needed:

- deduplicate abbreviations
- standardize the fund naming entry

## 9. TOC formatting and pagination formatting still need cleanup

The TOC still has formatting problems such as:

- headings running into page numbers
- the new `3.8.1` and `3.8.2` entries being merged awkwardly into surrounding lines

Changes needed:

- regenerate the TOC from Word after final heading cleanup
- verify all heading levels are mapped correctly

## 10. PDF symbol rendering still needs a manual final check

The updated extraction is cleaner than before, but the PDF text still shows replacement characters for:

- some quotes
- some symbols
- some Greek letters
- some superscripts/subscripts

This may partly be text-extraction noise, but the final exported PDF should still be manually checked for:

- equation readability
- `R²`
- Greek letters
- bullets and dashes
- appendix/table labels

## Recommended next editing order

1. Rewrite the Executive Summary in past tense.
2. Fix the Chapter 4 prose under the updated Table 4.1.
3. Refresh the Granger paragraph from current live output.
4. Update Chapter 5 and Appendix A rebalancing values, especially Random Forest AUC.
5. Refresh Appendix A forecasting values so they match Chapter 4.
6. Regenerate the TOC so Appendix C appears and subsection formatting is clean.
7. Rewrite Appendix D and Appendix E around `production_pipeline/`.
8. Standardize advisor title, SDG wording, and NTI/NIT naming.
9. Clean the abbreviations list.
10. Export a final PDF and manually inspect symbol/equation rendering.

## Bottom line

The latest update fixed some visible issues, especially the methodology subsections and the main Chapter 4 forecasting table. But the report is still in a mixed state:

- main table updated, nearby prose stale
- Appendix C added, TOC still missing it
- methodology improved, but numbering and framing still inconsistent
- appendices still describe the old pipeline structure
- Appendix A still carries older metrics

The next pass should focus on consistency, not just adding more material.
