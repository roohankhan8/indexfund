# Guide To Complete The Project

This guide uses [`FYDP.pdf`](./FYDP.pdf) as the formatting and completeness reference, and compares it with your current report draft in [`FYP Report (Analyzing Mutual Funds).pdf`](./FYP%20Report%20%28Analyzing%20Mutual%20Funds%29.pdf).

## Current Status

Your current report already has:

- cover page
- table of contents
- a draft executive summary
- introduction
- a partial literature review
- a partial methodology section
- chapter headings for recommendations and references

Your current report is still missing or incomplete compared with the sample FYDP report:

- `Author's Declaration`
- `Statement of Contribution`
- `Acknowledgement`
- `Similarity Index Report`
- `Abbreviations`
- `List of Figures`
- `List of Tables`
- a proper `Data Collection / Data Processing` chapter
- a detailed `Research Methodology` chapter tied to actual scripts
- a real `Results / Analysis` chapter
- a `Discussion / Interpretation` chapter
- a finished `Conclusion`
- a populated `Recommendations` section
- complete `References`
- `Appendices`

There is also one consistency issue to fix:

- Page 1 says the report is submitted to `Dr Fahim Raees`, but the executive summary says the work is under `Ms. Ubaida Fatima`. Keep one supervisor line everywhere.

## Best Report Structure

Using the sample report as the benchmark, your final report should be organized like this:

1. Cover page
2. Author's Declaration
3. Statement of Contribution
4. United Nations Sustainable Development Goals
5. Executive Summary
6. Acknowledgement
7. Similarity Index Report
8. List of Abbreviations
9. Table of Contents
10. List of Figures
11. List of Tables
12. Chapter 1: Introduction
13. Chapter 2: Literature Review
14. Chapter 3: Data Collection and Processing
15. Chapter 4: Research Methodology
16. Chapter 5: Results and Analysis
17. Chapter 6: Portfolio Tilt / Rebalancing Application
18. Chapter 7: Discussion
19. Chapter 8: Conclusion and Recommendations
20. References
21. Appendices

Your current PDF combines too much under `Methodology`. Split `data`, `methods`, and `results` into separate chapters.

## What To Use From This Repo

Use one primary pipeline for thesis numbers, then use the others for robustness and comparison.

### Primary results source

- `5_claude_pipeline/pipeline.py`
- Best if you want one self-contained end-to-end run.
- Use its `results_*.csv` files as the main source of quantitative tables.

### KSE-30 focused extension

- `6_cursor_model/pipeline.py`
- Best if your final story is specifically about KSE-30 index funds, index efficiency, volatility, and rebalancing.
- Use this for the strongest project-specific discussion chapter.

### Supporting / robustness track

- `3_final_model/scripts/enhanced-v7.py`
- Use this as a comparison track for machine learning forecasting and to discuss small-sample limits.

## Concrete Steps To Finish The Project

### 1. Freeze the final research question

Your report should clearly say that the project does all three of these:

- estimates aggregate monthly fund flows for AKD, NBP, and NTI
- studies KSE-30 market efficiency and volatility
- uses predicted flow direction to tilt portfolio weights or rebalancing decisions

If you do not freeze this scope first, the report will read like separate mini-projects.

### 2. Choose the canonical pipeline

Recommended choice:

- make `5_claude_pipeline/` the base workflow
- use `6_cursor_model/` for KSE-30 focused final analysis
- cite `3_final_model/` as a comparison / robustness branch

This keeps the thesis consistent and avoids conflicting numbers from different folders.

### 3. Re-run the final scripts and keep the outputs

Run:

```powershell
python 5_claude_pipeline/pipeline.py
python 6_cursor_model/pipeline.py
python 3_final_model/scripts/enhanced-v7.py
```

After running, collect:

- `results_fund_flow.csv`
- `results_efficiency.csv`
- `results_garch.csv`
- `results_rebalancing.csv`
- `results_rebalancing_forecast.csv`
- the PNG figures under each folder's `figures/` or `output*/`

These files should drive the report tables and charts.

### 4. Finish Chapter 2 with repo-backed literature

Use:

- `0-docs/psx-research-papers/`
- `0-docs/psx-research-papers/psx-research-papers-md/`
- `mds/flow.md`

Your literature review should cover:

- fund flows and investor behavior
- market efficiency in emerging/frontier markets
- GARCH and volatility modeling
- ARIMA / ARIMAX / VAR style forecasting
- machine learning for PSX or KSE-100 / KSE-30 prediction
- the gap: very limited work on Pakistani index-fund flows tied to portfolio tilting

### 5. Create a real Data Collection and Processing chapter

This chapter should explain:

- data sources: PSX, fund NAV/AUM files, oil, USD/PKR, interest rate, CPI
- time frequency: daily source data, monthly target construction, weekly robustness where relevant
- the flow formula used across the repo:

```text
flow_t = AUM_t - AUM_{t-1} * (NAV_t / NAV_{t-1})
```

- missing-value handling
- forward filling and row filtering
- feature engineering
- lag construction
- train/validation/test or walk-forward setup

Best repo references:

- `5_claude_pipeline/pipeline.py`
- `6_cursor_model/pipeline.py`
- `mds/5_claude_pipeline.md`
- `mds/6_cursor_model.md`

### 6. Expand Methodology into actual model subsections

Your methodology chapter should not stay generic. It should include separate subsections for:

- descriptive analysis and correlation
- fund-flow forecasting
- market efficiency testing
- volatility modeling
- rebalancing / portfolio tilt logic
- model evaluation metrics

Map them to the repo like this:

- Fund flow forecasting: ARIMAX, VAR, naive baseline from `5_claude_pipeline/` and `6_cursor_model/`
- Efficiency: runs test, variance ratio, Ljung-Box style diagnostics, Hurst exponent
- Volatility: GARCH(1,1), EGARCH(1,1)
- ML comparison: Ridge, ElasticNet, Random Forest, Gradient Boosting, XGBoost, LightGBM where used

Also explain the small-sample limitation clearly. That is one of the most important academic caveats in this repo.

### 7. Add a full Results and Analysis chapter

This is the biggest missing piece in your current report.

Minimum subsections:

1. Descriptive statistics and EDA
2. Fund-flow prediction results
3. Market efficiency findings
4. Volatility / GARCH findings
5. Rebalancing or portfolio tilt findings
6. Comparison between econometric and ML approaches

Best evidence files:

- `5_claude_pipeline/results_*.csv`
- `6_cursor_model/results_*.csv`
- `5_claude_pipeline/figures/`
- `6_cursor_model/figures/`
- `3_final_model/output-v7/`

### 8. Add a Discussion chapter

Your discussion should answer:

- Do macro variables help predict flows?
- Is KSE-30 fully efficient, weak-form efficient, or partially inefficient?
- Are volatility effects persistent?
- Are classical econometric models more defensible than deep learning for this dataset?
- Does flow direction help portfolio tilting in a meaningful way?

Use `mds/flow.md` as the high-level narrative source.

### 9. Rewrite Conclusion and Recommendations

Your current PDF only has headings here.

Conclusion should summarize:

- what was built
- what was found
- what was statistically strong
- what remained weak or inconclusive

Recommendations should include:

- extend the sample period
- obtain cleaner and more complete Pakistani mutual fund flow data
- test more macro variables
- evaluate out-of-sample portfolio performance over a longer horizon
- compare monthly vs weekly forecasting setups
- test stronger causal frameworks or regime-switching models

### 10. Finish the front matter and appendices

Add these before final submission:

- signed declaration page
- contribution statement for all group members
- acknowledgement
- Turnitin / similarity index page
- abbreviation table
- list of figures
- list of tables
- appendix with formulas, extra plots, and printed result tables

Useful appendix material from this repo:

- `txts/pipeline.txt` logs
- extra result CSVs
- additional figures not used in the main chapters
- feature lists and model settings

## Recommended Chapter Mapping To Repo

| Report chapter | Main repo sources |
|---|---|
| Introduction | `mds/flow.md`, current PDF intro |
| Literature Review | `0-docs/psx-research-papers/`, `psx-research-papers-md/` |
| Data Collection and Processing | `5_claude_pipeline/pipeline.py`, `6_cursor_model/pipeline.py` |
| Research Methodology | `mds/5_claude_pipeline.md`, `mds/6_cursor_model.md`, `mds/3_final_model.md` |
| Results and Analysis | `results_*.csv`, `figures/`, `output-v7/` |
| Discussion | `mds/flow.md` plus your interpreted result tables |
| Conclusion and Recommendations | your synthesized findings |

## Final Submission Checklist

- One supervisor name used consistently
- Chapter numbering fixed
- `METHADOLOGY` corrected to `METHODOLOGY`
- All tables have numbers and captions
- All figures have numbers and captions
- All result claims match CSV values
- References are in one citation style
- Appendices are cited in the main text
- Final PDF is regenerated after all edits

## Recommended Writing Order

Write in this order to finish faster:

1. Chapter 3: Data Collection and Processing
2. Chapter 4: Research Methodology
3. Chapter 5: Results and Analysis
4. Chapter 7: Discussion
5. Chapter 8: Conclusion and Recommendations
6. Chapter 2: Literature Review polishing
7. Front matter, references, appendices, formatting

This order works because the code and outputs already exist; the missing part is mainly turning them into a clean academic narrative.
