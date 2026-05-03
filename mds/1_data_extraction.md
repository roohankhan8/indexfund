# `1_data_extraction/` — complete file index (ETL scaffolding)

Purpose: Transform PSX spreadsheets into reproducible longitudinal datasets used by **`2_midyear_model/`** onward.

### Files (5 total)

| File | What it does | Models / analytics | Outputs / outcome |
|------|----------------|---------------------|-------------------|
| `legacy-scripts/1-extraction.ipynb` | First-pass scraping / reading of zipped daily PSX excels; validates sheet naming and merges into frames. | None beyond pandas transforms. | Intermediate cleaned tables (inspect notebook outputs). |
| `legacy-scripts/2-delete-sheets.ipynb` | Strips unrelated tabs (macros, blanks) reducing downstream parse errors before batch export. | N/A | Slimmer workbook → faster batch jobs. |
| `legacy-scripts/3-separate-data.py` | Batch reads daily Excel filenames as dates, separates **KSE-100** vs **KSE-30** sheets → writes `kse100_daily_data.csv` & `kse30_daily_data.csv`. ⚠ Uses hard-coded legacy drive paths (`e:\…`) update before reuse. | N/A deterministic ETL only. | Feeds **`0-raw-data/csvs/`** equivalents & later modeling copies. |
| `legacy-scripts/4-clean-company-col.ipynb` | Normalizes COMPANY text (typos/legal suffix inconsistencies) avoiding merge collisions in constituent panels. | N/A | Cleaner symbol↔company map for analytics. |
| `kse30_incremental_update.ipynb` | Applies incremental ingestion when new recompositions arrive (append-only style updates). | N/A — extend dataset while preserving QC checks. | Updated `*_engineered`/CSV exports staged for modeling folders. |

**Results summary:** notebooks themselves only produce intermediary tables; reproducible metrics appear later (`2_midyear_model/` plots, `5_claude_pipeline` CSV summaries). Mention this stage in methodology as **data engineering without statistical inference**.
