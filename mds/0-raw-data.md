# `0-raw-data/` — complete file index (immutable inputs)

**Models:** none (raw artefacts only).

**Results:** downstream CSV/XLSX in later stages derive from these files once cleaned — no metrics here.

| Path | Detailed role |
|------|----------------|
| `csvs/kse30_daily_data.csv` | Canonical long-format CSV for daily KSE-30 constituents — primary bridge into scripting pipelines (`5_claude_pipeline` reads this style of file). Columns typically include dates, ticker, weight, FF fields, volumes, etc. |
| `csvs/kse100_daily_data.csv` | Same pattern for broader KSE-100 constituents — useful contextual market variable or benchmarking. |
| `xlsx/kse-30-OHLCV.xlsx` | Excel snapshot of OHLCV-style constituent data gathered before consolidation to CSV pipelines. |
| `extras/PSX_Recomposition_Dataset_2020_2025.xlsx` | Master “recomposition” workbook spanning 2020–2025 edits (weight changes across review dates). |
| `extras/kse30_daily_data_engineered.xlsx` | Post-engineering enrichment / derived columns workbook from early processing (feature sketching stage). |
| `extras/kse30_daily_data_recomposition.xlsx` | Constituents aligned to rebalancing dates / weight schedules. |
| `extras/kse30_daily_data-changed.xlsx` | Iteration variant after manual corrections (“changed”) — keep for reproducibility lineage. |
| `extras/kse100_daily_data_pivottable.xlsx` | Pivot-heavy Excel view summarizing liquidity and sector buckets for exploratory reporting. |
| `extras/recompostition.xlsx` | Alternate spelling archival copy of recompositions (typo preserved in filename). |
| `zips/data.zip` | Portable archive bundling miscellaneous raw spreadsheets (backup). |
| `zips/csvs.zip` | Portable archive wrapping `csvs/` exports prior to ingestion by notebooks/scripts. |

**Use in thesis:** cite this folder under *Data Acquisition* — anything analytical should reference the **derived** copies under `*_model/data/` or `5_claude_pipeline/` after cleaning, not ad-hoc mutation of originals here.
