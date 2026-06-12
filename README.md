# FYP Index Funds

This repository is now organized around a single production workspace:

- `production_pipeline/` contains the retained executable code, runtime inputs, orchestration, and generated outputs.
- `docs/` contains formal documents, report assets, references, and copied report workspace material.
- `markdown/` contains narrative notes, QA material, guidance files, and copied markdown references.

## Production Entry Point

Run the end-to-end workflow from:

```bash
python production_pipeline/run_all.py
```

Run a single stage:

```bash
python production_pipeline/run_all.py --stage pipeline
```

Available stages:

- `prepare`
- `pipeline`
- `eda_raw`
- `eda_master`
- `stationarity`
- `chapter3`
- `report`
- `risk_map`

## Production Folder Layout

`production_pipeline/` contains:

- `prepare_kse30_basic.py`
- `pipeline.py`
- `eda_kse30.py`
- `eda_master_processed.py`
- `transform_for_stationarity.py`
- `generate_ch3_eda_graphs.py`
- `generate_report_assets.py`
- `generate_rebalancing_risk_map_with_labels.py`
- `run_all.py`
- `requirements.txt`
- `data/raw/`
- `data/prepared/`
- `output/`

## Runtime Data Kept

Only the input files still used by the retained pipeline were copied into `production_pipeline/data/raw/`:

- `kse30_daily_data.csv`
- `funds_data.xlsx`
- `macro_data.xlsx`
- `cpi.csv`
- `gold.csv`
- `gdp.xls`

## Consolidated Non-Code Content

- `docs/reference/` contains copied proposal, report, and research-paper material from the old document tree.
- `docs/report_workspace/` contains copied report chapter assets and text workspace material.
- `docs/reports/root_exports/` contains the top-level PDF and DOCX exports.
- `markdown/root-notes/` contains the top-level project notes.
- `markdown/qa/` and `markdown/project-mds/` contain copied explanatory material.

## Cleanup Status

Completed:

- removed `8_last_model/`
- removed `9_march15_2026_backtest/`
- created the new production pipeline folder
- copied only the retained runtime inputs into the production pipeline
- consolidated documents into `docs/`
- consolidated markdown into `markdown/`

Pending destructive cleanup:

- the old experiment folders still exist in the repo because a broad deletion pass was blocked for safety before final validation
- once you explicitly approve destructive cleanup, the old folders can be removed in a narrow final pass
