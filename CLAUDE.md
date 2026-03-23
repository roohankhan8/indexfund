# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Final Year Project: Predicts monthly mutual fund flows (net investor inflows/outflows) into KSE-30 index funds on the Pakistan Stock Exchange (PSX) using lagged market signals and macroeconomic variables. The predicted flow direction is used to tilt portfolio weights — overweight high-volume stocks on inflows, equal-weight on outflows.

Fund flow formula: `flow_t = AUM_t - AUM_{t-1} × (NAV_t / NAV_{t-1})`

Funds tracked: AKD, NBP, NTI (3 KSE-30 index mutual funds)

## Environment Setup

```bash
# Activate virtual environment
.venv\Scripts\activate    # Windows
source .venv/bin/activate # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

Python version: 3.12 (`.venv` is pre-configured).

## Running the Pipeline

The canonical entry point is the v5 script:

```bash
python new-pipeline/v5/enhanced-v5.py
```

Outputs 9 PNG charts to `new-pipeline/v5/output/`.

To regenerate the notebook from the `.py` source:

```bash
python new-pipeline/v5/_gen_notebook.py
```

Then open `new-pipeline/v5/enhanced-v5.ipynb` in Jupyter.

## Architecture

### Data Flow

```
data/                         # Raw PSX recomposition data
new-pipeline/data/            # Processed input data for the active pipeline
  ├── kse-30-basic.xlsx       # Daily: date, symbol, price, index weight %, volume
  ├── funds_data.xlsx         # Daily: NAV + AUM for AKD, NBP, NTI
  ├── macro_data.xlsx         # Daily: OIL (Brent), USD/PKR, IR (KIBOR)
  ├── cpi.csv                 # Monthly CPI
  └── kse30_index_level.csv   # Pre-computed daily KSE-30 index returns (primary return source)
```

### Pipeline Sections (enhanced-v5.py)

| Section | What it does |
|---------|-------------|
| 1 | Load `kse30_index_level.csv` → monthly index returns |
| 2 | Load `kse-30-basic.xlsx` → company-level price/volume (not used for returns — use index level instead) |
| 3 | Compute fund flows from `funds_data.xlsx` using the flow formula above |
| 4 | Load + forward-fill macro data (OIL daily→monthly, IR daily ffill→monthly, USD daily→monthly) |
| 5 | Load CPI, compute YoY inflation |
| 6+ | Merge all → winsorize outliers (1–99%) → engineer 12 features → train 4 models → evaluate → 6-month forecast → select top-10 companies |

### Feature Engineering (12 features)

Multi-lag (1+3 month) lags on: index return, macro variables, flow itself. Plus return volatility (rolling 3-month std) and abnormal volume (vs 3-month average).

### Models

- ElasticNet (primary linear)
- Ridge (baseline linear)
- LightGBM (falls back to XGBoostRegressor if lightgbm unavailable)
- GradientBoostingRegressor

Evaluation: `TimeSeriesSplit(5)` cross-validation. Metrics: RMSE, MAE, R², directional accuracy. Best model selected for 6-month iterative forward forecast.

### Dataset Size

~47–57 monthly rows after cleaning. This constrains model choice — LSTM was dropped (too few rows for sequence modeling), deep learning is not viable here.

## Key Design Decisions

- **2020 data excluded** from macro features — no reliable macro data; `bfill` from 2021 fabricates past values.
- **Use `kse30_index_level.csv`** as the index return source, not weighted price calculation from `kse-30-basic.xlsx`.
- **No LSTM** — confirmed non-viable at this dataset size.
- **Winsorize at 1–99%** before feature engineering to handle outlier fund flow months.
- **Forward-fill IR (KIBOR)** — policy rate changes infrequently; daily ffill then monthly aggregation is correct.

## Directory Guide

| Path | Purpose |
|------|---------|
| `new-pipeline/v5/` | **Active development** — final pipeline version |
| `new-pipeline/v1–v4/` | Historical iteration notebooks (for reference) |
| `new-pipeline/*.md` | `context.md` (overview), `plan.md` (v5 architecture), `bugs.md` (v4 bug list), `extra.md` (version comparison table) |
| `pipeline/` | One-time ETL scripts that produced the raw CSVs/Excel files |
| `model/` | Early exploratory notebooks (pre-pipeline) |
| `reports/` | Power BI `.pbix` dashboards |
| `graphs/`, `new-pipeline/new-pipeline-output/`, `new-pipeline/v5/output/` | Generated chart outputs |
