## Purpose

This repository is a Final Year Project that studies **KSE-30 index funds** on PSX and predicts **monthly fund flows** (inflow/outflow) using lagged market and macro signals, then uses the predicted direction to tilt portfolio weights.

This file tells coding agents how to work safely and effectively in this repo.

## How to run (Windows / PowerShell)

- **Python**: 3.12 (repo uses a local `.venv`)

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

### Main runnable pipelines (pick one)

- **Single-file canonical pipeline** (AKD/NBP/NTI splits + full workflow; masters + figures next to inputs):

```bash
python 5_claude_pipeline/pipeline.py
```

- **KSE-30-focused variant** (combined index-fund flows + index-centric GARCH/efficiency; reads CSV/Excel from `5_claude_pipeline/`, writes outputs under `6_cursor_model/`):

```bash
python 6_cursor_model/pipeline.py
```

- **Final-model “enhanced” script** (generates versioned plots under `3_final_model/`):

```bash
python 3_final_model/scripts/enhanced-v7.py
```

## Data conventions

- **Fund flow definition** (used across pipelines):
  - \(flow_t = AUM_t - AUM_{t-1} \times (NAV_t / NAV_{t-1})\)
- **Core inputs** usually include:
  - KSE-30 constituents panel (daily price/weight/volume),
  - fund NAV + AUM for AKD/NBP/NTI,
  - macro indicators (Oil, USD/PKR, interest rate),
  - CPI (monthly).
- Many folders keep **local copies** of these inputs (e.g. `3_final_model/data/`, `4_claude_model/data/`, `5_claude_pipeline/`).

## Repo structure (high-signal)

- **`0-docs/`**: papers, proposal docs (reference only).
- **`0-raw-data/`**: raw Excel/CSV/zips (treat as source artifacts).
- **`1_data_extraction/`**: extraction + cleaning notebooks/scripts that produced early clean daily tables.
- **`2_midyear_model/`**: mid-year experiments (notebook-driven; plots under `graphs/` and `output/`).
- **`3_final_model/`**: final enhanced iterations (multiple `enhanced-v*.ipynb`, key scripts under `scripts/`, outputs under `output*`).
- **`4_claude_model/`**: modular “nb*.py” research pipeline (preprocessing/EDA/GARCH/efficiency/portfolio/summary) with `figures/` and `processed_data/`.
- **`5_claude_pipeline/`**: single self-contained `pipeline.py` plus local copies of key inputs (`kse30_daily_data.csv`, Excel, etc.).
- **`6_cursor_model/`**: separate `pipeline.py` — same flow family trimmed to aggregated KSE-30 flows and index-centric stats; consumes inputs from `5_claude_pipeline/` and writes figures/CSVs locally.

## Agent rules (do/don’t)

- **Do** prefer editing runnable `.py` scripts over `.ipynb` unless the user explicitly requests notebook edits.
- **Do** keep outputs deterministic and saved to existing output folders (`figures/`, `output/`, etc.).
- **Do** avoid huge diffs in binary files (`.xlsx`, `.png`, `.pdf`, `.zip`). Treat them as inputs/outputs, not source code.
- **Do** keep paths portable:
  - Avoid hard-coded drive letters like `E:\...` in new code.
  - Use relative paths anchored to the script location (`__file__`) when possible.
- **Don’t** rename/remove historical folders (they are part of the research timeline).
- **Don’t** commit changes unless the user explicitly asks.

## Reporting help

- Folder-level writeups live in **`mds/`** (root). Update those docs when you add/change a pipeline stage so the FYP report stays in sync with the code.

