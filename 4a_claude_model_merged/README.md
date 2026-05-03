# 4a_claude_model_merged — Unified Analysis Pipeline

## Overview

This folder contains a **single, self-contained Python script** (`merged_pipeline.py`) that replicates the complete analytical workflow from `4_claude_model/` (all 8 notebooks combined into one orchestrated pipeline).

**What it does:**
- Preprocessing & cleaning of raw financial data
- Exploratory data analysis (EDA) with 4+ visualizations
- Fund flow prediction (ARIMAX + VAR models)
- GARCH volatility modeling
- Portfolio optimization (Markowitz efficient frontier)
- Rebalancing prediction (classification & regression)
- Market efficiency tests (Runs test, ACF, variance ratio)
- Results summary with dashboard figures

## Folder Structure

```
4a_claude_model_merged/
├── merged_pipeline.py          # Single orchestrator script
├── input_data/                 # Place input Excel/CSV files here
│   ├── kse-30-basic.xlsx
│   ├── funds_data.xlsx
│   ├── macro_data.xlsx
│   ├── cpi.csv
│   └── kse30_index_level.csv   (optional)
├── output_data/                # All outputs written here
│   ├── daily_master.csv
│   ├── monthly_master.csv
│   ├── kse30_stocks_daily.csv
│   ├── portfolio_weights.csv
│   ├── results_fund_flow.csv
│   ├── results_garch.csv
│   ├── results_efficiency.csv
│   ├── results_rebalancing.csv
│   ├── pipeline_report.txt
│   └── figures/
│       ├── eda/
│       ├── garch/
│       ├── fund_flow/
│       ├── portfolio/
│       ├── efficiency/
│       ├── rebalancing/
│       └── summary/
└── README.md                   # This file
```

## How to Run

### 1. Prepare Input Data

Copy the required data files to `input_data/`:
- `kse-30-basic.xlsx` (KSE-30 stock daily data)
- `funds_data.xlsx` (Fund NAV/AUM for AKD, NBP, NIT)
- `macro_data.xlsx` (Oil, USD/PKR, Interest Rate)
- `cpi.csv` (Monthly CPI)
- `kse30_index_level.csv` (optional; index-level returns)

**Fallback:** If files are not found in `input_data/`, the script will look in `../4_claude_model/data/` (original location).

### 2. Activate Virtual Environment

```powershell
# Windows
.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate
```

### 3. Run the Pipeline

```powershell
python 4a_claude_model_merged/merged_pipeline.py
```

The script will:
- Load and preprocess all data
- Generate 8+ analysis sections
- Save all outputs to `output_data/`
- Print progress to console and save to `pipeline_report.txt`

### 4. Expected Runtime

~2–5 minutes (depending on machine and data size).

## Output Sections

| Section | Output Files | What it does |
|---------|--------------|-------------|
| **1. Preprocessing** | `daily_master.csv`, `monthly_master.csv` | Data cleaning, alignment, feature engineering |
| **2. EDA** | `figures/eda/*.png` (4 figures) | Macro trends, AUM, flows, correlations |
| **3. Fund Flow** | `results_fund_flow.csv`, `figures/fund_flow/*.png` | ARIMAX + VAR forecasting |
| **4. GARCH** | `results_garch.csv`, `figures/garch/*.png` | Volatility modeling |
| **5. Portfolio** | `portfolio_weights.csv`, `figures/portfolio/*.png` | Markowitz efficient frontier |
| **6. Efficiency** | `results_efficiency.csv`, `figures/efficiency/*.png` | Runs test, ACF, variance ratio |
| **7. Rebalancing** | `results_rebalancing.csv`, `figures/rebalancing/*.png` | Index composition prediction |
| **8. Summary** | `figures/summary/*.png` | Dashboard collage |

## Configuration

Edit the top of `merged_pipeline.py` (Section 0) to customize:

```python
WINDOW_START = pd.Timestamp("2021-01-04")       # Analysis start date
WINDOW_END = pd.Timestamp("2025-10-01")         # Analysis end date
TRAIN_END = "2023-12-31"                        # Train/test split
RISK_FREE = 0.105 / 252                         # Risk-free rate
```

## Key Differences from Original Notebooks

| Aspect | Original (`4_claude_model/`) | Merged (`4a_claude_model_merged/`) |
|--------|-------|----------|
| **Execution** | 8 separate notebooks | 1 unified script |
| **Data loading** | Manual path adjustment per notebook | Automatic fallback logic |
| **Output location** | Mixed: `processed_data/`, `figures/`, various | Centralized: `output_data/` |
| **Configuration** | Scattered across notebooks | Unified Section 0 |
| **Reproducibility** | Requires running notebooks in order | Single command, deterministic |

## Dependencies

```
pandas >= 1.3
numpy >= 1.20
openpyxl >= 3.0        (for Excel I/O)
scipy >= 1.7
scikit-learn >= 0.24
matplotlib >= 3.4
seaborn >= 0.11
```

Install via:
```powershell
pip install -r ../requirements.txt
```

## Troubleshooting

### "File not found" error
- Check that `input_data/` folder contains all required Excel/CSV files
- Verify file names match exactly (case-sensitive on Linux/macOS)
- Script will try fallback path: `../4_claude_model/data/`

### Memory error on large datasets
- Reduce analysis window: edit `WINDOW_START`, `WINDOW_END`
- Reduce portfolio `n_portfolios` in Section 5 (currently 5,000)

### Missing optional file warning
- `kse30_index_level.csv` is optional; script works without it (index columns skipped)
- Other files are required

## Report & Logging

All console output is captured in:
```
output_data/pipeline_report.txt
```

Copy this file into your thesis appendix as a trace of analysis steps.

## Integration with Thesis

**Citation template:**
> "Analysis pipeline executed via merged_pipeline.py (4a_claude_model_merged/).
> Outputs: results_*.csv and figures/ as referenced in Methods section."

**Results tables:**
- `results_fund_flow.csv` → Methods/Results (forecasting metrics)
- `results_garch.csv` → Volatility section (GARCH parameters)
- `results_efficiency.csv` → Efficiency section (test statistics)
- `portfolio_weights.csv` → Portfolio optimization section

**Figures:**
- `figures/eda/` → Introduction / Data section
- `figures/fund_flow/` → Flow prediction section
- `figures/garch/` → Volatility section
- `figures/portfolio/` → Portfolio section
- `figures/efficiency/` → Efficiency section
- `figures/summary/` → Summary / conclusions

---

For questions, refer to `../AGENTS.md` or `../mds/4_claude_model.md` for original modular pipeline documentation.
