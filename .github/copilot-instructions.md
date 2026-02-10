# AI Coding Agent Instructions - KSE Index Fund Project

## Project Overview
This is a **Pakistan Stock Exchange (PSX) index fund analysis project** that processes, cleans, and analyzes historical stock data for KSE-100 and KSE-30 indices. The project transforms raw PSX data into engineered features for machine learning models predicting fund flows.

**Key Data Flow**: Raw Excel files → Extracted daily data → Cleaned sheets → Merged by index → Separated by company → Missing value handling → Engineered features → ML model

## Architecture & Data Pipeline

### Sequential Processing Stages (Work Folder)
1. **`1_data_extraction/`** - Extract daily OHLCV data from raw PSX Excel files
   - Handles inconsistent sheet naming (KSE 100/KSE 100/kse 100)
   - Outputs consolidated CSVs: `kse100_daily_data.csv`, `kse30_daily_data.csv`

2. **`2_data_sheets_cleaning/`** - Remove unnecessary sheets, standardize formats

3. **`3_merge_data/`** - Consolidate daily data from multiple files into index-level datasets
   - See [separate_data.py](work/3_merge_data/separate_data.py): Uses filename dates as index dates

4. **`4_separate_companies/`** - Create per-company Excel sheets from consolidated data
   - Groups by `SYMBOL` or `COMPANY` column (fallback logic in [separate_companies.py](work/4_separate_companies/separate_companies.py))
   - `merge_volume_columns/` - Handles duplicate volume columns from misaligned source data

5. **`5_data_visualization/`** - Exploratory analysis notebooks

6. **`6_handle_missing_values/`** - **Critical stage with domain-specific rules**
   - **Zeros represent missing data** (not NULL), not actual zero values
   - Strategy per column type:
     - **Price columns (OPEN, HIGH, LOW, CLOSE)**: Replace zeros → forward fill → backward fill
     - **Volume**: Replace zeros with median (from non-zero values)
     - **All-zero rows**: Drop entirely
     - **Other numeric**: Replace zeros with median

### ML Model ([model/index.ipynb](model/index.ipynb))
- Uses `kse30_daily_data_engineered.xlsx` with engineered features
- Random Forest model predicting `FLOW_PROXY` (fund flow indicator)
- Features: `DELTA_WT`, `DELTA_PRICE`, `DELTA_VOLUME`, `PREV_IDX_WT`, `PREV_FF_BASED_MCAP`, `RETURN`
- Test set uses `shuffle=False` (time-series data must maintain temporal order)

## Project Conventions & Patterns

### Column Naming
- **Standardized format**: UPPERCASE with underscores (e.g., `DELTA_WT`, `FLOW_PROXY`)
- **Column cleaning**: Always strip spaces and uppercase: `df.columns = df.columns.str.replace(" ", "_").str.upper()`
- Dates are in `DATE` column, convert to datetime: `pd.to_datetime(df["DATE"])`

### File Formats & Paths
- **Input**: Raw Excel files (`.xlsx`, rarely `.xls`) from PSX with multiple sheets per date
- **Intermediate**: CSV for consolidated data (`*_daily_data.csv`)
- **Output**: Cleaned Excel files with per-company sheets
- **Hard-coded paths** in verification/processing scripts (e.g., `r'e:\Codes\Python\indexfundproject\...'`)

### Data Quality Concerns
- **Lookup pattern**: Check for `SYMBOL` vs `COMPANY` columns; use `.notna()` filtering
- **Validation**: Use verification scripts (`verify_*.py`) to audit data before proceeding to next stage
- Processing logs saved to `processing_log.csv` with row counts and operations applied

### Time Series Handling
- **No shuffling in train/test splits** for time-series models (see model code line 52-54)
- Data is indexed by date; maintain temporal integrity

## Common Developer Tasks

### Inspecting Raw Data
```python
import pandas as pd
xl = pd.ExcelFile('file.xlsx')
print(xl.sheet_names)  # Check actual sheet names (vary by date)
df = pd.read_excel('file.xlsx', sheet_name=sheet_name)
```
See [examine_file.py](work/examine_file.py) for reference.

### Running Pipeline Steps
Each stage is a standalone Python script or notebook in numbered folders. Run sequentially:
1. Extract → 2. Clean → 3. Merge → 4. Separate Companies → 5. Visualize → 6. Handle Missing Values

### Debugging Sheet/Column Issues
- Verify with [verify_company_sheets.py](work/verify_company_sheets.py) - checks per-company data integrity
- Check column consistency with [verify_cleaned_files.py](work/verify_cleaned_files.py)

## External Dependencies
- **pandas** ≥1.5 - Core data manipulation
- **openpyxl** ≥3.0 - Excel I/O with sheet control
- **scikit-learn** - Random Forest model
- **matplotlib/plotly** - Visualization (in analysis notebooks)

## Key Files & Patterns to Reference
- [work/6_handle_missing_values/README.md](work/6_handle_missing_values/README.md) - Complete missing value strategy
- [work/3_merge_data/separate_data.py](work/3_merge_data/separate_data.py) - Multi-file consolidation pattern
- [work/4_separate_companies/separate_companies.py](work/4_separate_companies/separate_companies.py) - Multi-sheet Excel creation pattern
- [model/index.ipynb](model/index.ipynb) - ML feature engineering and time-series model validation

## Critical Do's & Don'ts
- ✅ Always treat **zeros as missing data** in stock price/volume columns after step 3
- ✅ Preserve **temporal order** when splitting train/test data
- ✅ **Validate sheet names** before reading (KSE naming varies)
- ❌ Don't shuffle time-series data in train/test split
- ❌ Don't assume column names are consistent across raw files
- ❌ Don't import `.xls` without checking dependencies (xlrd may be needed)
