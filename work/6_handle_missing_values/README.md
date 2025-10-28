# Handle Missing Values - Documentation

## Overview
This folder contains tools for identifying and handling missing values in company stock data files.

**Important Note**: The data files use **zeros (0)** to represent missing values instead of NULL/NaN values.

## Files
- `handle_missing_values.ipynb` - Main notebook for processing zero values (missing data)
- `cleaned_data/` - Output folder containing cleaned Excel files
- `processing_log.csv` - Detailed log of all changes made

## Strategy

### Zero Value Handling Rules:

1. **Price Columns (OPEN, HIGH, LOW, CLOSE)**
   - Replace zeros with NaN
   - Fill using forward fill method (use previous day's price)
   - Then apply backward fill for any remaining gaps
   - Rationale: Stock prices are continuous and previous price is best estimate; zeros are not valid prices

2. **Volume Column**
   - Replace zeros with median value (calculated from non-zero values)
   - Rationale: Median is more robust to outliers; some days may legitimately have zero volume

3. **Rows with ALL Price Data as Zeros**
   - Remove these rows entirely
   - Rationale: Cannot reconstruct meaningful price data without any reference

4. **Other Numeric Columns**
   - Replace zeros with median value (calculated from non-zero values)
   - Rationale: Safe default approach

## How to Use

1. Open `handle_missing_values.ipynb`
2. Run all cells sequentially
3. The notebook will:
   - Analyze zero values (missing data) in all sheets
   - Show detailed statistics
   - Apply the cleaning strategy
   - Save cleaned files to `cleaned_data/` folder
   - Generate a processing log

## Output

### Cleaned Data Files
- Located in: `cleaned_data/`
- Same structure as input files
- Missing values handled according to strategy

### Processing Log
- File: `processing_log.csv`
- Contains details for each sheet processed:
  - Original row count
  - Final row count
  - Rows removed
  - Zero values replaced
  - Actions taken

## Verification

The notebook includes verification steps to ensure:
- All zero values (representing missing data) are properly handled
- No unexpected data loss
- Quality of imputed values

## Notes

- Original files in `4_separate_companies/merge_columns/` are NOT modified
- All changes are saved to new files in `cleaned_data/` folder
- Processing log allows audit trail of all changes
- **Zero values are treated as missing data** and replaced according to the strategy
- Some legitimate zeros may remain (e.g., days with no trading activity)
