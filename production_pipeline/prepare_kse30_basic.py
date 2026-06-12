"""
KSE-30 Basic CSV Formatter
==========================
Reads the incremental kse30_daily_data.csv and creates a filtered version
with only essential columns: Date, SYMBOL, COMPANY, PRICE, IDX WT %, VOLUME

Usage:
    python create_kse30_basic.py

Output:
    kse-30-basic.csv (in same directory)
"""

import os
import sys
from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

BASE = Path(__file__).parent.resolve()
RAW_DIR = BASE / "data" / "raw"
PREPARED_DIR = BASE / "data" / "prepared"
INPUT_CSV = RAW_DIR / "kse30_daily_data.csv"
OUTPUT_CSV = PREPARED_DIR / "kse-30-basic.csv"
OUTPUT_XLSX = PREPARED_DIR / "kse-30-basic.xlsx"

# Required columns in output (in this order)
REQUIRED_COLS = ["Date", "SYMBOL", "COMPANY", "PRICE", "IDX WT %", "VOLUME"]

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Read input CSV and create filtered output."""
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check input file exists
    if not INPUT_CSV.exists():
        print(f"❌ Error: Input file not found: {INPUT_CSV}")
        print(f"   Make sure you've copied kse30_daily_data.csv to this folder.")
        sys.exit(1)
    
    print(f"📖 Reading: {INPUT_CSV}")
    
    # Read CSV
    try:
        df = pd.read_csv(INPUT_CSV)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        sys.exit(1)
    
    print(f"   ✓ Loaded {len(df):,} rows")
    
    # Check all required columns exist
    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        print(f"❌ Error: Missing columns in input file: {missing_cols}")
        print(f"   Available columns: {list(df.columns)}")
        sys.exit(1)
    
    # Select required columns (in specified order)
    df_filtered = df[REQUIRED_COLS].copy()
    
    # Remove rows where SYMBOL is missing or empty
    initial_rows = len(df_filtered)
    df_filtered = df_filtered.dropna(subset=["SYMBOL"])
    df_filtered = df_filtered[df_filtered["SYMBOL"].astype(str).str.strip() != ""]
    removed_rows = initial_rows - len(df_filtered)
    
    if removed_rows > 0:
        print(f"   ⚠ Removed {removed_rows:,} rows with missing/empty SYMBOL")
    
    # Clean and normalize SYMBOL and COMPANY
    df_filtered["SYMBOL"] = df_filtered["SYMBOL"].astype(str).str.strip()
    df_filtered["COMPANY"] = df_filtered["COMPANY"].astype(str).str.strip()
    
    # Ensure Date is in standard format (YYYY-MM-DD)
    try:
        df_filtered["Date"] = pd.to_datetime(df_filtered["Date"]).dt.strftime("%Y-%m-%d")
    except Exception as e:
        print(f"⚠ Warning: Could not standardize Date format: {e}")
    
    # Sort by Date and SYMBOL
    df_filtered = df_filtered.sort_values(["Date", "SYMBOL"]).reset_index(drop=True)
    
    # Write output CSV and XLSX
    try:
        df_filtered.to_csv(OUTPUT_CSV, index=False)
        df_filtered.to_excel(OUTPUT_XLSX, index=False)
        print(f"\n✅ Success! Created: {OUTPUT_CSV}")
        print(f"✅ Success! Created: {OUTPUT_XLSX}")
        print(f"   ✓ {len(df_filtered):,} rows × {len(REQUIRED_COLS)} columns")
        print(f"   ✓ Date range: {df_filtered['Date'].min()} to {df_filtered['Date'].max()}")
        print(f"   ✓ Symbols: {df_filtered['SYMBOL'].nunique()} unique")
    except Exception as e:
        print(f"❌ Error writing output CSV: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
