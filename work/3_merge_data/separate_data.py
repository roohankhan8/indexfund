import pandas as pd
import os
from pathlib import Path
import glob

# Paths
input_folder = r'e:\Codes\Python\indexfundproject\work\kse-100-30-data'
output_folder = r'e:\Codes\Python\indexfundproject\work\separate_data'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get all Excel files
excel_files = sorted(glob.glob(os.path.join(input_folder, '*.xlsx')) + 
                     glob.glob(os.path.join(input_folder, '*.xls')))

print(f"Found {len(excel_files)} Excel files to process")

# Initialize dictionaries to store data for each index
kse100_data = {}
kse30_data = {}

# Process each file
for file_path in excel_files:
    # Extract date from filename (e.g., '2020-01-01.xlsx' -> '2020-01-01')
    filename = os.path.basename(file_path)
    date = os.path.splitext(filename)[0]
    
    try:
        # Read both sheets
        xl = pd.ExcelFile(file_path)
        
        # Find sheet names (they might vary slightly)
        kse100_sheet = None
        kse30_sheet = None
        
        for sheet in xl.sheet_names:
            if 'KSE 100' in sheet or 'KSE-100' in sheet or 'kse 100' in sheet.lower():
                kse100_sheet = sheet
            elif 'KSE 30' in sheet or 'KSE-30' in sheet or 'kse 30' in sheet.lower():
                kse30_sheet = sheet
        
        # Read KSE 100 data
        if kse100_sheet:
            df_kse100 = pd.read_excel(file_path, sheet_name=kse100_sheet)
            kse100_data[date] = df_kse100
        
        # Read KSE 30 data
        if kse30_sheet:
            df_kse30 = pd.read_excel(file_path, sheet_name=kse30_sheet)
            kse30_data[date] = df_kse30
        
        if len(kse100_data) % 100 == 0:
            print(f"Processed {len(kse100_data)} files...")
    
    except Exception as e:
        print(f"Error processing {filename}: {e}")

print(f"\nTotal files processed successfully:")
print(f"KSE 100: {len(kse100_data)} dates")
print(f"KSE 30: {len(kse30_data)} dates")

# Create combined DataFrames with dates as first column
# For KSE 100
if kse100_data:
    # Get all unique columns from all dates
    all_columns = set()
    for df in kse100_data.values():
        all_columns.update(df.columns)
    
    # Create a list to store all rows
    kse100_rows = []
    
    for date, df in sorted(kse100_data.items()):
        # Add date column
        df_copy = df.copy()
        df_copy.insert(0, 'Date', date)
        kse100_rows.append(df_copy)
    
    # Concatenate all data
    kse100_combined = pd.concat(kse100_rows, ignore_index=True)
    
    # Save to CSV
    output_file_kse100 = os.path.join(output_folder, 'kse100_daily_data.csv')
    kse100_combined.to_csv(output_file_kse100, index=False)
    print(f"\nKSE 100 data saved to: {output_file_kse100}")
    print(f"Shape: {kse100_combined.shape}")
    print(f"Columns: {list(kse100_combined.columns)}")

# For KSE 30
if kse30_data:
    # Create a list to store all rows
    kse30_rows = []
    
    for date, df in sorted(kse30_data.items()):
        # Add date column
        df_copy = df.copy()
        df_copy.insert(0, 'Date', date)
        kse30_rows.append(df_copy)
    
    # Concatenate all data
    kse30_combined = pd.concat(kse30_rows, ignore_index=True)
    
    # Save to CSV
    output_file_kse30 = os.path.join(output_folder, 'kse30_daily_data.csv')
    kse30_combined.to_csv(output_file_kse30, index=False)
    print(f"\nKSE 30 data saved to: {output_file_kse30}")
    print(f"Shape: {kse30_combined.shape}")
    print(f"Columns: {list(kse30_combined.columns)}")

print("\nData separation completed successfully!")
