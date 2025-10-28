import pandas as pd
import os
from pathlib import Path

# Paths
input_folder = r'e:\Codes\Python\indexfundproject\work\separate_data'
output_folder = r'e:\Codes\Python\indexfundproject\work\separate_data'

# Read the consolidated files
print("Reading KSE 100 data...")
kse100_df = pd.read_csv(os.path.join(input_folder, 'kse100_daily_data.csv'), low_memory=False)

print("Reading KSE 30 data...")
kse30_df = pd.read_csv(os.path.join(input_folder, 'kse30_daily_data.csv'), low_memory=False)

print(f"\nKSE 100 - Total rows: {len(kse100_df)}")
print(f"KSE 30 - Total rows: {len(kse30_df)}")

# Function to create separate sheets for each company
def create_company_sheets(df, output_filename):
    """
    Create an Excel file where each sheet represents a company with its daily data
    """
    print(f"\nProcessing {output_filename}...")
    
    # Get unique companies - try SYMBOL first, fallback to COMPANY if SYMBOL is missing
    if 'SYMBOL' in df.columns:
        # Remove rows where SYMBOL is NaN
        df_clean = df[df['SYMBOL'].notna()].copy()
        unique_companies = df_clean['SYMBOL'].unique()
        group_by_col = 'SYMBOL'
    elif 'COMPANY' in df.columns:
        df_clean = df[df['COMPANY'].notna()].copy()
        unique_companies = df_clean['COMPANY'].unique()
        group_by_col = 'COMPANY'
    else:
        print("Error: Neither SYMBOL nor COMPANY column found!")
        return
    
    print(f"Found {len(unique_companies)} unique companies")
    
    # Create Excel writer
    output_path = os.path.join(output_folder, output_filename)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for idx, company in enumerate(sorted(unique_companies), 1):
            # Get all data for this company
            company_data = df_clean[df_clean[group_by_col] == company].copy()
            
            # Sort by date
            company_data = company_data.sort_values('Date')
            
            # Create a safe sheet name (Excel sheet names have restrictions)
            # Max 31 characters, no special characters like : \ / ? * [ ]
            sheet_name = str(company)[:31]
            sheet_name = sheet_name.replace('/', '-').replace('\\', '-').replace('?', '').replace('*', '').replace('[', '').replace(']', '').replace(':', '-')
            
            # Write to sheet
            company_data.to_excel(writer, sheet_name=sheet_name, index=False)
            
            if idx % 20 == 0:
                print(f"  Processed {idx}/{len(unique_companies)} companies...")
    
    print(f"✓ Successfully created {output_filename}")
    print(f"  Location: {output_path}")
    print(f"  Total sheets: {len(unique_companies)}")

# Process KSE 100
create_company_sheets(kse100_df, 'kse100_daily_data_sep_companies.xlsx')

# Process KSE 30
create_company_sheets(kse30_df, 'kse30_daily_data_sep_companies.xlsx')

print("\n" + "="*80)
print("Company separation completed successfully!")
print("="*80)
