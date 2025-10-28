import pandas as pd
import os
from openpyxl import load_workbook

# Paths
input_folder = r'e:\Codes\Python\indexfundproject\work\separate_data'
output_folder = r'e:\Codes\Python\indexfundproject\work\separate_data\cleaned'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

def merge_volume_columns(input_file, output_file):
    """
    Merge Volume and VOLUME columns, keeping data in VOLUME and removing Volume column
    """
    filename = os.path.basename(input_file)
    print(f"\nProcessing {filename}...")
    
    # Read all sheets
    xl = pd.ExcelFile(input_file)
    sheet_names = xl.sheet_names
    print(f"  Total sheets: {len(sheet_names)}")
    
    # Create Excel writer
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for idx, sheet_name in enumerate(sheet_names, 1):
            # Read the sheet
            df = pd.read_excel(input_file, sheet_name=sheet_name)
            
            # Check if both Volume and VOLUME columns exist
            has_volume = 'Volume' in df.columns
            has_VOLUME = 'VOLUME' in df.columns
            
            if has_volume and has_VOLUME:
                # Merge: prioritize non-NaN values
                # If VOLUME is NaN, use Volume; otherwise keep VOLUME
                df['VOLUME'] = df['VOLUME'].fillna(df['Volume'])
                
                # Drop the Volume column
                df = df.drop(columns=['Volume'])
                
            elif has_volume and not has_VOLUME:
                # Rename Volume to VOLUME
                df = df.rename(columns={'Volume': 'VOLUME'})
            
            # Write to new file
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            if idx % 20 == 0:
                print(f"    Processed {idx}/{len(sheet_names)} sheets...")
    
    print(f"  ✓ Saved to: {output_file}")

# Process KSE 100 file
kse100_input = os.path.join(input_folder, 'kse100_daily_data_sep_companies.xlsx')
kse100_output = os.path.join(output_folder, 'kse100_daily_data_sep_companies.xlsx')
merge_volume_columns(kse100_input, kse100_output)

# Process KSE 30 file
kse30_input = os.path.join(input_folder, 'kse30_daily_data_sep_companies.xlsx')
kse30_output = os.path.join(output_folder, 'kse30_daily_data_sep_companies.xlsx')
merge_volume_columns(kse30_input, kse30_output)

print("\n" + "="*80)
print("Volume columns merged successfully!")
print(f"Output folder: {output_folder}")
print("="*80)
