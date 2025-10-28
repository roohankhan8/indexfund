import pandas as pd
import os

# Paths to the new files
kse100_file = r'e:\Codes\Python\indexfundproject\work\separate_data\kse100_daily_data_sep_companies.xlsx'
kse30_file = r'e:\Codes\Python\indexfundproject\work\separate_data\kse30_daily_data_sep_companies.xlsx'

print("=" * 80)
print("KSE 100 - COMPANY SEPARATED FILE")
print("=" * 80)

# Read KSE 100 file
xl_kse100 = pd.ExcelFile(kse100_file)
print(f"Total sheets (companies): {len(xl_kse100.sheet_names)}")
print(f"\nFirst 10 company sheets:")
for i, sheet in enumerate(xl_kse100.sheet_names[:10], 1):
    print(f"  {i}. {sheet}")

# Read one sample sheet
sample_sheet = xl_kse100.sheet_names[0]
df_sample = pd.read_excel(kse100_file, sheet_name=sample_sheet)
print(f"\nSample data from sheet '{sample_sheet}':")
print(f"  Total rows: {len(df_sample)}")
print(f"  Columns: {list(df_sample.columns)}")
print(f"  Date range: {df_sample['Date'].min()} to {df_sample['Date'].max()}")
print(f"\nFirst 5 rows:")
print(df_sample.head())

print("\n" + "=" * 80)
print("KSE 30 - COMPANY SEPARATED FILE")
print("=" * 80)

# Read KSE 30 file
xl_kse30 = pd.ExcelFile(kse30_file)
print(f"Total sheets (companies): {len(xl_kse30.sheet_names)}")
print(f"\nFirst 10 company sheets:")
for i, sheet in enumerate(xl_kse30.sheet_names[:10], 1):
    print(f"  {i}. {sheet}")

# Read one sample sheet
sample_sheet = xl_kse30.sheet_names[0]
df_sample = pd.read_excel(kse30_file, sheet_name=sample_sheet)
print(f"\nSample data from sheet '{sample_sheet}':")
print(f"  Total rows: {len(df_sample)}")
print(f"  Columns: {list(df_sample.columns)}")
print(f"  Date range: {df_sample['Date'].min()} to {df_sample['Date'].max()}")
print(f"\nFirst 5 rows:")
print(df_sample.head())

print("\n" + "=" * 80)
print("File sizes:")
kse100_size = os.path.getsize(kse100_file) / (1024 * 1024)
kse30_size = os.path.getsize(kse30_file) / (1024 * 1024)
print(f"  kse100_daily_data_sep_companies.xlsx: {kse100_size:.2f} MB")
print(f"  kse30_daily_data_sep_companies.xlsx: {kse30_size:.2f} MB")
print("=" * 80)
