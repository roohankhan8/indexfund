import pandas as pd
import os

# Paths
cleaned_folder = r'e:\Codes\Python\indexfundproject\work\separate_data\cleaned'
kse100_file = os.path.join(cleaned_folder, 'kse100_daily_data_sep_companies.xlsx')
kse30_file = os.path.join(cleaned_folder, 'kse30_daily_data_sep_companies.xlsx')

print("=" * 80)
print("CLEANED FILES VERIFICATION")
print("=" * 80)

# Verify KSE 100
print("\nKSE 100 - Cleaned File")
print("-" * 80)
xl_kse100 = pd.ExcelFile(kse100_file)
print(f"Total sheets: {len(xl_kse100.sheet_names)}")

# Check first sheet
sample_sheet = xl_kse100.sheet_names[0]
df_sample = pd.read_excel(kse100_file, sheet_name=sample_sheet)
print(f"\nSample sheet: '{sample_sheet}'")
print(f"Columns: {list(df_sample.columns)}")
print(f"Total columns: {len(df_sample.columns)}")

# Check if Volume column exists
if 'Volume' in df_sample.columns:
    print("⚠️  WARNING: 'Volume' column still exists!")
else:
    print("✓ 'Volume' column successfully removed")

if 'VOLUME' in df_sample.columns:
    print("✓ 'VOLUME' column exists")
    print(f"  Non-null VOLUME values: {df_sample['VOLUME'].notna().sum()}/{len(df_sample)}")
else:
    print("⚠️  WARNING: 'VOLUME' column not found!")

print(f"\nFirst 5 rows:")
print(df_sample.head())

# Verify KSE 30
print("\n" + "=" * 80)
print("KSE 30 - Cleaned File")
print("-" * 80)
xl_kse30 = pd.ExcelFile(kse30_file)
print(f"Total sheets: {len(xl_kse30.sheet_names)}")

# Check first sheet
sample_sheet = xl_kse30.sheet_names[0]
df_sample = pd.read_excel(kse30_file, sheet_name=sample_sheet)
print(f"\nSample sheet: '{sample_sheet}'")
print(f"Columns: {list(df_sample.columns)}")
print(f"Total columns: {len(df_sample.columns)}")

# Check if Volume column exists
if 'Volume' in df_sample.columns:
    print("⚠️  WARNING: 'Volume' column still exists!")
else:
    print("✓ 'Volume' column successfully removed")

if 'VOLUME' in df_sample.columns:
    print("✓ 'VOLUME' column exists")
    print(f"  Non-null VOLUME values: {df_sample['VOLUME'].notna().sum()}/{len(df_sample)}")
else:
    print("⚠️  WARNING: 'VOLUME' column not found!")

print(f"\nFirst 5 rows:")
print(df_sample.head())

# File sizes
print("\n" + "=" * 80)
print("File Sizes:")
kse100_size = os.path.getsize(kse100_file) / (1024 * 1024)
kse30_size = os.path.getsize(kse30_file) / (1024 * 1024)
print(f"  kse100_daily_data_sep_companies.xlsx: {kse100_size:.2f} MB")
print(f"  kse30_daily_data_sep_companies.xlsx: {kse30_size:.2f} MB")
print("=" * 80)
