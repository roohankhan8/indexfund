import pandas as pd

# Read the created files
kse100_df = pd.read_csv(r'e:\Codes\Python\indexfundproject\work\separate_data\kse100_daily_data.csv')
kse30_df = pd.read_csv(r'e:\Codes\Python\indexfundproject\work\separate_data\kse30_daily_data.csv')

print("=" * 80)
print("KSE 100 DATA")
print("=" * 80)
print(f"Total rows: {len(kse100_df)}")
print(f"Columns: {list(kse100_df.columns)}")
print(f"\nFirst few rows:")
print(kse100_df.head(10))
print(f"\nUnique dates: {kse100_df['Date'].nunique()}")
print(f"Date range: {kse100_df['Date'].min()} to {kse100_df['Date'].max()}")

print("\n" + "=" * 80)
print("KSE 30 DATA")
print("=" * 80)
print(f"Total rows: {len(kse30_df)}")
print(f"Columns: {list(kse30_df.columns)}")
print(f"\nFirst few rows:")
print(kse30_df.head(10))
print(f"\nUnique dates: {kse30_df['Date'].nunique()}")
print(f"Date range: {kse30_df['Date'].min()} to {kse30_df['Date'].max()}")
