import pandas as pd

file_path = r'e:\Codes\Python\indexfundproject\work\kse-100-30-data\2020-01-01.xlsx'
xl = pd.ExcelFile(file_path)
print('Sheet names:', xl.sheet_names)

# Read first sheet
df1 = pd.read_excel(file_path, sheet_name=xl.sheet_names[0])
print('\nFirst sheet preview:')
print(df1.head())
print('\nFirst sheet shape:', df1.shape)

# Read second sheet if exists
if len(xl.sheet_names) > 1:
    df2 = pd.read_excel(file_path, sheet_name=xl.sheet_names[1])
    print('\nSecond sheet preview:')
    print(df2.head())
    print('\nSecond sheet shape:', df2.shape)
