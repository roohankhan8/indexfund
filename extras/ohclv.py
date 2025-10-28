import pyreadr

# Read the .rds file
result = pyreadr.read_r("ohclv_data.rds")

# Extract the object (assuming it's a single object in the .rds file)
data = result[None]

# Print or process the data
print(data)

data.to_csv("output.csv", index=False)