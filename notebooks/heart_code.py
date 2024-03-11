import pandas as pd

# Load the data from the CSV file
df = pd.read_csv('heart.csv')

# Specify the data types
data_types = {
    'age': 'int',
    'sex': 'int',
    'cp': 'int',
    'trestbps': 'int',
    'chol': 'int',
    'fbs': 'int',
    'restecg': 'int',
    'thalach': 'int',
    'exang': 'int',
    'oldpeak': 'float',
    'slope': 'int',
    'ca': 'int',
    'thal': 'int',
    'target': 'int'
}

# Convert the DataFrame columns to the specified data types
df = df.astype(data_types)

# Display the DataFrame and datatypes
print(df)
print(df.dtypes)

# Checking for missing values
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

#Fill the mean of the values to handle missing values
df_filled = df.fillna(df.mean())

# drop rows with missing values
# df_cleaned = df.dropna()

# Get the number of rows before removing duplicates
rows_before = df.shape[0]

# Remove duplicates
df.drop_duplicates(inplace=True)

# Get the number of rows after removing duplicates
rows_after = df.shape[0]

# Calculate the number of duplicates removed
duplicates_removed = rows_before - rows_after

# Save the cleaned DataFrame back to a CSV file
df.to_csv('cleaned_dataset.csv', index=False)

print(f"Removed {duplicates_removed} duplicates.")
