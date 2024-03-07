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

# Display the DataFrame
print(df)
