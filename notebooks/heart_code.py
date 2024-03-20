import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt



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


# Load the cleaned dataset
df = pd.read_csv('heart_cleaned.csv')

# Normalize the columns
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']])
normalized_df = pd.DataFrame(normalized_data, columns=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'])

# Standardize the columns
scaler = StandardScaler()
standardized_data = scaler.fit_transform(df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']])
standardized_df = pd.DataFrame(standardized_data, columns=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'])

# One-hot encode categorical variables
one_hot_encoded_df = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

# Combine all dataframes and save to single csv file 
final_df = pd.concat([normalized_df, standardized_df, one_hot_encoded_df], axis=1)
final_df.to_csv('transformed_heartdata.csv', index=False)



print("The cleaned dataset has been transformed using normalization, standardization, and one-hot encoding of categorical values to improve data processing.")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(final_df)

# Feature Selection
x = final_df.drop(columns=['target']) 
y = final_df['target']
selector = SelectKBest(score_func=f_classif, k=5)  # Select top 5 features
selected_features = selector.fit_transform(x, y)

# Data Splitting
x_train, x_test, y_train, y_test = train_test_split(selected_features, y, test_size=0.2, random_state=42)

# Print the shape of the datasets
print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

# Define the data
data_shapes = {
    'x_train': x_train.shape,
    'y_train': y_train.shape,
    'x_test': x_test.shape,
    'y_test': y_test.shape
}

# Plot the shapes
plt.figure(figsize=(10, 6))
plt.bar(data_shapes.keys(), [shape[0] for shape in data_shapes.values()], color='skyblue')
plt.xlabel('Data Type')
plt.ylabel('Number of Samples')
plt.title('Shapes of Data')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



