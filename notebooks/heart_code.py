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

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Split the data into training and testing sets
X = final_df.drop(columns=['target'])
y = final_df['target']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(x_train, y_train)

# Predict using the SVM model
y_pred_svm = svm_model.predict(x_test)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

# Predict using the Random Forest model
y_pred_rf = rf_model.predict(x_test)

# Evaluate the models
svm_accuracy = accuracy_score(y_test, y_pred_svm)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print(f"SVM Accuracy: {svm_accuracy}")
print(f"Random Forest Accuracy: {rf_accuracy}")

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have trained SVM and Random Forest classifiers named svm_clf and rf_clf
# and x_train, x_test, y_train, y_test are your training and testing sets

# SVM predictions
svm_preds = svm_clf.predict(x_test)

# Random Forest predictions
rf_preds = rf_clf.predict(x_test)

# Confusion matrix for SVM
svm_cm = confusion_matrix(y_test, svm_preds)

# Confusion matrix for Random Forest
rf_cm = confusion_matrix(y_test, rf_preds)

# Accuracy for SVM and Random Forest
svm_acc = accuracy_score(y_test, svm_preds)
rf_acc = accuracy_score(y_test, rf_preds)

# F1 score for SVM and Random Forest
svm_f1 = f1_score(y_test, svm_preds)
rf_f1 = f1_score(y_test, rf_preds)

# Correlation matrix
correlation_matrix = df.corr()

# Plot confusion matrices
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.heatmap(svm_cm, annot=True, cmap='Blues', fmt='g')
plt.title('SVM Confusion Matrix')

plt.subplot(1, 2, 2)
sns.heatmap(rf_cm, annot=True, cmap='Blues', fmt='g')
plt.title('Random Forest Confusion Matrix')

plt.show()

# Print accuracy and F1 score
print("SVM Accuracy:", svm_acc)
print("SVM F1 Score:", svm_f1)
print("Random Forest Accuracy:", rf_acc)
print("Random Forest F1 Score:", rf_f1)

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()





