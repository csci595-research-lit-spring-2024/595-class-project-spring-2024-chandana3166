import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# Loading cleaned data csv file
data = pd.read_csv("cleaned_dataset.csv")

# Dataset is split into features and target
X = data.drop('target', axis=1)
y = data['target']

# 0.2 indicates 20 % data for testing and rest 80% for training the models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and test SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_scaled, y_train)

# Evaluate SVM model on training set
svm_train_predictions = svm_model.predict(X_train_scaled)
svm_train_accuracy = accuracy_score(y_train, svm_train_predictions)
svm_train_precision = precision_score(y_train, svm_train_predictions)
svm_train_recall = recall_score(y_train, svm_train_predictions)
svm_train_f1 = f1_score(y_train, svm_train_predictions)

# Evaluate SVM model on testing set
svm_test_predictions = svm_model.predict(X_test_scaled)
svm_test_accuracy = accuracy_score(y_test, svm_test_predictions)
svm_test_precision = precision_score(y_test, svm_test_predictions)
svm_test_recall = recall_score(y_test, svm_test_predictions)
svm_test_f1 = f1_score(y_test, svm_test_predictions)

print("SVM Training Accuracy:", svm_train_accuracy)
print("SVM Training Precision:", svm_train_precision)
print("SVM Training Recall:", svm_train_recall)
print("SVM Training F1 Score:", svm_train_f1)

print("\nSVM Testing Accuracy:", svm_test_accuracy)
print("SVM Testing Precision:", svm_test_precision)
print("SVM Testing Recall:", svm_test_recall)
print("SVM Testing F1 Score:", svm_test_f1)

# Define the parameter grid to search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Instantiate the GridSearchCV object
rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)

# Perform the grid search
rf_grid_search.fit(X_train, y_train)

# Get the best parameters and best estimator
best_params = rf_grid_search.best_params_
best_rf_model = rf_grid_search.best_estimator_

# Use the best model to predict on training and testing sets
rf_train_predictions = best_rf_model.predict(X_train)
rf_test_predictions = best_rf_model.predict(X_test)

# Evaluate Random Forest model with best parameters on training and testing sets
rf_train_accuracy = accuracy_score(y_train, rf_train_predictions)
rf_train_precision = precision_score(y_train, rf_train_predictions)
rf_train_recall = recall_score(y_train, rf_train_predictions)
rf_train_f1 = f1_score(y_train, rf_train_predictions)

rf_test_accuracy = accuracy_score(y_test, rf_test_predictions)
rf_test_precision = precision_score(y_test, rf_test_predictions)
rf_test_recall = recall_score(y_test, rf_test_predictions)
rf_test_f1 = f1_score(y_test, rf_test_predictions)

print("\nRandom Forest Training Accuracy:", rf_train_accuracy)
print("Random Forest Training Precision:", rf_train_precision)
print("Random Forest Training Recall:", rf_train_recall)
print("Random Forest Training F1 Score:", rf_train_f1)

print("\nRandom Forest Testing Accuracy:", rf_test_accuracy)
print("Random Forest Testing Precision:", rf_test_precision)
print("Random Forest Testing Recall:", rf_test_recall)
print("Random Forest Testing F1 Score:", rf_test_f1)

def plot_metrics_and_confusion_matrix(model_name, train_accuracy, train_precision, train_recall, train_f1, test_accuracy, test_precision, test_recall, test_f1, train_confusion_matrix, test_confusion_matrix):
    # Plot metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    train_scores = [train_accuracy, train_precision, train_recall, train_f1]
    test_scores = [test_accuracy, test_precision, test_recall, test_f1]

    plt.figure(figsize=(18, 8))
    plt.subplot(1, 2, 1)
    plt.bar(metrics, train_scores, color='b', alpha=0.7, label='Training')
    plt.bar(metrics, test_scores, color='r', alpha=0.7, label='Testing')
    plt.ylim(0, 1)
    plt.legend()
    plt.title(f'{model_name} Metrics Comparison')

    # Plot confusion matrix
    plt.subplot(1, 2, 2)
    plt.title(f'{model_name} Confusion Matrix')
    plt.subplot(2, 1, 1)
    sns.heatmap(train_confusion_matrix, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Training Dataset')
    plt.subplot(2, 1, 2)
    sns.heatmap(test_confusion_matrix, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Testing Dataset')

    plt.tight_layout()
    plt.show()

# SVM metrics and confusion matrix
svm_train_confusion_matrix = confusion_matrix(y_train, svm_train_predictions)
svm_test_confusion_matrix = confusion_matrix(y_test, svm_test_predictions)
plot_metrics_and_confusion_matrix('SVM', svm_train_accuracy, svm_train_precision, svm_train_recall, svm_train_f1, svm_test_accuracy, svm_test_precision, svm_test_recall, svm_test_f1, svm_train_confusion_matrix, svm_test_confusion_matrix)

# Random Forest metrics and confusion matrix
rf_train_confusion_matrix = confusion_matrix(y_train, rf_train_predictions)
rf_test_confusion_matrix = confusion_matrix(y_test, rf_test_predictions)
plot_metrics_and_confusion_matrix('Random Forest', rf_train_accuracy, rf_train_precision, rf_train_recall, rf_train_f1, rf_test_accuracy, rf_test_precision, rf_test_recall, rf_test_f1, rf_train_confusion_matrix, rf_test_confusion_matrix)
