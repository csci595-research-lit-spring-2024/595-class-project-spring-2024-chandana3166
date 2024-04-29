import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Loading cleaned data csv file
data = pd.read_csv("cleaned_dataset.csv")

#Dataset is split into features and target
X = data.drop('target', axis=1)
y = data['target']

# 0.2 indicates 20 % data for testing and rest 80% for training the models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and test SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_scaled, y_train)
svm_predictions = svm_model.predict(X_test_scaled)

# Evaluate SVM model
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_precision = precision_score(y_test, svm_predictions)
svm_recall = recall_score(y_test, svm_predictions)
svm_f1 = f1_score(y_test, svm_predictions)
svm_confusion_matrix = confusion_matrix(y_test, svm_predictions)

print("SVM Accuracy:", svm_accuracy)
print("SVM Precision:", svm_precision)
print("SVM Recall:", svm_recall)
print("SVM F1 Score:", svm_f1)
print("SVM Confusion Matrix:\n", svm_confusion_matrix)

# Train and test Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Evaluate Random Forest model
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions)
rf_recall = recall_score(y_test, rf_predictions)
rf_f1 = f1_score(y_test, rf_predictions)
rf_confusion_matrix = confusion_matrix(y_test, rf_predictions)

print("\nRandom Forest Accuracy:", rf_accuracy)
print("Random Forest Precision:", rf_precision)
print("Random Forest Recall:", rf_recall)
print("Random Forest F1 Score:", rf_f1)
print("Random Forest Confusion Matrix:\n", rf_confusion_matrix)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Define function to plot metrics and confusion matrix
def plot_metrics_and_confusion_matrix(model_name, accuracy, precision, recall, f1, confusion_matrix):
    # Plot metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    scores = [accuracy, precision, recall, f1]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.barplot(x=metrics, y=scores)
    plt.ylim(0, 1)
    plt.title(f'{model_name} Metrics')

    # Plot confusion matrix
    plt.subplot(1, 2, 2)
    sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} Confusion Matrix')

    plt.tight_layout()
    plt.show()

# SVM metrics and confusion matrix
plot_metrics_and_confusion_matrix('SVM', svm_accuracy, svm_precision, svm_recall, svm_f1, svm_confusion_matrix)

# Random Forest metrics and confusion matrix
plot_metrics_and_confusion_matrix('Random Forest', rf_accuracy, rf_precision, rf_recall, rf_f1, rf_confusion_matrix)

