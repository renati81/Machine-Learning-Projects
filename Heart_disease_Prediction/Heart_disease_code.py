# Importing Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Dataset
file_path = r"C:\Users\HP\OneDrive\Desktop\ML PROJECT\Heart_disease_dataset.csv"
dataset = pd.read_csv(file_path)

# Display Basic Info
print("\n Initial Dataset Information:")
print(dataset.info())
print(f"Dataset Shape: {dataset.shape}")

# Ensuring All Columns are Numeric
assert all(dataset.dtypes != 'object'), "Dataset contains non-numeric data!"

# Handling Missing Values (Replace with Mean)
dataset.fillna(dataset.mean(), inplace=True)

# Removing Duplicates
dataset = dataset.drop_duplicates()

# Handling Outliers Using Z-score
z_scores = np.abs(stats.zscore(dataset))
dataset = dataset[(z_scores < 3).all(axis=1)]  # Keeping values within 3 standard deviations

# Convert Target Column to Integer (Ensure it's Binary Classification)
dataset['target'] = dataset['target'].astype(int)

# Feature Scaling (Standardization)
scaler = StandardScaler()
features = dataset.drop(columns=['target'])  # Features Only
dataset[features.columns] = scaler.fit_transform(features)

# Save Cleaned Dataset
cleaned_file_path = r"C:\Users\HP\OneDrive\Desktop\ML PROJECT\Cleaned_Heart_Disease_Prediction.csv"
dataset.to_csv(cleaned_file_path, index=False)
print("\nData Preprocessing & Cleaning Completed. Cleaned Data Saved.")

# Exploratory Data Analysis (EDA)
print("\nPerforming Exploratory Data Analysis")

# Distribution Plots
plt.figure(figsize=(12, 8))
for i, col in enumerate(dataset.columns[:-1]):
    plt.subplot(4, 4, i + 1)
    sns.histplot(dataset[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

# Box Plots
plt.figure(figsize=(12, 8))
for i, col in enumerate(dataset.columns[:-1]):
    plt.subplot(4, 4, i + 1)
    sns.boxplot(x=dataset[col])
    plt.title(f"Box Plot of {col}")
plt.tight_layout()
plt.show()

# Correlation Matrix Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(dataset.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Splitting Features & Target
X = dataset.drop(columns=['target'])
Y = dataset['target']

# Splitting the Data: 60% Train, 20% Validation, 20% Test
X_train, X_val_test, Y_train, Y_val_test = train_test_split(X, Y, test_size=0.40, stratify=Y, random_state=3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size=0.50, stratify=Y_val_test, random_state=3)

# Display Dataset Sizes
print(f"\nDataset Sizes:\n- Train: {X_train.shape}\n- Validation: {X_val.shape}\n- Test: {X_test.shape}")

#MODEL TRAINING
#1.Logistic Regression
log_model = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=3)
log_model.fit(X_train, Y_train)

# Predictions
Y_test_pred_log = log_model.predict(X_test)

# Accuracy
log_accuracy = accuracy_score(Y_test, Y_test_pred_log)

# Print Results
print(f"\nLogistic Regression Accuracy: {log_accuracy:.4f}")
print("Classification Report:\n", classification_report(Y_test, Y_test_pred_log))
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_test_pred_log))
print("="*60)

#2.Support Vector Machine(SVM)
svm_model = SVC(kernel='linear', random_state=3)
svm_model.fit(X_train, Y_train)

# Predictions
Y_test_pred_svm = svm_model.predict(X_test)

# Accuracy
svm_accuracy = accuracy_score(Y_test, Y_test_pred_svm)

# Print Results
print(f"\nSVM Accuracy: {svm_accuracy:.4f}")
print("Classification Report:\n", classification_report(Y_test, Y_test_pred_svm))
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_test_pred_svm))
print("="*60)

#3.Decision Tree
dt_model = DecisionTreeClassifier(random_state=3)
dt_model.fit(X_train, Y_train)

# Predictions
Y_test_pred_dt = dt_model.predict(X_test)

# Accuracy
dt_accuracy = accuracy_score(Y_test, Y_test_pred_dt)

# Print Results
print(f"\nDecision Tree Accuracy: {dt_accuracy:.4f}")
print("Classification Report:\n", classification_report(Y_test, Y_test_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_test_pred_dt))
print("="*60)

#4.Random Forest
# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=3)
rf_model.fit(X_train, Y_train)

# Predictions
Y_test_pred_rf = rf_model.predict(X_test)

# Accuracy
rf_accuracy = accuracy_score(Y_test, Y_test_pred_rf)

# Print Results
print(f"\nRandom Forest Accuracy: {rf_accuracy:.4f}")
print("Classification Report:\n", classification_report(Y_test, Y_test_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_test_pred_rf))
print("="*60)

#5.XGBoost
xgb_model = XGBClassifier(eval_metric='logloss', random_state=3)
xgb_model.fit(X_train, Y_train)

# Predictions
Y_test_pred_xgb = xgb_model.predict(X_test)

# Accuracy
xgb_accuracy = accuracy_score(Y_test, Y_test_pred_xgb)

# Print Results
print(f"\nXGBoost Accuracy: {xgb_accuracy:.4f}")
print("Classification Report:\n", classification_report(Y_test, Y_test_pred_xgb))
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_test_pred_xgb))
print("="*60)

#Comparision Plot
# Accuracy scores
accuracy_scores = {
    "Logistic Regression": log_accuracy,
    "SVM": svm_accuracy,
    "Decision Tree": dt_accuracy,
    "Random Forest": rf_accuracy,
    "XGBoost": xgb_accuracy
}

# Convert dictionary to DataFrame
accuracy_df = pd.DataFrame(list(accuracy_scores.items()), columns=['Model', 'Accuracy'])

# Plot comparison
plt.figure(figsize=(10, 5))
sns.barplot(data=accuracy_df, x='Model', y='Accuracy')

# Labels and Formatting
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy Score")
plt.title("Comparison of Model Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.show()
