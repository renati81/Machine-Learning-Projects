import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from math import sqrt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import ttest_rel
from sklearn.neighbors import KNeighborsClassifier  

##Read all three datasets provided
def load_data(data_source):
    data_files = {
        "hayes-roth": "Hayes-Roth_dataset.csv",
        "car": "Car-Evaluation_dataset.csv",
        "breast-cancer": "Breast-cancer_dataset.csv",
    }

    data = pd.read_csv(data_files[data_source], header=None)
    print(f"\nLoaded {data_source} dataset. Shape: {data.shape}")
    print(data.head(5))
    return data

#Handling Missing values, removing outliers and encoding categorical data
def preprocessing_data(data):
    
    # Handle missing values
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            if data[col].dtype == "object":
                data[col].fillna(data[col].mode()[0], inplace=True)
            else:
                data[col].fillna(data[col].median(), inplace=True)

    # Encode categorical variables
    for col in data.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    # Normalize features
    features = data.iloc[:, :-1].values
    target = data.iloc[:, -1].values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return features, target

#Calculation of manhattan distance
def manhattan_distance(point1, point2):
    return sum(abs(a - b) for a, b in zip(point1, point2))

#Calculation of minkowski distance
def minkowski_distance(point1, point2, p=3):
    return sum(abs(a - b) ** p for a, b in zip(point1, point2)) ** (1 / p)

#K nearest negibhour classifier implementation
class KNN:
    
    def __init__(self, neighbors=3, distance_type="euclidean", use_weights=False):
        self.neighbors = neighbors
        self.use_weights = use_weights

    def train(self, training_features, training_labels):
        self.training_features = training_features
        self.training_labels = training_labels

    def test(self, test_features):
        return [self.test_instances(feature) for feature in test_features]

    def test_instances(self, test_instance):
        distances = [(self._euclidean_distance(test_instance, train_instance), label)
                     for train_instance, label in zip(self.training_features, self.training_labels)]
        distances.sort(key=lambda item: item[0])
        nearest_neighbors = distances[:self.neighbors]

        class_votes = {}
        for dist, label in nearest_neighbors:
            weight = 1 / (dist + 1e-9) if self.use_weights else 1
            class_votes[label] = class_votes.get(label, 0) + weight

        return max(class_votes, key=class_votes.get)

    def _euclidean_distance(self, x1, x2):
        return sqrt(sum((a - b) ** 2 for a, b in zip(x1, x2)))

#Evaluate model performance
def evaluate_performance(train_features, validation_features, train_labels, validation_labels, k_values, use_weights):
    best_k = k_values[0]
    top_accuracy = 0

    for k in k_values:
        model = KNN(neighbors=k, use_weights=use_weights)
        model.train(train_features, train_labels)
        predictions = model.test(validation_features)
        accuracy = accuracy_score(validation_labels, predictions)

        print(f"\nK={k}, Validation Accuracy: {accuracy:.4f}")
        if accuracy > top_accuracy:
            best_k = k
            top_accuracy = accuracy

    print(f"Optimal K: {best_k} with Accuracy: {top_accuracy:.4f}")
    return best_k

# K-Fold Cross Validation & Comparison
def k_fold_cross_validation(X, y, k_values, folds=10):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    knn_results = {k: [] for k in k_values}
    sklearn_results = {k: [] for k in k_values}

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for k in k_values:
            # KNN
            my_knn = KNN(neighbors=k, use_weights=True)
            my_knn.train(X_train, y_train)
            y_pred_knn = my_knn.test(X_test)
            knn_results[k].append(accuracy_score(y_test, y_pred_knn))

            # Sklearn KNN
            sklearn_knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
            sklearn_knn.fit(X_train, y_train)
            y_pred_sklearn = sklearn_knn.predict(X_test)
            sklearn_results[k].append(accuracy_score(y_test, y_pred_sklearn))

    
    for k in k_values:
        print(f"K={k}")
        print(f"My KNN (Avg Accuracy): {np.mean(knn_results[k]):.4f}")
        print(f"Sklearn KNN (Avg Accuracy): {np.mean(sklearn_results[k]):.4f}")

    return knn_results, sklearn_results


#Hypothesis Testing
def hypothesis_test(knn_results, sklearn_results):
    """Performs Paired T-Test to compare My KNN and Sklearn KNN."""
    for k in knn_results.keys():
        t_stat, p_value = ttest_rel(knn_results[k], sklearn_results[k])
        print(f"Paired T-Test for K={k}")
        print(f"t-statistic = {t_stat:.4f}")
        print(f"p-value = {p_value:.4f}")

        if p_value < 0.05:
            print("   ✅ Significant Difference Found!")
        else:
            print("   ❌ No Significant Difference Found.")


# Main Execution
neighbor_counts = [1, 3, 5, 7, 9]
data_sources = ["hayes-roth", "car", "breast-cancer"]
performance_results = {}

for source in data_sources:
    print(f"\n Processing Data Source: {source}")
    data_frame = load_data(source)
    features, labels = preprocessing_data(data_frame)

    
    train_features, temp_features, train_labels, temp_labels = train_test_split(features, labels, test_size=0.4, random_state=42)
    validation_features, test_features, validation_labels, test_labels = train_test_split(temp_features, temp_labels, test_size=0.5, random_state=42)

    
    optimal_k = evaluate_performance(train_features, validation_features, train_labels, validation_labels, neighbor_counts, True)

    knn_results, sklearn_results = k_fold_cross_validation(features, labels, neighbor_counts)

    # Run Hypothesis Testing
    hypothesis_test(knn_results, sklearn_results)
