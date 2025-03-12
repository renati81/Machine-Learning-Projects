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

#Read all three datasets provided
def load_data(data_source):
    data_files = {
        "hayes-roth": "Hayes-Roth_dataset.csv",
        "car": "Car-Evaluation_dataset.csv",
        "breast-cancer": "Breast-cancer_dataset.csv",
    }

    data = pd.read_csv(data_files[data_source], header=None)
    print(f"\nLoaded {data_source} data. Dimensions: {data.shape}")
    print(data.head(5))
    return data

#Handling Missing values, removing outliers and encoding categorical data
def preprocessing_data(data):
    print("\nIdentifying missing values before processing the data:")
    missing_values = data.isnull().sum()
    
    # Check if there are any missing values
    if missing_values.sum() == 0:
        print("No missing values found. \nSkipping missing value handling.")
    else:
        # Display which columns have missing values
        for column, count in missing_values.items():
            if count > 0:
                print(f"Column '{column}' has {count} missing value(s).")
        
        # Handle missing values only when they exist
        for column in data.columns:
            if data[column].isnull().sum() > 0:
                if pd.api.types.is_object_dtype(data[column]):
                    data[column] = data[column].fillna(data[column].mode()[0])
                else:
                    median_value = data[column].median()
                    data[column] = data[column].fillna(median_value)
        
        # Verify missing values were handled
        missing_values_after = data.isnull().sum()
        if missing_values_after.sum() == 0:
            print("All missing values successfully corrected.")
        else:
            print("Missing values remain after processing:")
            print(missing_values_after[missing_values_after > 0])
    
    # Handle outliers 
    numerical_data = data.select_dtypes(include=[np.number])
    if not numerical_data.empty:  
        z_scores = np.abs(zscore(numerical_data))
        data = data[(z_scores < 3).all(axis=1)]
        print(f"\nOutliers filtered. Updated data dimensions: {data.shape}")

    # Encode categorical features
    for column in data.select_dtypes(include=["object"]).columns:
        label_encoder = LabelEncoder()
        data[column] = label_encoder.fit_transform(data[column])

    print("\nData preview:")
    print(data.head(5))
    return data

#standarization and feature selection
def prepare_features(data):
    features = data.iloc[:, :-1].values
    target = data.iloc[:, -1].values

    feature_scaler = StandardScaler()
    features = feature_scaler.fit_transform(features)

    print("Features prepared. Statistics:")
    print(f"Average: {features.mean():.4f}, Standard Deviation: {features.std():.4f}")

    return features, target

#Calculation of euclidean distance
def euclidean_distance(point1, point2):
    return sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

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
        self.distance_function = euclidean_distance
        
    def train(self, training_features, training_labels):
        self.training_features = training_features
        self.training_labels = training_labels

    def test(self, test_features):
        return [self.test_instances(feature) for feature in test_features]

    def test_instances(self, test_instance):
        distances = [(self.distance_function(test_instance, train_instance), label)
                     for train_instance, label in zip(self.training_features, self.training_labels)]
        distances.sort(key=lambda item: item[0])
        nearest_neighbors = distances[:self.neighbors]

        class_votes = {}
        for dist, label in nearest_neighbors:
            weight = 1 / (dist + 1e-9) if self.use_weights else 1
            class_votes[label] = class_votes.get(label, 0) + weight

        highest_votes = max(class_votes.values())
        winning_classes = [class_label for class_label, votes in class_votes.items() if votes == highest_votes]

        return random.choice(winning_classes) if len(winning_classes) > 1 else winning_classes[0]


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

    print(f"Optimal K: {best_k} with accuracy: {top_accuracy:.4f}")
    return best_k


#K-Fold Cross Validation & Comparison
def k_fold_cross_validation(X, y, k_values, folds=10):
    """Runs K-Fold Cross Validation on both My KNN and Sklearn KNN."""
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
      for k in knn_results.keys():
        t_stat, p_value = ttest_rel(knn_results[k], sklearn_results[k])
        print(f"\nPaired T-Test for K={k}")
        print(f"T-statistic = {t_stat:.4f}")
        print(f"P-value = {p_value:.4f}")

        if p_value < 0.05:
            print("Significant Difference Found!")
        else:
            print("No Significant Difference Found.")

#Main Execution
neighbor_counts = [1, 3, 5, 7, 9]
using_weights = True
data_sources = ["hayes-roth", "car", "breast-cancer"]
performance_results = {}

for source in data_sources:
    print(f"\nProcessing Data Source: {source}")

    data_frame = load_data(source)
    data_frame = preprocessing_data(data_frame)
    features, labels = prepare_features(data_frame)

    train_features, temp_features, train_labels, temp_labels = train_test_split(features, labels, test_size=0.4, random_state=42)
    validation_features, test_features, validation_labels, test_labels = train_test_split(temp_features, temp_labels, test_size=0.5, random_state=42)

    optimal_k = evaluate_performance(train_features, validation_features, train_labels, validation_labels, neighbor_counts, using_weights)

    knn_results, sklearn_results = k_fold_cross_validation(features, labels, neighbor_counts)

    hypothesis_test(knn_results, sklearn_results)
    
    final_model = KNN(neighbors=optimal_k, use_weights=using_weights)
    final_model.train(train_features, train_labels)
    final_predictions = final_model.test(test_features)

    final_accuracy = accuracy_score(test_labels, final_predictions)
    print(f"Final Test Accuracy for {source} with K={optimal_k}: {final_accuracy:.4f}")

    performance_results[source] = final_accuracy
