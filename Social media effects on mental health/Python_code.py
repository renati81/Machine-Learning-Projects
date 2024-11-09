import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()  # Enables interactive mode
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Corrected file path - use the absolute path to the dataset
train_data_path = r"C:\Users\HP\Desktop\damt project\Train_test2.csv"  # Update this path as needed

# Load the dataset
try:
    train_df = pd.read_csv(train_data_path)
except FileNotFoundError:
    print(f"File not found: {train_data_path}")
    exit()

# Checking columns and dataset structure
print(train_df.columns)
print(train_df.info())
print(train_df.describe())
print(train_df.isnull().sum())

# Plot age distribution if 'Age' column exists
if 'Age' in train_df.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(train_df['Age'].dropna(), kde=True, bins=30)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.savefig('age_distribution.png') #save plot as image
else:
    print("Column 'Age' not found in the dataset.")

# Plot gender distribution if 'Gender' column exists
if 'Gender' in train_df.columns:
    plt.figure(figsize=(7, 5))
    sns.countplot(x='Gender', data=train_df)
    plt.title('Gender Distribution')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.savefig('age_distribution.png') #save plot as image
else:
    print("Column 'Gender' not found in the dataset.")

# Plot average daily usage time by platform if 'Platform' and 'Daily_Usage_Time (minutes)' exist
if 'Platform' in train_df.columns and 'Daily_Usage_Time (minutes)' in train_df.columns:
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Platform', y='Daily_Usage_Time (minutes)', data=train_df, estimator=np.mean)
    plt.title('Average Daily Usage Time by Platform')
    plt.xlabel('Platform')
    plt.ylabel('Daily Usage Time (minutes)')
    plt.savefig('age_distribution.png') #save plot as image
else:
    print("Column 'Platform' or 'Daily_Usage_Time (minutes)' not found in the dataset.")

# Plot dominant emotion distribution if 'Dominant_Emotion' exists
if 'Dominant_Emotion' in train_df.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Dominant_Emotion', data=train_df)
    plt.title('Dominant Emotion Distribution')
    plt.xlabel('Dominant Emotion')
    plt.ylabel('Count')
    plt.savefig('age_distribution.png') #save plot as image
else:
    print("Column 'Dominant_Emotion' not found in the dataset.")

# ---------------------- Pre-Processing of Data --------------------------------------------

# 1 One Hot Encoding for categorical columns if they exist
if 'Gender' in train_df.columns and 'Platform' in train_df.columns:
    train_df = pd.get_dummies(train_df, columns=['Gender', 'Platform'], drop_first=True)
else:
    print("One or both of the columns 'Gender' and 'Platform' not found for one-hot encoding.")

# 2 Label Encoding for 'Dominant_Emotion' if it exists
if 'Dominant_Emotion' in train_df.columns:
    le = LabelEncoder()
    train_df['Dominant_Emotion'] = le.fit_transform(train_df['Dominant_Emotion'])
else:
    print("Column 'Dominant_Emotion' not found for label encoding.")

# Dropping 'User_ID' if it exists
if 'User_ID' in train_df.columns:
    train_df = train_df.drop(columns=['User_ID'])
else:
    print("Column 'User_ID' not found for dropping.")

# Ensure 'Age' is numeric and handle missing values
train_df['Age'] = pd.to_numeric(train_df['Age'], errors='coerce')
mean_value_train = train_df['Age'].mean()
train_df['Age'] = train_df['Age'].fillna(mean_value_train)

# Display shape and columns after preprocessing
pd.set_option('display.max_rows', 1001)
print(train_df.shape)
print(train_df.columns)
print(train_df['Dominant_Emotion'])

# ---------- Importing Model ---------------------------------

# Define X and y for the model if 'Dominant_Emotion' is present
if 'Dominant_Emotion' in train_df.columns:
    X = train_df.drop(columns='Dominant_Emotion')
    y = train_df['Dominant_Emotion']
    
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Train the RandomForestClassifier
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train) 
    pred = rfc.predict(x_test)

    # Display predictions and accuracy
    print(pred)
    print("Accuracy:", accuracy_score(pred, y_test))
else:
    print("Target column 'Dominant_Emotion' not found in dataset.")
