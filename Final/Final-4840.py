import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import os

# Print the current working directory
print("Current working directory:", os.getcwd())

# Load the drebin dataset
df_drebin = pd.read_csv('/run/media/Brandon/D820DA8720DA6BCE/Classes/CS4840/Final/drebin.csv')  # Replace with the actual path to your drebin.csv file

# Convert class label -1 to 0
df_drebin['class'] = df_drebin['class'].replace(-1, 0)

# Separate features and labels
X = df_drebin.drop('class', axis=1)
y = df_drebin['class']

# Split into training (60%), validation (20%), and test (20%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")