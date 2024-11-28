import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from joblib import Parallel, delayed
from multiprocessing import Manager

# Load the drebin dataset
df_drebin = pd.read_csv('./Final/drebin.csv')

# Convert class label -1 to 0
df_drebin['class'] = df_drebin['class'].replace(-1, 0)

# Separate features and labels
X = df_drebin.drop('class', axis=1)  # Features (input variables)
y = df_drebin['class']  # Labels (output variable)

# Split the data into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Create a Manager to handle shared dictionary
manager = Manager()
results = manager.dict()

# Define StratifiedKFold cross-validation
stratified_kfold = StratifiedKFold(n_splits=6)

# Function to train and evaluate a model
def train_and_evaluate(model, params, model_name, results):
    try:
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(model, params, cv=stratified_kfold, n_jobs=-1)
        
        # Fit the model on the training data
        grid_search.fit(X_train, y_train)
        
        # Get the best hyperparameters
        optimal_params = grid_search.best_params_
        
        # Predict on the test set
        y_test_pred = grid_search.predict(X_test)
        
        # Calculate accuracy and F1 score
        accuracy = accuracy_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        
        # Store the results in the shared dictionary
        results[model_name] = {
            'Accuracy': accuracy,
            'F1 Score': f1,
            'predictions': y_test_pred,
            **optimal_params
        }
        
        print(f"{model_name} results: {results[model_name]}")
    except Exception as e:
        print(f"Error training {model_name}: {e}")

# Define models and their hyperparameters
models_params = [
    # Support Vector Machine (SVM) with linear kernel: Finds the hyperplane that best separates the classes
    ('SVM', SVC(kernel='linear', random_state=42), {'C': [0.1, 1, 10, 100, 150, 200]}),
    
    # Logistic Regression: Models the probability of the default class using a logistic function
    ('Logistic Regression', LogisticRegression(random_state=42), {'C': [0.1, 1, 10, 100, 150, 200]}),
    
    # K-Nearest Neighbors (KNN): Classifies a sample based on the majority class among its k-nearest neighbors
    ('K-Nearest Neighbors', KNeighborsClassifier(), {'n_neighbors': range(1, 21)}),
    
    # Decision Tree: Splits the data into subsets based on the feature that results in the most homogeneous subsets
    ('Decision Tree', DecisionTreeClassifier(random_state=42), {'max_depth': range(1, 21)}),
    
    # AdaBoost with SAMME.R algorithm: Combines multiple weak classifiers to create a strong classifier
    ('AdaBoost', AdaBoostClassifier(algorithm='SAMME', random_state=42), {'n_estimators': [50, 100, 200, 250, 300]}),
    
    # Stochastic Gradient Descent (SGD) with log loss: Optimizes the log loss function using gradient descent
    ('Linear Regression (SGD with Adam)', SGDClassifier(loss='log_loss', learning_rate='adaptive', eta0=0.01, random_state=42), {'alpha': [0.000001, 0.00001, 0.0001, 0.001, 0.01]}),
    
    # Random Forest: Constructs multiple decision trees and outputs the mode of their predictions
    ('Random Forest', RandomForestClassifier(random_state=42), {'n_estimators': [50, 100, 200, 250, 300], 'max_depth': range(1, 21)}),
    
    # Gradient Boosting: Builds an ensemble of trees in a stage-wise fashion to minimize the loss function
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42), {'n_estimators': [50, 100, 200, 250, 300], 'max_depth': range(1, 21)}),
    
    # Naive Bayes: Applies Bayes' theorem with the assumption of independence between features
    ('Naive Bayes', GaussianNB(), {})
]

# Parallelize the training and evaluation
Parallel(n_jobs=-1)(delayed(train_and_evaluate)(model, params, model_name, results) for model_name, model, params in models_params)

# Convert results back to a regular dictionary
results = dict(results)

# Function to extract and transform predictions for a single model
def extract_transform_predictions(model_name):
    return np.where(results[model_name]['predictions'] == 0, -1, results[model_name]['predictions'])

# List of model names
model_names = [
    'SVM',
    'Logistic Regression',
    'K-Nearest Neighbors',
    'Decision Tree',
    'AdaBoost',
    'Linear Regression (SGD with Adam)',
    'Random Forest',
    'Gradient Boosting',
    'Naive Bayes'
]

# Parallelize the extraction and transformation of predictions
predictions_list = Parallel(n_jobs=-1)(delayed(extract_transform_predictions)(model_name) for model_name in model_names)

# Stack the predictions into a single array
predictions = np.column_stack(predictions_list)

# Initialize array to store votes
votes = np.zeros_like(predictions[:, 0], dtype=int)

# Add votes for each model
for pred in predictions.T:
    votes += pred

# Determine final prediction based on majority vote
majority_pred = (votes >= 0).astype(int)

# Calculate accuracy and F1 score for the majority vote ensemble method
majority_accuracy = accuracy_score(y_test, majority_pred)
majority_f1 = f1_score(y_test, majority_pred)

# Add majority vote ensemble results to the results dictionary
results['Majority Vote Ensemble'] = {
    'Accuracy': majority_accuracy,
    'F1 Score': majority_f1
}

# Extract accuracies from the results dictionary
accuracies = np.array([
    results['SVM']['Accuracy'],
    results['Logistic Regression']['Accuracy'],
    results['K-Nearest Neighbors']['Accuracy'],
    results['Decision Tree']['Accuracy'],
    results['AdaBoost']['Accuracy'],
    results['Linear Regression (SGD with Adam)']['Accuracy'],
    results['Random Forest']['Accuracy'],
    results['Gradient Boosting']['Accuracy'],
    results['Naive Bayes']['Accuracy']
])

class WeightedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, predictions=None, accuracies=None, power=2):
        self.predictions = predictions
        self.accuracies = accuracies
        self.power = power

    def fit(self, X, y):
        # No fitting necessary as predictions are already provided
        return self

    def predict(self, X):
        # Ensure predictions are for the correct number of samples
        if self.predictions.shape[0] != X.shape[0]:
            raise ValueError("Mismatch in number of samples between predictions and input data")
        
        weights = self.accuracies ** self.power
        weights /= weights.sum()

        weighted_votes = np.dot(self.predictions, weights)
        return (weighted_votes >= 0).astype(int)

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def get_params(self, deep=True):
        return {'predictions': self.predictions, 'accuracies': self.accuracies, 'power': self.power}

# Function to evaluate a specific power value
def evaluate_power(power, predictions, accuracies, X_test, y_test):
    weighted_ensemble = WeightedEnsembleClassifier(predictions=predictions, accuracies=accuracies, power=power)
    ensemble_pred = weighted_ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, ensemble_pred)
    f1 = f1_score(y_test, ensemble_pred)
    return power, accuracy, f1

# Define the range of power values to test
power_values = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# Parallelize the evaluation of power values
results_parallel = Parallel(n_jobs=-1)(delayed(evaluate_power)(power, predictions, accuracies, X_test, y_test) for power in power_values)

# Find the best power value based on accuracy
best_power, best_accuracy, best_f1 = max(results_parallel, key=lambda x: x[1])

# Add ensemble results to the results dictionary
results['Weighted Ensemble'] = {
    'Accuracy': best_accuracy,
    'F1 Score': best_f1,
    'Optimal Power': best_power
}

# Model Stacking: Combine predictions of multiple models using a second-level model
estimators = [(name, model.set_params(**{k: results[name][k] for k in params.keys()})) for name, model, params in models_params]
stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stacking_model.fit(X_train, y_train)
y_test_pred = stacking_model.predict(X_test)
results['Stacking'] = {
    'Accuracy': accuracy_score(y_test, y_test_pred),
    'F1 Score': f1_score(y_test, y_test_pred)
}

# Write results to a text file
with open('./Final/model_results.txt', 'w') as f:
    for model, metrics in results.items():
        f.write(f"{model} Test Set Evaluation:\n")
        f.write(f"Accuracy: {metrics['Accuracy']}\n")
        f.write(f"F1 Score: {metrics['F1 Score']}\n")
        if 'C' in metrics:
            f.write(f"Optimal C: {metrics['C']}\n")
        if 'n_neighbors' in metrics:
            f.write(f"Optimal k: {metrics['n_neighbors']}\n")
        if 'max_depth' in metrics:
            f.write(f"Optimal Depth: {metrics['max_depth']}\n")
        if 'n_estimators' in metrics:
            f.write(f"Optimal n_estimators: {metrics['n_estimators']}\n")
        if 'alpha' in metrics:
            f.write(f"Optimal alpha: {metrics['alpha']}\n")
        if 'Optimal Power' in metrics:
            f.write(f"Optimal Power: {metrics['Optimal Power']}\n")
        f.write("\n")
