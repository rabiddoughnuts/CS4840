import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from joblib import Parallel, delayed
from multiprocessing import Manager
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

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

# Function to plot and save performance as hyperparameters change
def plot_performance(model_name, param_grid, cv_results):
    # Extract parameter combinations and their corresponding scores
    params = cv_results['params']
    mean_test_scores = cv_results['mean_test_score']
    mean_train_scores = cv_results['mean_train_score']

    # Create a list of parameter combinations as strings
    param_combinations = [str(param) for param in params]

    # Determine the y-axis limits
    all_scores = np.concatenate([mean_test_scores, mean_train_scores])
    y_min = np.floor(np.min(all_scores) * 1000) / 1000
    y_max = np.ceil(np.max(all_scores) * 1000) / 1000
    range_of_values = y_max - y_min

    # Adjust major ticks based on the range of values
    if range_of_values > 0.02:
        major_tick = 0.01
    else:
        major_tick = 0.005

    # Zoom in more per value
    zoom_factor = 0.001  # Adjust this factor to zoom in more
    y_min -= zoom_factor
    y_max += zoom_factor

    # Create a horizontal bar plot for accuracy
    plt.figure(figsize=(20, 12))  # Increase the figure size
    bar_width = 0.35
    index = np.arange(len(param_combinations))

    plt.barh(index, mean_test_scores, bar_width, color='blue', alpha=0.6, label='Mean Test Score')
    plt.barh(index + bar_width, mean_train_scores, bar_width, color='green', alpha=0.6, label='Mean Train Score')

    plt.ylabel('Parameter Combinations')
    plt.xlabel('Mean Score')
    plt.title(f'{model_name} - Performance')
    plt.yticks(index + bar_width / 2, param_combinations)
    plt.xlim(y_min, y_max)
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    plt.gca().xaxis.set_major_locator(MultipleLocator(major_tick))  # Major ticks based on range
    plt.gca().xaxis.set_minor_locator(MultipleLocator(0.001))  # Minor ticks for every unit
    plt.grid(axis='x', linestyle='--', linewidth=0.5, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{model_name}_performance.png')
    plt.close()

# Function to train and evaluate a model
def train_and_evaluate(model, param_grid, model_name, results):
    X_train_used, X_test_used = X_train, X_test

    grid_search = GridSearchCV(model, param_grid, cv=stratified_kfold, scoring='accuracy', n_jobs=-1, return_train_score=True)
    grid_search.fit(X_train_used, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_used)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[model_name] = {
        'Best Params': grid_search.best_params_,
        'Accuracy': accuracy,
        'F1 Score': f1
    }

    print(f"Optimal parameters for {model_name}: {grid_search.best_params_}")

    # Plot performance
    plot_performance(model_name, param_grid, grid_search.cv_results_)

# Define models and their hyperparameters
models_params = [
    # Support Vector Machine (SVM): Finds the hyperplane that best separates the classes
    ('SVM', SVC(random_state=42, max_iter=12000), {
        'C': [25, 50, 75],  # Regularization parameter
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel type (removed 'precomputed')
        'gamma': ['scale', 'auto'],  # Kernel coefficient
        'degree': [1, 2, 3],  # Degree of the polynomial kernel
        'coef0': [0.0, 0.01, 0.02]  # Independent term in kernel function
    }),

    # # Logistic Regression (liblinear): Models the probability of the default class using a logistic function
    # ('Logistic Regression (liblinear)', LogisticRegression(random_state=42, max_iter=2400, solver='liblinear'), {
    #     'C': [130],  # Inverse of regularization strength
    #     'penalty': ['l2'],  # Norm used in penalization
    #     'fit_intercept': [True],  # Whether to include intercept
    #     'intercept_scaling': [2]  # Scaling of the intercept
    # }),

    # # K-Nearest Neighbors (KNN): Classifies a sample based on the majority class among its k-nearest neighbors
    # ('K-Nearest Neighbors', KNeighborsClassifier(), {
    #     'n_neighbors': range(1, 3),  # Number of neighbors to use
    #     'weights': ['uniform', 'distance'],  # Weight function used in prediction
    #     'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm used to compute the nearest neighbors
    #     'leaf_size': [10, 20, 30],  # Leaf size passed to BallTree or KDTree
    #     'p': [1, 2],  # Power parameter for the Minkowski metric
    #     'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']  # Distance metric to use
    # }),

    # # K-Nearest Neighbors (wminkowski): Uses weighted Minkowski distance
    # ('K-Nearest Neighbors (wminkowski)', KNeighborsClassifier(), {
    #     'n_neighbors': range(1, 3),  # Number of neighbors to use
    #     'weights': ['uniform', 'distance'],  # Weight function used in prediction
    #     'algorithm': ['auto', 'ball_tree', 'brute'],  # Algorithm used to compute the nearest neighbors
    #     'leaf_size': [10, 20, 30],  # Leaf size passed to BallTree or KDTree
    #     'p': [1, 2],  # Power parameter for the Minkowski metric
    #     'metric': ['minkowski'],  # Distance metric to use
    #     'metric_params': [{'w': np.ones(X_train.shape[1])}]  # Weight vector for wminkowski metric
    # }),

    # # K-Nearest Neighbors (seuclidean): Uses standardized Euclidean distance
    # ('K-Nearest Neighbors (seuclidean)', KNeighborsClassifier(), {
    #     'n_neighbors': range(1, 3),  # Number of neighbors to use
    #     'weights': ['uniform', 'distance'],  # Weight function used in prediction
    #     'algorithm': ['auto', 'ball_tree', 'brute'],  # Algorithm used to compute the nearest neighbors
    #     'leaf_size': [10, 20, 30],  # Leaf size passed to BallTree or KDTree
    #     'p': [1, 2],  # Power parameter for the Minkowski metric
    #     'metric': ['seuclidean'],  # Distance metric to use
    #     'metric_params': [{'V': np.var(X_train, axis=0)}]  # Variance vector for seuclidean metric
    # }),

    # # K-Nearest Neighbors (mahalanobis): Uses Mahalanobis distance
    # ('K-Nearest Neighbors (mahalanobis)', KNeighborsClassifier(), {
    #     'n_neighbors': range(1, 3),  # Number of neighbors to use
    #     'weights': ['uniform', 'distance'],  # Weight function used in prediction
    #     'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm used to compute the nearest neighbors
    #     'leaf_size': [10, 20, 30],  # Leaf size passed to BallTree or KDTree
    #     'p': [1, 2],  # Power parameter for the Minkowski metric
    #     'metric': ['mahalanobis'],  # Distance metric to use
    #     'metric_params': [{'VI': np.linalg.inv(np.cov(X_train, rowvar=False))}]  # Inverse covariance matrix for mahalanobis metric
    # }),

    # Decision Tree: Splits the data into subsets based on the feature that results in the most homogeneous subsets
    ('Decision Tree', DecisionTreeClassifier(random_state=42), {
        'max_depth': range(1, 3),  # Maximum depth of the tree
        'min_samples_split': [0.2, 0.3, 0.4],  # Minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2, 3],  # Minimum number of samples required to be at a leaf node
        'criterion': ['gini', 'entropy'],  # Function to measure the quality of a split
        'splitter': ['best', 'random'],  # Strategy used to choose the split at each node
        'max_features': [None, 'sqrt', 'log2', 10, 100, 0.4, 0.8]  # Number of features to consider when looking for the best split
    }),

    # # AdaBoost: Combines multiple weak classifiers to create a strong classifier
    # ('AdaBoost', AdaBoostClassifier(random_state=42), {
    #     'n_estimators': [200, 250, 300],  # Number of weak learners
    #     'learning_rate': [0.01, 0.1, 1],  # Weight applied to each classifier
    #     'estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2)],  # Base estimator from which the boosted ensemble is built
    #     'algorithm': ['SAMME', 'SAMME.R']  # Boosting algorithm to use
    # }),

    # # Stochastic Gradient Descent (SGD): Optimizes the loss function using stochastic gradient descent
    # ('Linear Regression (SGD with Adam)', SGDClassifier(learning_rate='adaptive', eta0=0.01, random_state=42, max_iter=5000), {
    #     'alpha': [0.00001, 0.0001, 0.001],  # Regularization term
    #     'loss': ['log_loss', 'hinge', 'modified_huber', 'perceptron', 'squared_hinge'],  # Loss function
    #     'penalty': ['l2', 'l1', 'elasticnet'],  # Penalty (regularization term)
    #     'epsilon': [0.1, 0.01, 0.001],  # Epsilon in the epsilon-insensitive loss functions
    #     'l1_ratio': [0.15, 0.5, 0.85],  # The Elastic Net mixing parameter
    #     'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive']  # Learning rate schedule
    # }),

    # # Random Forest: Constructs multiple decision trees and outputs the mode of their predictions
    # ('Random Forest (bootstrap)', RandomForestClassifier(random_state=42), {
    #     'n_estimators': [200, 250, 300],  # Number of trees in the forest
    #     'max_depth': range(1, 3),  # Maximum depth of the tree
    #     'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    #     'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    #     'criterion': ['gini', 'entropy'],  # Function to measure the quality of a split
    #     'bootstrap': [True],  # Whether bootstrap samples are used when building trees
    #     'oob_score': [True, False],  # Whether to use out-of-bag samples to estimate the generalization accuracy
    #     'max_features': [None, 'sqrt', 'log2', 1, 2, 3, .1, .2, .3]  # Number of features to consider when looking for the best split
    # }),

    # # Random Forest (no bootstrap): Constructs multiple decision trees without bootstrap sampling
    # ('Random Forest (no bootstrap)', RandomForestClassifier(random_state=42), {
    #     'n_estimators': [200, 250, 300],  # Number of trees in the forest
    #     'max_depth': range(1, 3),  # Maximum depth of the tree
    #     'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    #     'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    #     'criterion': ['gini', 'entropy'],  # Function to measure the quality of a split
    #     'bootstrap': [False],  # Whether bootstrap samples are used when building trees
    #     'max_features': [None, 'auto', 'sqrt', 'log2']  # Number of features to consider when looking for the best split
    # }),

    # # Gradient Boosting: Builds an ensemble of trees in a stage-wise fashion to minimize the loss function
    # ('Gradient Boosting', GradientBoostingClassifier(random_state=42), {
    #     'n_estimators': [200, 250, 300],  # Number of boosting stages to be run
    #     'learning_rate': [0.01, 0.1, 0.2],  # Learning rate shrinks the contribution of each tree
    #     'max_depth': range(1, 3),  # Maximum depth of the individual regression estimators
    #     'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    #     'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    #     'criterion': ['friedman_mse', 'mse', 'mae'],  # Function to measure the quality of a split
    #     'subsample': [0.5, 0.7, 1.0],  # Fraction of samples used for fitting the individual base learners
    #     'max_features': [None, 'auto', 'sqrt', 'log2'],  # Number of features to consider when looking for the best split
    #     'loss': ['deviance', 'exponential']  # Loss function to be optimized
    # }),

    # # Naive Bayes: Applies Bayes' theorem with the assumption of independence between features
    # ('Naive Bayes', GaussianNB(), {
    #     'priors': [None, [0.5, 0.5], [0.3, 0.7]],  # Prior probabilities of the classes
    #     'var_smoothing': [1e-9, 1e-8, 1e-7]  # Portion of the largest variance of all features that is added to variances for calculation stability
    # }),
]

# Parallelize the training and evaluation
Parallel(n_jobs=-1)(delayed(train_and_evaluate)(model, params, model_name, results) for model_name, model, params in models_params)

breakpoint()

# Convert results back to a regular dictionary
results = dict(results)

# Function to extract and transform predictions for a single model
def extract_transform_predictions(model_name):
    return np.where(results[model_name]['predictions'] == 0, -1, results[model_name]['predictions'])

# List of model names
model_names = [name for name, _, _ in models_params]

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
accuracies = np.array([results[model_name]['Accuracy'] for model_name in model_names])

class WeightedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, predictions=None, accuracies=None, power=2):
        self.predictions = predictions
        self.accuracies = accuracies
        self.power = power

    def fit(self, X, y):
        return self # No fitting necessary as predictions are already provided

    def predict(self, X):   # Ensure predictions are for the correct number of samples
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

# Plot performance of the weighted ensemble method as power changes
powers, accuracies, f1_scores = zip(*results_parallel)

# Use the existing plot_performance function to plot accuracy and F1 score
plot_performance('Weighted Ensemble - Accuracy', {'Power': power_values}, {'Power': accuracies})
plot_performance('Weighted Ensemble - F1 Score', {'Power': power_values}, {'Power': f1_scores})

# Model Stacking: Combine predictions of multiple models using a second-level model
estimators = [(name, model.set_params(**{k: results[name][k] for k in params.keys()})) for name, model, params in models_params]

# Define hyperparameters for StackingClassifier
stacking_params = {
    'final_estimator': [LogisticRegression(), RandomForestClassifier(), GradientBoostingClassifier()],
    'cv': [3, 5, 10],
    'passthrough': [False, True]
}

stacking_model = StackingClassifier(estimators=estimators)
stacking_grid_search = GridSearchCV(stacking_model, stacking_params, cv=stratified_kfold, n_jobs=-1, return_train_score=True)
stacking_grid_search.fit(X_train, y_train)
y_test_pred = stacking_grid_search.predict(X_test)

results['Stacking'] = {
    'Accuracy': accuracy_score(y_test, y_test_pred),
    'F1 Score': f1_score(y_test, y_test_pred),
    'Best Params': stacking_grid_search.best_params_
}

# Plot performance of the stacking model
stacking_cv_results = stacking_grid_search.cv_results_
plot_performance('Stacking Model - Accuracy', {'cv': stacking_params['cv']}, {'cv': stacking_cv_results['mean_test_score']})
plot_performance('Stacking Model - F1 Score', {'cv': stacking_params['cv']}, {'cv': stacking_cv_results['mean_test_score']})

# Write results to a text file
with open('./Final/models_results.txt', 'w') as f:
    for model, metrics in results.items():
        f.write(f"{model} Test Set Evaluation:\n")
        f.write(f"Accuracy: {metrics['Accuracy']}\n")
        f.write(f"F1 Score: {metrics['F1 Score']}\n")
        for param, value in metrics.get('Best Params', {}).items():
            f.write(f"Optimal {param}: {value}\n")
        f.write("\n")
