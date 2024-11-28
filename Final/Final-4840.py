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
stratified_kfold = StratifiedKFold(n_splits=4)

# Function to train and evaluate a model
def train_and_evaluate(model, params, model_name, results):
    try:
        grid_search = GridSearchCV(model, params, cv=stratified_kfold, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        optimal_params = grid_search.best_params_
        y_test_pred = grid_search.predict(X_test)
        results[model_name] = {
            'Accuracy': accuracy_score(y_test, y_test_pred),
            'F1 Score': f1_score(y_test, y_test_pred),
            **optimal_params
        }
        print(f"{model_name} results: {results[model_name]}")
    except Exception as e:
        print(f"Error training {model_name}: {e}")

# Define models and their hyperparameters
models_params = [
    (SVC(kernel='linear', random_state=42), {'C': [0.1, 1, 10, 100, 150, 200]}, 'SVM'),
    (LogisticRegression(random_state=42), {'C': [0.1, 1, 10, 100, 150, 200]}, 'Logistic Regression'),
    (KNeighborsClassifier(), {'n_neighbors': range(1, 21)}, 'K-Nearest Neighbors'),
    (DecisionTreeClassifier(random_state=42), {'max_depth': range(1, 21)}, 'Decision Tree'),
    (AdaBoostClassifier(algorithm='SAMME', random_state=42), {'n_estimators': [50, 100, 200, 250, 300]}, 'AdaBoost'),
    (SGDClassifier(loss='log_loss', learning_rate='adaptive', eta0=0.01, random_state=42), {'alpha': [0.000001, 0.00001, 0.0001, 0.001, 0.01]}, 'Linear Regression (SGD with Adam)'),
    (RandomForestClassifier(random_state=42), {'n_estimators': [50, 100, 200, 250, 300], 'max_depth': range(1, 21)}, 'Random Forest'),
    (GradientBoostingClassifier(random_state=42), {'n_estimators': [50, 100, 200, 250, 300], 'max_depth': range(1, 21)}, 'Gradient Boosting'),
    (GaussianNB(), {}, 'Naive Bayes')
]

# Parallelize the training and evaluation
Parallel(n_jobs=-1)(delayed(train_and_evaluate)(model, params, model_name, results) for model, params, model_name in models_params)

# Convert results back to a regular dictionary
results = dict(results)

# Check if 'SVM' key exists in results
if 'SVM' not in results:
    print("SVM results not found in the results dictionary.")
else:
    # Continue with the rest of the code
    svm_pred = results['SVM']['Accuracy']
    log_reg_pred = results['Logistic Regression']['Accuracy']
    knn_pred = results['K-Nearest Neighbors']['Accuracy']
    dt_pred = results['Decision Tree']['Accuracy']
    ada_pred = results['AdaBoost']['Accuracy']
    sgd_pred = results['Linear Regression (SGD with Adam)']['Accuracy']
    rf_pred = results['Random Forest']['Accuracy']
    gb_pred = results['Gradient Boosting']['Accuracy']
    nb_pred = results['Naive Bayes']['Accuracy']

    # Convert predictions: 0 to -1, 1 remains 1
    svm_pred = np.where(svm_pred == 0, -1, 1)
    log_reg_pred = np.where(log_reg_pred == 0, -1, 1)
    knn_pred = np.where(knn_pred == 0, -1, 1)
    dt_pred = np.where(dt_pred == 0, -1, 1)
    ada_pred = np.where(ada_pred == 0, -1, 1)
    sgd_pred = np.where(sgd_pred == 0, -1, 1)
    rf_pred = np.where(rf_pred == 0, -1, 1)
    gb_pred = np.where(gb_pred == 0, -1, 1)
    nb_pred = np.where(nb_pred == 0, -1, 1)

    # Initialize array to store votes
    votes = np.zeros_like(y_test, dtype=int)

    # Add votes for each model
    votes += svm_pred
    votes += log_reg_pred
    votes += knn_pred
    votes += dt_pred
    votes += ada_pred
    votes += sgd_pred
    votes += rf_pred
    votes += gb_pred
    votes += nb_pred

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

    class WeightedEnsembleClassifier(BaseEstimator, ClassifierMixin):
        def __init__(self, models, power=2):
            self.models = models
            self.power = power

        def fit(self, X, y):
            for name, model in self.models:
                model.fit(X, y)
            return self

        def predict(self, X):
            predictions = np.zeros((X.shape[0], len(self.models)))
            accuracies = np.zeros(len(self.models))

            def get_predictions_and_accuracy(i, name, model):
                pred = model.predict(X)
                pred = np.where(pred == 0, -1, 1)
                accuracy = accuracy_score(y_train, model.predict(X_train))
                return pred, accuracy

            results = Parallel(n_jobs=-1)(delayed(get_predictions_and_accuracy)(i, name, model) for i, (name, model) in enumerate(self.models))

            for i, (pred, accuracy) in enumerate(results):
                predictions[:, i] = pred
                accuracies[i] = accuracy

            weights = accuracies ** self.power
            weights /= weights.sum()

            weighted_votes = np.dot(predictions, weights)
            return (weighted_votes >= 0).astype(int)

    # Define the models
    models = [
        ('svm', SVC(kernel='linear', C=results['SVM']['C'], random_state=42)),
        ('log_reg', LogisticRegression(C=results['Logistic Regression']['C'], random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=results['K-Nearest Neighbors']['n_neighbors'])),
        ('dt', DecisionTreeClassifier(max_depth=results['Decision Tree']['max_depth'], random_state=42)),
        ('ada', AdaBoostClassifier(n_estimators=results['AdaBoost']['n_estimators'], algorithm='SAMME', random_state=42)),
        ('sgd', SGDClassifier(alpha=results['Linear Regression (SGD with Adam)']['alpha'], loss='log_loss', learning_rate='adaptive', eta0=0.01, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=results['Random Forest']['n_estimators'], max_depth=results['Random Forest']['max_depth'], random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=results['Gradient Boosting']['n_estimators'], max_depth=results['Gradient Boosting']['max_depth'], random_state=42)),
        ('nb', GaussianNB())
    ]

    # Define the parameter grid
    param_grid = {'power': [1, 2, 3, 4, 5, 6]}

    # Initialize the weighted ensemble classifier
    weighted_ensemble = WeightedEnsembleClassifier(models=models)

    # Use GridSearchCV to find the optimal power
    grid_search = GridSearchCV(weighted_ensemble, param_grid, cv=stratified_kfold, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best power
    best_power = grid_search.best_params_['power']

    # Predict on the test set using the best power
    weighted_ensemble = WeightedEnsembleClassifier(models=models, power=best_power)
    weighted_ensemble.fit(X_train, y_train)
    ensemble_pred = weighted_ensemble.predict(X_test)

    # Calculate accuracy and F1 score for the ensemble method
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    ensemble_f1 = f1_score(y_test, ensemble_pred)

    # Add ensemble results to the results dictionary
    results['Weighted Ensemble'] = {
        'Accuracy': ensemble_accuracy,
        'F1 Score': ensemble_f1,
        'Optimal Power': best_power
    }

    # Model Stacking: Combine predictions of multiple models using a second-level model
    estimators = [
        ('svm', SVC(kernel='linear', C=results['SVM']['C'], random_state=42)),
        ('log_reg', LogisticRegression(C=results['Logistic Regression']['C'], random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=results['K-Nearest Neighbors']['n_neighbors'])),
        ('dt', DecisionTreeClassifier(max_depth=results['Decision Tree']['max_depth'], random_state=42)),
        ('ada', AdaBoostClassifier(n_estimators=results['AdaBoost']['n_estimators'], algorithm='SAMME', random_state=42)),
        ('sgd', SGDClassifier(alpha=results['Linear Regression (SGD with Adam)']['alpha'], loss='log_loss', learning_rate='adaptive', eta0=0.01, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=results['Random Forest']['n_estimators'], max_depth=results['Random Forest']['max_depth'], random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=results['Gradient Boosting']['n_estimators'], max_depth=results['Gradient Boosting']['max_depth'], random_state=42)),
        ('nb', GaussianNB())
    ]
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
