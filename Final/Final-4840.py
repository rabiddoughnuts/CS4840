import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier

# Load the drebin dataset
df_drebin = pd.read_csv('/run/media/Brandon/D820DA8720DA6BCE/Classes/CS4840/Final/drebin.csv')  # Replace with the actual path to your drebin.csv file

# Convert class label -1 to 0
df_drebin['class'] = df_drebin['class'].replace(-1, 0)

# Function to print correlation with 'class' column
def print_feature_correlations(df):
    correlations = df.corr()['class'].drop('class')
    correlation_table = pd.DataFrame(correlations).reset_index()
    correlation_table.columns = ['Feature', 'Correlation with Malware']
    correlation_table['Absolute Correlation'] = correlation_table['Correlation with Malware'].abs()
    correlation_table = correlation_table.sort_values(by='Absolute Correlation', ascending=False).drop('Absolute Correlation', axis=1)
    for index, row in correlation_table.iterrows():
        print(f"Feature: {row['Feature']}, Correlation with Malware: {row['Correlation with Malware']}")

# Print feature correlations
# print_feature_correlations(df_drebin)

# Separate features and labels
X = df_drebin.drop('class', axis=1)  # Features (input variables)
y = df_drebin['class']  # Labels (output variable)

# Split the data into training (80%) and test (20%) sets
# Stratify ensures that the class distribution is similar in both training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Dictionary to store results of each model
results = {}

# Define StratifiedKFold cross-validation
# This ensures that each fold has the same proportion of class labels as the original dataset
stratified_kfold = StratifiedKFold(n_splits=6)

# Train and evaluate SVM model with GridSearchCV to find optimal C
# GridSearchCV helps in finding the best hyperparameters by trying all combinations in the provided grid
svm_params = {'C': [0.1, 1, 10, 100]}  # Hyperparameter grid for SVM. 'C' controls the trade-off between smooth decision boundary and classifying training points correctly.
svm_model = GridSearchCV(SVC(kernel='linear', random_state=42), svm_params, cv=stratified_kfold)
svm_model.fit(X_train, y_train)  # Fit the model on training data
optimal_C = svm_model.best_params_['C']  # Best hyperparameter found
y_test_pred = svm_model.predict(X_test)  # Predict on test set
results['SVM'] = {
    'Accuracy': accuracy_score(y_test, y_test_pred),
    'F1 Score': f1_score(y_test, y_test_pred),
    'Optimal C': optimal_C
}

# Train and evaluate Logistic Regression model with GridSearchCV to find optimal C
log_reg_params = {'C': [0.1, 1, 10, 100]}  # Hyperparameter grid for Logistic Regression. 'C' controls the regularization strength.
log_reg_model = GridSearchCV(LogisticRegression(random_state=42), log_reg_params, cv=stratified_kfold)
log_reg_model.fit(X_train, y_train)
optimal_C = log_reg_model.best_params_['C']
y_test_pred = log_reg_model.predict(X_test)
results['Logistic Regression'] = {
    'Accuracy': accuracy_score(y_test, y_test_pred),
    'F1 Score': f1_score(y_test, y_test_pred),
    'Optimal C': optimal_C
}

# Train and evaluate K-Nearest Neighbors model with GridSearchCV to find optimal k
knn_params = {'n_neighbors': range(1, 21)}  # Hyperparameter grid for KNN. 'n_neighbors' is the number of neighbors to use.
knn_model = GridSearchCV(KNeighborsClassifier(), knn_params, cv=stratified_kfold)
knn_model.fit(X_train, y_train)
optimal_k = knn_model.best_params_['n_neighbors']
y_test_pred = knn_model.predict(X_test)
results['K-Nearest Neighbors'] = {
    'Accuracy': accuracy_score(y_test, y_test_pred),
    'F1 Score': f1_score(y_test, y_test_pred),
    'Optimal k': optimal_k 
}

# Train and evaluate Decision Tree model with GridSearchCV to find optimal depth
dt_params = {'max_depth': range(1, 21)}  # Hyperparameter grid for Decision Tree. 'max_depth' is the maximum depth of the tree.
dt_model = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=stratified_kfold)
dt_model.fit(X_train, y_train)
optimal_depth = dt_model.best_params_['max_depth']
y_test_pred = dt_model.predict(X_test)
results['Decision Tree'] = {
    'Accuracy': accuracy_score(y_test, y_test_pred),
    'F1 Score': f1_score(y_test, y_test_pred),
    'Optimal Depth': optimal_depth
}

# Train and evaluate AdaBoost model with GridSearchCV to find optimal n_estimators
ada_params = {'n_estimators': [50, 100, 200]}  # Hyperparameter grid for AdaBoost. 'n_estimators' is the number of boosting stages to perform.
ada_model = GridSearchCV(AdaBoostClassifier(algorithm='SAMME', random_state=42), ada_params, cv=stratified_kfold)
ada_model.fit(X_train, y_train)
optimal_n_estimators = ada_model.best_params_['n_estimators']
y_test_pred = ada_model.predict(X_test)
results['AdaBoost'] = {
    'Accuracy': accuracy_score(y_test, y_test_pred),
    'F1 Score': f1_score(y_test, y_test_pred),
    'Optimal n_estimators': optimal_n_estimators
}

# Train and evaluate Linear Regression model using SGD with GridSearchCV to find optimal alpha
sgd_params = {'alpha': [0.0001, 0.001, 0.01, 0.1]}  # Hyperparameter grid for SGD. 'alpha' is the regularization term.
sgd_model = GridSearchCV(SGDClassifier(loss='log_loss', learning_rate='adaptive', eta0=0.01, random_state=42), sgd_params, cv=stratified_kfold)
sgd_model.fit(X_train, y_train)
optimal_alpha = sgd_model.best_params_['alpha']
y_test_pred = sgd_model.predict(X_test)
results['Linear Regression (SGD with Adam)'] = {
    'Accuracy': accuracy_score(y_test, y_test_pred),
    'F1 Score': f1_score(y_test, y_test_pred),
    'Optimal alpha': optimal_alpha
}

# Write results to a text file
with open('model_results.txt', 'w') as f:
    for model, metrics in results.items():
        f.write(f"{model} Test Set Evaluation:\n")
        f.write(f"Accuracy: {metrics['Accuracy']}\n")
        f.write(f"F1 Score: {metrics['F1 Score']}\n")
        if 'Optimal C' in metrics:
            f.write(f"Optimal C: {metrics['Optimal C']}\n")
        if 'Optimal k' in metrics:
            f.write(f"Optimal k: {metrics['Optimal k']}\n")
        if 'Optimal Depth' in metrics:
            f.write(f"Optimal Depth: {metrics['Optimal Depth']}\n")
        if 'Optimal n_estimators' in metrics:
            f.write(f"Optimal n_estimators: {metrics['Optimal n_estimators']}\n")
        if 'Optimal alpha' in metrics:
            f.write(f"Optimal alpha: {metrics['Optimal alpha']}\n")
        f.write("\n")