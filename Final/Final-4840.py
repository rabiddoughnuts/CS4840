import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
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
X = df_drebin.drop('class', axis=1)
y = df_drebin['class']

# Split into training (60%), validation (20%), and test (20%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Dictionary to store results
results = {}

# Train and evaluate SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
y_test_pred = svm_model.predict(X_test)
results['SVM'] = {
    'Accuracy': accuracy_score(y_test, y_test_pred),
    'F1 Score': f1_score(y_test, y_test_pred)
}

# Train and evaluate Logistic Regression model
log_reg_model = LogisticRegression(random_state=42)
log_reg_model.fit(X_train, y_train)
y_test_pred = log_reg_model.predict(X_test)
results['Logistic Regression'] = {
    'Accuracy': accuracy_score(y_test, y_test_pred),
    'F1 Score': f1_score(y_test, y_test_pred)
}

# Train and evaluate K-Nearest Neighbors model with GridSearchCV to find optimal k
knn_params = {'n_neighbors': range(1, 21)}
knn_model = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5)
knn_model.fit(X_train, y_train)
optimal_k = knn_model.best_params_['n_neighbors']
y_test_pred = knn_model.predict(X_test)
results['K-Nearest Neighbors'] = {
    'Accuracy': accuracy_score(y_test, y_test_pred),
    'F1 Score': f1_score(y_test, y_test_pred),
    'Optimal k': optimal_k
}

# Train and evaluate Decision Tree model with GridSearchCV to find optimal depth
dt_params = {'max_depth': range(1, 21)}
dt_model = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=5)
dt_model.fit(X_train, y_train)
optimal_depth = dt_model.best_params_['max_depth']
y_test_pred = dt_model.predict(X_test)
results['Decision Tree'] = {
    'Accuracy': accuracy_score(y_test, y_test_pred),
    'F1 Score': f1_score(y_test, y_test_pred),
    'Optimal Depth': optimal_depth
}

# Train and evaluate AdaBoost model
ada_model = AdaBoostClassifier(algorithm='SAMME', random_state=42)
ada_model.fit(X_train, y_train)
y_test_pred = ada_model.predict(X_test)
results['AdaBoost'] = {
    'Accuracy': accuracy_score(y_test, y_test_pred),
    'F1 Score': f1_score(y_test, y_test_pred)
}

# Train and evaluate Linear Regression model using SGD with Adam optimizer
sgd_model = SGDClassifier(loss='log_loss', learning_rate='adaptive', eta0=0.01, random_state=42)
sgd_model.fit(X_train, y_train)
y_test_pred = sgd_model.predict(X_test)
results['Linear Regression (SGD with Adam)'] = {
    'Accuracy': accuracy_score(y_test, y_test_pred),
    'F1 Score': f1_score(y_test, y_test_pred)
}

# Print results
for model, metrics in results.items():
    print(f"{model} Test Set Evaluation:")
    print(f"Accuracy: {metrics['Accuracy']}")
    print(f"F1 Score: {metrics['F1 Score']}")
    if 'Optimal k' in metrics:
        print(f"Optimal k: {metrics['Optimal k']}")
    if 'Optimal Depth' in metrics:
        print(f"Optimal Depth: {metrics['Optimal Depth']}")
    print()