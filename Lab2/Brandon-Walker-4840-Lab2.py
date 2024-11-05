import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Please place housing.csv and your notebook/python file in the same directory; otherwise, change DATA_PATH 
DATA_PATH = ""

def load_housing_data(housing_path=DATA_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()

#Add three useful features
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

#Divide the data frame into features and labels
housing_labels = housing["ocean_proximity"].copy() # use ocean_proximity as classification label
housing_features = housing.drop("ocean_proximity", axis=1) # use colums other than ocean_proximity as features

#Preprocessing the missing feature values
median = housing_features["total_bedrooms"].median()
housing_features["total_bedrooms"].fillna(median, inplace=True) 
median = housing_features["bedrooms_per_room"].median()
housing_features["bedrooms_per_room"].fillna(median, inplace=True)

#Scale the features
std_scaler  = StandardScaler()
housing_features_scaled = std_scaler.fit_transform(housing_features)

#Final housing features X
X = housing_features_scaled

#Binary labels - 0: INLAND; 1: CLOSE TO OCEAN
y_binary = (housing_labels != 1).astype(int)
#Multi-class labels - 0: <1H OCEAN; 1: INLAND; 2: NEAR OCEAN; 3: NEAR BAY
y_multi = housing_labels.astype(int)

#Data splits for binary classification
X_train_bi, X_test_bi, y_train_bi, y_test_bi = train_test_split(X, y_binary, test_size=0.20, random_state=42)

#Data splits for multi-class classification
X_train_mu, X_test_mu, y_train_mu, y_test_mu = train_test_split(X, y_multi, test_size=0.20, random_state=42)

print(X_train_bi.shape)
print(X_test_bi.shape)