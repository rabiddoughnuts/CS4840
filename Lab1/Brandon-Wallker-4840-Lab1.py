import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = ""

def load_diabetes_data(data_path=DATA_PATH):
    csv_path = os.path.join(data_path, "diabetes.csv")
    return pd.read_csv(csv_path)

diabetes = load_diabetes_data()

def answer_one():
    print(diabetes.all())
    print(diabetes.describe(include='all'))

answer_one()

def answer_two():
    corr_matrix = diabetes.corr(method='pearson', numeric_only=True)
    print(corr_matrix)

    diabetes.plot(kind="scatter", x="bmi", y="diabetes_progression_one_year", alpha=0.3)
    plt.axis([0, 45, 0, 360])

    diabetes.plot(kind="scatter", x="low_density_lipoproteins", y="diabetes_progression_one_year", alpha=0.3)
    plt.axis([0, 250, 0, 360])

    plt.show()

answer_two()

diabetes_labels = diabetes["diabetes_progression_one_year"].copy()
diabetes_features = diabetes.drop("diabetes_progression_one_year", axis=1)

def answer_three():
    median_cholesterol = diabetes_features["total_cholesterol"].median()
    diabetes_features["total_cholesterol"] = diabetes_features["total_cholesterol"].fillna(median_cholesterol)
    print(diabetes_features[diabetes_features.isnull().any(axis=1)].head())

answer_three()

diabetes_num = diabetes_features.drop("gender", axis=1)

diabetes_cat = diabetes_features[["gender"]]
print(diabetes_cat["gender"].value_counts())

from sklearn.preprocessing import OrdinalEncoder

def answer_four():
    ordinal_encoder = OrdinalEncoder()
    diabetes_cat_encoded = ordinal_encoder.fit_transform(diabetes_cat)

    return diabetes_cat_encoded

diabetes_cat_encoded = answer_four()
print(diabetes_cat_encoded)

from sklearn.preprocessing import StandardScaler

def answer_five():
    std_scaler = StandardScaler()
    diabetes_num_scaled = std_scaler.fit_transform(diabetes_num)

    return diabetes_num_scaled

diabetes_num_scaled = answer_five()
print(diabetes_num_scaled)

X = np.concatenate((diabetes_num_scaled, diabetes_cat_encoded), axis=1)

Y = diabetes_labels.to_numpy()

print(X.shape)
print(Y.shape)

from sklearn.model_selection import train_test_split

def answer_six():
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test = answer_six()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error

def answer_seven():
    lin_reg = LinearRegression()

    lin_reg.fit(X_train, Y_train)

    Y_predict = lin_reg.predict(X_test)

    rmse = root_mean_squared_error(Y_test, Y_predict)

    mae = mean_absolute_error(Y_test, Y_predict)

    return rmse, mae

rmse_sklearn, mae_sklearn = answer_seven()
print(rmse_sklearn, mae_sklearn)

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(42)
torch.manual_seed(42)

#Data convertion class
class ConvertDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        feature = self.X[idx]
        label = self.y[idx]
        sample = {'feature': feature, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

#ToTensor function
class ToTensor(object):
    def __call__(self, sample):
        feature, label = sample['feature'], sample['label'] 
        label = np.array(label)
        return {'feature': torch.from_numpy(feature).float(),
                'label': torch.from_numpy(label).float()}

#Training function
def train(epoch, model, train_dataloader, optimizer):
    model.train()
    
    train_loss = 0.0
    
    for i, data in enumerate(train_dataloader):
        X, y = data['feature'], data['label'] 

        optimizer.zero_grad()

        predictions = model(X).squeeze()

        loss = lossfunction(predictions, y)
        loss.backward()
        optimizer.step()
        
        #print statistics
        train_loss += loss.item()

    print("epoch (%d): Train loss: %.3f" % (epoch, train_loss/10000))

b_size = 32

def answer_eight(b_size, X_train, y_train):
    #Please convert X_train and y_train
    train_dataset = ConvertDataset(X_train, Y_train, transform=ToTensor())

    #Load the converted training data into DataLoader: pass b_size you choose to the parameter batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=b_size, shuffle=True)

    return train_dataloader

#Run your function in the cell to return the result
train_dataloader = answer_eight(b_size, X_train, Y_train)

epochs = 1234
learning_rate = 0.01
lossfunction = nn.MSELoss()

def answer_nine(train_dataloader):
    class LinearRegression(nn.Module):
        def __init__(self):
            super(LinearRegression, self).__init__()
            # The first parameter should be the feature dimension: refer to the answer to Question 1 
            # The second parameter should be the number of regression output, which is 1
            self.fc = nn.Linear(X_train.shape[1], 1)

        def forward(self, x):
            # Define the calculation from x to y
            y = self.fc(x)
            return y

    # Instantiate an object from the class as the model
    model = LinearRegression()
    # Define optimizer using Adam
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(1, epochs + 1):
        # Run train() function
        train(epoch, model, train_dataloader, optimizer)

    # Make prediction on the X_test using the trained model
    y_predict = model(torch.FloatTensor(X_test)).detach().numpy()

    # Please calculate root mean square error using root_mean_squared_error API
    rmse = np.sqrt(np.mean((Y_test - y_predict) ** 2))

    # Please calculate mean absolute error using mean_absolute_error API
    mae = np.mean(np.abs(Y_test - y_predict))

    return rmse, mae

# Run your function in the cell to return the result
rmse_pytorch, mae_pytorch = answer_nine(train_dataloader)
print(rmse_pytorch, mae_pytorch)