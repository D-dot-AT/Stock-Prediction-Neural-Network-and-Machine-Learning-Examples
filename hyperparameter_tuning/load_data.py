import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset


def load_data():
    train_data = pd.read_csv('../example_data/train.csv', header=None)
    X = train_data.iloc[:, :-1].values
    Y = train_data.iloc[:, -1].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    train_dataset = TensorDataset(torch.tensor(X_scaled), torch.tensor(Y))

    test_data = pd.read_csv('../example_data/test.csv', header=None)
    X_test = test_data.iloc[:, :-1].values
    Y_test = test_data.iloc[:, -1].values
    X_test_scaled = scaler.transform(X_test)

    input_feature_size = X_scaled.shape[1]

    return train_dataset, X_test_scaled, Y_test, input_feature_size