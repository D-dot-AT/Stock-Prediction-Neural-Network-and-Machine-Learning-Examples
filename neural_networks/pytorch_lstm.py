import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from common import print_statistics

# Step 1: Data Preparation
data_train = pd.read_csv('../example_data/train.csv', header=None)
X_train = data_train.iloc[:, :-1].values
Y_train = data_train.iloc[:, -1].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Step 2: Model Creation and Training
input_features = X_train.shape[1]


class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_features, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 1, input_features)
        h_0, _ = self.lstm(x)
        h_0 = h_0[:, -1, :]
        x = self.sigmoid(self.fc(h_0))
        return x


model = LSTMModel()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                         torch.tensor(Y_train, dtype=torch.float32))
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(10):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()

# Step 3: Model Testing
data_test = pd.read_csv('../example_data/test.csv', header=None)
X_test = data_test.iloc[:, :-1].values
Y_test = data_test.iloc[:, -1].values
X_test = scaler.transform(X_test)

predictions = model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()
predictions = (predictions > 0.5).astype(int).squeeze()

TN, FP, FN, TP = confusion_matrix(Y_test, predictions).ravel()
print_statistics(tp=TP, fp=FP, tn=TN, fn=FN)

# Step 4: Creating Predictions
data_latest = pd.read_csv('../example_data/latest.csv')
X_latest = data_latest.iloc[:, 1:].values
X_latest = scaler.transform(X_latest)

scores = model(torch.tensor(X_latest, dtype=torch.float32)).detach().numpy()
top5_indices = scores.squeeze().argsort()[-5:][::-1]

for i in top5_indices:
    print(f"{data_latest.iloc[i, 0]}: {scores[i][0] * 100:.2f}%")
