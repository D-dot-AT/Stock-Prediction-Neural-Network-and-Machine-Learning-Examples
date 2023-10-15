import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from common import print_statistics

# Step 1: Data Preparation
# 1. Load the training data
data_train = pd.read_csv('../../example_data/train.csv', header=None)
# 2. Separate data into X (features) and Y (labels)
X_train = data_train.iloc[:, :-1].values
Y_train = data_train.iloc[:, -1].values
# 3. Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Step 2: Model Creation and Training
# 1. Create the model
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
)
# 2. Train the model
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
for epoch in range(10):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()

# Step 3: Model Testing
# 1. Load and scale the test data
data_test = pd.read_csv('../../example_data/test.csv', header=None)
X_test = data_test.iloc[:, :-1].values
Y_test = data_test.iloc[:, -1].values
X_test = scaler.transform(X_test)
# 2. Get predictions
predictions = model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()
predictions = (predictions > 0.5).astype(int).squeeze()
# 3. Obtain the confusion matrix variables
TN, FP, FN, TP = confusion_matrix(Y_test, predictions).ravel()
# Execute print_statistics
print_statistics(tp=TP, fp=FP, tn=TN, fn=FN)

# Step 4: Creating Predictions
# 1. Load data
data_latest = pd.read_csv('../../example_data/latest.csv')
X_latest = data_latest.iloc[:, 1:].values
# 2. Predict scores
X_latest = scaler.transform(X_latest)
scores = model(torch.tensor(X_latest, dtype=torch.float32)).detach().numpy()
# Print the top 5 stock tickers along with their percentage scores
top5_indices = scores.squeeze().argsort()[-5:][::-1]
for i in top5_indices:
    print(f"{data_latest.iloc[i, 0]}: {scores[i][0] * 100:.2f}%")
