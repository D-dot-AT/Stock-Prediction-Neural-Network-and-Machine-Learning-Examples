import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score
import pandas as pd
import pickle


# Create a custom Dataset class
class CSVDataset(Dataset):
    def __init__(self, path, scaler=None):
        data = pd.read_csv(path, header=None)
        self.X = data.iloc[:, :-1].values
        self.y = data.iloc[:, -1].values

        if scaler is None:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X)
        else:
            self.X = scaler.transform(self.X)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Define your neural net model
class Net(nn.Module):
    def __init__(self, n_features):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_features, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


# Create a PyTorch DataLoader
train_dataset = CSVDataset('../example_data/train.csv')
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

# Create a model
model = Net(train_dataset.X.shape[1])

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(10):  # let's just train for 10 epochs
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = torch.tensor(inputs, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.float).view(-1, 1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Save the model and the scaler
os.makedirs('output', exist_ok=True)
torch.save(model, 'output/model.pkl')
with open('output/scaler.pkl', 'wb') as f:
    pickle.dump(train_dataset.scaler, f)

# Load test data
test_dataset = CSVDataset('../example_data/test.csv', scaler=train_dataset.scaler)
test_loader = DataLoader(dataset=test_dataset, batch_size=32)

# Predict on test data
model.eval()
with torch.no_grad():
    predictions = []
    actuals = []
    for i, (inputs, labels) in enumerate(test_loader):
        inputs = torch.tensor(inputs, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.float).view(-1, 1)

        outputs = model(inputs)
        predictions.extend(outputs.round().numpy())
        actuals.extend(labels.numpy())

# Print precision and accuracy
print(f'Precision: {precision_score(actuals, predictions)}')
print(f'Accuracy: {accuracy_score(actuals, predictions)}')
