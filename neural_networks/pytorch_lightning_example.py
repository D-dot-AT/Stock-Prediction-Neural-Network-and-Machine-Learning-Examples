import pandas as pd
import pytorch_lightning as L
import torch
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import print_statistics

# Step 1: Data Preparation
# Loading the training data
train_data = pd.read_csv('../example_data/train.csv', header=None)
X = train_data.iloc[:, :-1].values
Y = train_data.iloc[:, -1].values

# Scaling the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Model Creation
# Determining the number of input features
input_features = X_scaled.shape[1]


# Creating the neural network model
class SimpleNN(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(input_features, 64)
        self.layer2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.type(self.layer1.weight.dtype)
        x = nn.ReLU()(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.BCELoss()(y_hat, y.view(-1, 1).float())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


model = SimpleNN()

# Step 3: Training the Model
# Preparing data loaders
train_dataset = TensorDataset(torch.tensor(X_scaled), torch.tensor(Y))
train_loader = DataLoader(train_dataset, batch_size=32)

# Initializing a trainer and training the model
trainer = L.Trainer(max_epochs=5)
trainer.fit(model, train_loader)

# Step 4: Testing the Model
# Loading the test data
test_data = pd.read_csv('../example_data/test.csv', header=None)
X_test = test_data.iloc[:, :-1].values
Y_test = test_data.iloc[:, -1].values

# Scaling the test data
X_test_scaled = scaler.transform(X_test)

# Making predictions
model.eval()
with torch.no_grad():
    predictions = model(torch.tensor(X_test_scaled)).numpy()

# Binarizing predictions
predictions_bin = (predictions > 0.6).astype(int)

# Calculating metrics
precision = precision_score(Y_test, predictions_bin)
accuracy = accuracy_score(Y_test, predictions_bin)
TN, FP, FN, TP = confusion_matrix(Y_test, predictions_bin).ravel()

print_statistics(tp=TP, fp=FP, tn=TN, fn=FN)

# Load latest data
latest_data = pd.read_csv('../example_data/latest.csv')
tickers = latest_data.iloc[:, 0].values
X_latest = scaler.transform(latest_data.iloc[:, 1:].values)

# Predict scores using the model and print the top 5 stock tickers along with their percentage scores
scores = model(torch.Tensor(X_latest)).detach().numpy()
top_5_indices = scores.flatten().argsort()[-5:][::-1]
for idx in top_5_indices:
    print(f'{tickers[idx]}: {scores[idx][0] * 100:.2f}%')
