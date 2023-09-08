import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix
from scipy.stats import fisher_exact
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
class SimpleNN(pl.LightningModule):
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
trainer = pl.Trainer(max_epochs=5)
trainer.fit(model, train_loader)

# Saving the trained model and scaler
torch.save(model.state_dict(), 'model.pt')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

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
predictions_bin = (predictions > 0.5).astype(int)

# Calculating metrics
precision = precision_score(Y_test, predictions_bin)
accuracy = accuracy_score(Y_test, predictions_bin)
TN, FP, FN, TP = confusion_matrix(Y_test, predictions_bin).ravel()

# Step 5: Statistical Analysis
# Performing Fisher's exact test
contingency_table = [[TP, FP], [FN, TN]]
oddsratio, pvalue = fisher_exact(contingency_table)

# Step 6: Output
# Printing the results
print(f'Positive rate in test set: {(TP + FN) / (TP + FP + TN + FN)}')
print(f'Model precision: {precision}')
print(f'Accuracy: {accuracy}')
print(f'P-value of precision: {pvalue}')
