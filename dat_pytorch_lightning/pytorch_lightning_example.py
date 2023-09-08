import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix
from scipy.stats import fisher_exact
import pandas as pd

# Step 1: Data Preparation
# Load training data
train_data = pd.read_csv('../example_data/train.csv', header=None)

# Separate features and labels
X_train = train_data.iloc[:, :-1].values
Y_train = train_data.iloc[:, -1].values

# Scale feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)


# Step 2: Model Creation
class SimpleNN(pl.LightningModule):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y.view(-1, 1).float())
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


input_size = X_train.shape[1]
model = SimpleNN(input_size)
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.int64))
train_loader = DataLoader(train_dataset, batch_size=32)


# Step 3: Training the Model
trainer = pl.Trainer(max_epochs=5)
trainer.fit(model, train_loader)

# Save model and scaler
torch.save(model.state_dict(), 'model.pth')
torch.save(scaler, 'scaler.pth')

# Step 4: Testing the Model
# Load test data
test_data = pd.read_csv('../example_data/test.csv', header=None)
X_test = test_data.iloc[:, :-1].values
Y_test = test_data.iloc[:, -1].values

# Scale feature data
X_test = scaler.transform(X_test)

# Get predictions
model.eval()
with torch.no_grad():
    predictions = model(torch.tensor(X_test, dtype=torch.float32))
    predictions = (predictions.view(-1) > 0.5).int()

# Calculate metrics
precision = precision_score(Y_test, predictions)
accuracy = accuracy_score(Y_test, predictions)
tn, fp, fn, tp = confusion_matrix(Y_test, predictions).ravel()


# Step 5: Statistical Analysis
# Perform Fisher's exact test
contingency_table = [[tp, fp], [fn, tn]]
_, p_value = fisher_exact(contingency_table)

# Step 6: Output
print("Precision:", precision)
print("Accuracy:", accuracy)
print("True Positives:", tp)
print("False Positives:", fp)
print("True Negatives:", tn)
print("False Negatives:", fn)
print("P-value from Fisher's exact test:", p_value)
