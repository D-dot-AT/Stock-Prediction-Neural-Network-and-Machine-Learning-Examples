import time
from statistics import median

import pandas as pd
import pytorch_lightning as L
import torch
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import calculate_precision_p_value

PREDICTION_THRESHOLD = 0.5


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

    input_features = X_scaled.shape[1]

    return train_dataset, X_test_scaled, Y_test, input_features


def neural_network(input_features, hidden_layer_size, learning_rate):
    class SimpleNN(L.LightningModule):
        def __init__(self):
            super().__init__()
            if hidden_layer_size:
                self.layer1 = nn.Linear(input_features, hidden_layer_size)
                self.layer2 = nn.Linear(hidden_layer_size, 1)
            else:
                self.layer1 = nn.Linear(input_features, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = x.type(self.layer1.weight.dtype)
            x = nn.ReLU()(self.layer1(x))
            if self.layer2:
                x = self.sigmoid(self.layer2(x))
            else:
                x = self.sigmoid(x)
            return x

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = nn.BCELoss()(y_hat, y.view(-1, 1).float())
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=learning_rate)

    return SimpleNN


def run_model(train_dataset, X_test_scaled, Y_test, model_class, max_epochs, batch_size):
    model = model_class()

    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    trainer = L.Trainer(max_epochs=max_epochs)
    trainer.fit(model, train_loader)

    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(X_test_scaled)).numpy()

    predictions_bin = (predictions > PREDICTION_THRESHOLD).astype(int)

    # calculating p-value
    TN, FP, FN, TP = confusion_matrix(Y_test, predictions_bin).ravel()
    p_value = calculate_precision_p_value(tp=TP, fp=FP, fn=FN, tn=TN)

    return p_value


def evaluate_hyperparameters(
        learning_rate,
        max_epochs,
        batch_size,
        hidden_layer_size=None
):
    model_class = neural_network(input_features, hidden_layer_size, learning_rate)

    # running multiple times to get the median value
    # this helps account for random performance variation
    p_values = [
        run_model(
            train_dataset,
            X_test_scaled,
            Y_test,
            model_class,
            max_epochs=max_epochs,
            batch_size=batch_size)
        for _ in range(3)
    ]
    return median(p_values)


# loading data
train_dataset, X_test_scaled, Y_test, input_features = load_data()


# Loop through each combination of hyperparameters and evaluate them
def iterate_hyperparameters(
        learning_rates,
        batch_sizes,
        max_epochs_list,
        hidden_layer_sizes
):
    # Initialize an empty list to store the results
    results = []

    # Tracking where we are in the exploration
    iteration_count = 0
    total_iterations = len(learning_rates) * len(batch_sizes) * len(max_epochs_list) * len(hidden_layer_sizes)

    for learning_rate in learning_rates:
        for batch_size in batch_sizes:
            for max_epochs in max_epochs_list:
                for hidden_layer_size in hidden_layer_sizes:
                    start_time = time.time()
                    p_value = evaluate_hyperparameters(
                        learning_rate=learning_rate,
                        max_epochs=max_epochs,
                        batch_size=batch_size,
                        hidden_layer_size=hidden_layer_size
                    )
                    end_time = time.time()
                    execution_time = end_time - start_time
                    iteration_count += 1

                    print(f"iteration: {iteration_count} of {total_iterations}")
                    print(f"learning_rate: {learning_rate}")
                    print(f"max_epochs: {max_epochs}")
                    print(f"batch_size: {batch_size}")
                    print(f"hidden_layer_size: {hidden_layer_size}")
                    print(f"execution_time: {execution_time}")
                    print(f"p_value: {p_value}")

                    # Append the results as a new row to the results list
                    results.append([
                        learning_rate,
                        max_epochs,
                        batch_size,
                        hidden_layer_size,
                        execution_time,
                        p_value
                    ])

    # Create a DataFrame from the results list
    df = pd.DataFrame(results, columns=[
        'Learning Rate',
        'Max Epochs',
        'Batch Size',
        'Hidden Layer Size',
        'Execution Time',
        'P-Value'
    ])

    # Save the DataFrame to a CSV file
    df.to_csv('hyperparameter_tuning_results.csv', index=False)

    # Find the index of the row with the lowest P-value
    min_p_value_idx = df['P-Value'].idxmin()

    # Print the row with the lowest P-value
    print(df.loc[min_p_value_idx])


iterate_hyperparameters(
    learning_rates=[0.01, 0.001],
    batch_sizes=[32, 64, 256],
    max_epochs_list=[10, 50, 100],
    hidden_layer_sizes=[input_features, input_features // 2, input_features * 2],
)
