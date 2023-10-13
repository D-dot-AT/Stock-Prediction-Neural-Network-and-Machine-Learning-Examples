import pytorch_lightning as L
import torch
from torch import nn

from hyperparameter_tuning.config import Hyper
from hyperparameter_tuning.nn_constants import OPTIMIZER_CLASSES, WEIGHT_INITIALIZATIONS, LOSS_FUNCTIONS, \
    ACTIVATION_FUNCTIONS


# Returns a NN class based on a set of hyperparameters
def get_ffnn(params, input_feature_size):
    class SimpleNN(L.LightningModule):
        def __init__(self):
            super().__init__()

            # Calculate hidden layer sizes
            # Guarantee size of at least one
            sizes = [max(int(round(input_feature_size * h)), 1) for h in params[Hyper.HIDDEN_LAYERS]]

            # Dynamically build layers
            all_sizes = [input_feature_size] + sizes + [1]
            self.layers = nn.ModuleList([nn.Linear(all_sizes[i], all_sizes[i + 1])
                                         for i in range(len(all_sizes) - 1)])

            # Settings
            self.activation_function = ACTIVATION_FUNCTIONS[params[Hyper.ACTIVATION_FUNCTION]]()
            self.dropout = nn.Dropout(params[Hyper.DROPOUT])
            self.init_weights()

        def init_weights(self):
            init_func = WEIGHT_INITIALIZATIONS.get(params[Hyper.WEIGHT_INITIALIZATION])
            for layer in self.layers:
                init_func(layer.weight)

        def forward(self, x):
            x = x.type(self.layers[0].weight.dtype)
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i < len(self.layers) - 1:  # Only apply dropout and activation to non-last layers
                    x = self.activation_function(x)
                    x = self.dropout(x)
            x = nn.Sigmoid()(x)  # Using Sigmoid for the output as I assume it's a binary classification
            return x

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss_function = LOSS_FUNCTIONS[params[Hyper.LOSS_FUNCTION]]
            loss = loss_function()(y_hat, y.view(-1, 1).float())

            # L1 Regularization
            l1_reg = 0.0
            for param in self.parameters():
                l1_reg += torch.norm(param, 1)
            loss = loss + params[Hyper.L1_REGULARIZATION] * l1_reg

            return loss

        def configure_optimizers(self):
            optimizer_class = OPTIMIZER_CLASSES[params[Hyper.OPTIMIZER]]
            print(f"optimizer_class: {optimizer_class}")
            optimizer = optimizer_class(self.parameters(), lr=params[Hyper.LEARNING_RATE])

            # If optimizer has its own L2 regularization handling, skip. For instance, AdamW.
            if params[Hyper.OPTIMIZER] not in ['AdamW'] and params[Hyper.L2_REGULARIZATION] > 0:
                for group in optimizer.param_groups:
                    group['weight_decay'] = params[Hyper.L2_REGULARIZATION]

            return optimizer

    return SimpleNN
