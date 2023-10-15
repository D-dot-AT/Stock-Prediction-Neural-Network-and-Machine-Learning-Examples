import pytorch_lightning as L
import torch
from torch import nn

from hyperparameter_tuning.nn_constants import (
    Hyper,
    OPTIMIZER_CLASSES,
    WEIGHT_INITIALIZATIONS,
    LOSS_FUNCTIONS,
    ACTIVATION_FUNCTIONS
)


def get_ffnn(params, input_feature_size):
    """Returns a FFNN (Feed Forward Neural Network) class based on a set of hyperparameters."""

    class SimpleNN(L.LightningModule):
        def __init__(self):
            super().__init__()

            # Calculate and guarantee a minimum hidden layer size of one
            sizes = [max(int(round(input_feature_size * h)), 1) for h in params[Hyper.HIDDEN_LAYERS]]

            # Construct all layer sizes
            all_sizes = [input_feature_size] + sizes + [1]
            self.layers = nn.ModuleList([
                nn.Linear(all_sizes[i], all_sizes[i + 1])
                for i in range(len(all_sizes) - 1)
            ])

            self.activation_function = ACTIVATION_FUNCTIONS[params[Hyper.ACTIVATION_FUNCTION]]()
            self.dropout = nn.Dropout(params[Hyper.DROPOUT])
            self.init_weights()

        def init_weights(self):
            """Initialize weights using the specified method."""
            init_func = WEIGHT_INITIALIZATIONS[params[Hyper.WEIGHT_INITIALIZATION]]
            for layer in self.layers:
                init_func(layer.weight)

        def forward(self, x):
            """Forward pass."""
            x = x.type(self.layers[0].weight.dtype)
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i < len(self.layers) - 1:  # Apply dropout and activation to non-output layers
                    x = self.activation_function(x)
                    x = self.dropout(x)
            return nn.Sigmoid()(x)  # Assuming binary classification

        def training_step(self, batch, batch_idx):
            """Compute the training loss."""
            x, y = batch
            y_hat = self(x)
            loss_func = LOSS_FUNCTIONS[params[Hyper.LOSS_FUNCTION]]
            loss = loss_func()(y_hat, y.view(-1, 1).float())

            # L1 Regularization
            l1_reg = sum(torch.norm(param, 1) for param in self.parameters())
            loss += params[Hyper.L1_REGULARIZATION] * l1_reg

            return loss

        def configure_optimizers(self):
            """Setup the optimizer with the given hyperparameters."""
            optimizer_class = OPTIMIZER_CLASSES[params[Hyper.OPTIMIZER]]
            optimizer = optimizer_class(self.parameters(), lr=params[Hyper.LEARNING_RATE])

            # Handle L2 regularization outside of optimizers like AdamW
            if params[Hyper.OPTIMIZER] not in ['AdamW'] and params[Hyper.L2_REGULARIZATION] > 0:
                for group in optimizer.param_groups:
                    group['weight_decay'] = params[Hyper.L2_REGULARIZATION]

            return optimizer

    return SimpleNN
