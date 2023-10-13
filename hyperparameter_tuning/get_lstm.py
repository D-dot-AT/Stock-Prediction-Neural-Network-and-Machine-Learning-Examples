import pytorch_lightning as L
import torch
from torch import nn

from hyperparameter_tuning.nn_constants import Hyper, OPTIMIZER_CLASSES, WEIGHT_INITIALIZATIONS, LOSS_FUNCTIONS, ActivationFunction as af


def get_lstm(params, input_feature_size):
    class SimpleLSTM(L.LightningModule):
        def __init__(self):
            super().__init__()

            # Number of features for each time step in input sequence
            self.input_feature_size = input_feature_size
            # Number of LSTM layers stacked together
            self.num_layers = len(params[Hyper.HIDDEN_LAYERS])
            # LSTM layers
            self.lstm = nn.LSTM(input_size=input_feature_size,
                                hidden_size=params[Hyper.HIDDEN_LAYERS][0],
                                num_layers=self.num_layers,
                                dropout=params[Hyper.DROPOUT] if self.num_layers > 1 else 0,
                                batch_first=True)

            # Fully connected layer for classification
            self.fc = nn.Linear(params[Hyper.HIDDEN_LAYERS][-1], 1)

            self.activation_function = params[Hyper.ACTIVATION_FUNCTION]()
            self.dropout = nn.Dropout(params[Hyper.DROPOUT])
            self.init_weights()

        def init_weights(self):
            init_func = WEIGHT_INITIALIZATIONS.get(params[Hyper.WEIGHT_INITIALIZATION])
            for name, param in self.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    init_func(param)

        def forward(self, x):
            x = x.type(self.lstm.weight_ih_l0.dtype)
            x = x.view(-1, 1, input_feature_size)  # Reshaping to match the LSTM input shape
            h_0, _ = self.lstm(x)
            h_0 = h_0[:, -1, :]
            x = self.sigmoid(self.fc(h_0))
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
            optimizer = optimizer_class(self.parameters(), lr=params[Hyper.LEARNING_RATE])

            if params[Hyper.OPTIMIZER] not in ['AdamW'] and params[Hyper.L2_REGULARIZATION] > 0:
                for group in optimizer.param_groups:
                    group['weight_decay'] = params[Hyper.L2_REGULARIZATION]

            return optimizer

    return SimpleLSTM
