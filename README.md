# Stock Prediction Neural Network and Machine Learning Examples (Python)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

![Stock Prediction Neural Network and Machine Learning Examples ](https://repository-images.githubusercontent.com/669594930/b6bf1290-6278-4973-b7df-f698e226f23)

* [Simple Examples](#simple-examples)
* [Hyperparameter Optimization](#hyperparameter-optimization)

## Simple Examples

Simple ML and NN methods for those looking to learn new techniques for
stock prediction. These examples are meant to be easy to understand and highlight the essential components of each
method. Examples also show how to run the models on current data in order to get stock predictions.

ML examples include:
* Genetic algorithms
* Gradient boost
* K-means clustering
* Logistic regression
* Random Forest
* Support vector machines (SVM)

NN examples include:
* Feed-forward neural networks (FFNN)
* Long short-term memory (LSTM)
* Recurrant Neural Networkds (RNN)

NN library examples for:
* Keras
* Lightning
* PyTorch
* Tensorflow

## Hyperparameter Optimization
Designed for easy configuration of what hyperparameter values are explored.  
Multi-threaded processing for quick runtimes.

1. code is in `hyperparameter_tuning`
2. Edit `config.py` to suit your needs
3. run `main.py`

## Hyperparameters Explored

Here are the hyperparameters we currently search across:

- **Learning Rate**: The step size at each iteration while moving towards a minimum of the loss function.
- **Max Epochs**: The maximum number of times the learning algorithm will work through the entire training dataset.
- **Batch Size**: The number of training examples utilized in one iteration.
- **Hidden Layers**: The architecture of the neural network in terms of layers and nodes.
- **Loss Function**: Determines the difference between the network's predictions and the actual data.
- **Activation Function**: The function used to introduce non-linearity to the network.
- **Optimizer**: Algorithms or methods used to change the attributes of the neural network such as weights to reduce the
  losses.
- **Dropout**: A regularization method where input and recurrent connections to a layer are probabilistically excluded
  from during training.
- **L1 Regularization**: Adds a penalty for non-zero coefficients.
- **L2 Regularization**: Adds a penalty for larger coefficient values.
- **Weight Initialization**: Methods to set the initial random weights of neural network layers.

Hyperparameter readme here:  [Hyperparameter Tuning](hyperparameter_tuning/README.md)

## Getting Started

1. Clone this repository.
2. Navigate to the project directory.
3. Install the necessary libraries:

```bash
pip install -r requirements.txt
```

Then, run any of the scripts in `simple_examples`


## About the Example Data

The data provided in `example_data` is an example of what is downloadable on the D.AT platform.

This dataset encapsulates 5 years of price data of the companies comprising the S&P 500,
segmented into intervals of 30 trading days each. The data in each segment
has been normalized using a method where values are divided by the most
recent data point within the segment. Each row in the dataset represents a
specific segment, providing a snapshot of the stock data available on a
particular trading day. Rows are labeled to indicate when the
stock had a minimum gain of 5% within the subsequent 10 trading days.

* `train.csv`: Of the 5 years, it contains the first 4 years of data.
* `test.csv`: Of the 5 years, it contains the final year of data.
* `latest.csv`: This file contains data from the most recent trading
  day for all stocks listed. While it lacks labels (since these pertain to future events),
  each row maintains the same feature vector structure as those in the `train` and `test`
  files. The rows commence with the stock ticker symbol, serving as a key tool to pinpoint
  stocks with promising prospects for good performance.

### Getting new data

Recent data customizable with different trading strategies and feature engineering options can be [downloaded for free
at D.AT](https://d.at/ref/github-python-examples).


