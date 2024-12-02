# Stock Prediction Neural Network and Machine Learning Examples (Python)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

![Stock Prediction Neural Network and Machine Learning Examples ](https://repository-images.githubusercontent.com/669594930/1b661bb8-d5d8-40ad-9d94-c3084f3df2fc)

## Contents:
* [Simple Examples](#simple-examples)
* [Hyperparameter Optimization](#hyperparameter-optimization)
* [Getting Started](#getting-started)
* [About the Example Stock Data](#about-the-example-stock-data)

## Overview
These are ML and NN methods ready to launch out of the box. Designed to be easy for those looking to learn new techniques for stock prediction. These examples are meant to be simple to understand and highlight the essential components of each method. Examples also show how to run the models on current data in order to get stock predictions.

### Machine Learning examples:
* Genetic algorithms
* Gradient boost
* K-means clustering
* Logistic regression
* Random Forest
* Support vector machines (SVM)

### Neural Net examples:
* Feed-forward neural networks (FFNN)
* Long short-term memory (LSTM)
* Recurrent Neural Networks (RNN)

### Neural Net library examples:
* Keras
* Lightning
* PyTorch
* Tensorflow

## Getting Started

1. **Clone this repository.**
2. **Navigate to the project directory.**
3. **Install the necessary libraries:**

```bash
pip install -r requirements.txt
```

4. **Download the starter data:**
   - These examples require precisely formatted stock data with the following properties:
     * **Windowed:** Time series is segmented into regular windows.
     * **Boolean labels:** Rather than predicting specific values at specific times, the data is classified.
     * **Test/Train split:** Split chronologically for no data overlaps and look-ahead bias.
   - [Download the starter data.](https://d.at/example-data) and save the `example_data` directory to this project folder.

5. **Run any of the scripts in `simple_examples`.**

## Neural Net Hyperparameter Optimization
Designed for easy configuration of what hyperparameter values are explored. Multi-threaded processing for quick runtimes.

1. Code is in `hyperparameter_tuning`
2. Edit `config.py` to suit your needs
3. Run `hyper_main.py`

Hyperparameter readme here: [Hyperparameter Tuning](hyperparameter_tuning/README.md)

## About the Example Stock Data

This code can be run with the example stock data available at [D.AT example data](https://d.at/example-data).

This dataset encapsulates 5 years of price data of the companies comprising the S&P 500, segmented into intervals of 30 trading days each. The data in each segment has been normalized using a method where values are divided by the most recent data point within the segment. Each row in the dataset represents a specific segment, providing a snapshot of the stock data available on a particular trading day. Rows are labeled to indicate when the stock had a minimum gain of 5% within the subsequent 10 trading days.

* `train.csv`: Of the 5 years, it contains the first 4 years of data.
* `test.csv`: Of the 5 years, it contains the final year of data.
* `latest.csv`: This file contains data from the most recent trading day for all stocks listed. While it lacks labels (since these pertain to future events), each row maintains the same feature vector structure as those in the `train` and `test` files. The rows commence with the stock ticker symbol, serving as a key tool to pinpoint stocks with promising prospects for good performance.

### Getting new data

The example data is static and does not contain current stock price values.
Recent data customizable with different trading strategies and feature engineering options can be [downloaded for free at D.AT](https://d.at/ref/github-python-examples).
