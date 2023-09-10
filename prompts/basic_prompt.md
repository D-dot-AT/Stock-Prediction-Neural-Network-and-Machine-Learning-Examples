
Your task is to create a Python script to develop a model for stock data classification.
The goal is to build a model that maximizes precision.

Method: Neural Network
Library: Tensorflow

You will be training and testing using two datasets: `train.csv` 
and `test.csv`, both located in the `../example_data/` directory. 
The data in these files are structured similarly, where all but the final column are 
floats representing the features, and the final column are boolean 1/0 integer labels.

Here is a detailed breakdown of the tasks you need to complete:

## Step 1: Data Preparation

* Load the training data from `train.csv`. Note that the file does not contain a header.
* Separate the features and labels into X (features) and Y (labels) variables.
* Use the StandardScaler from scikit-learn to scale the feature data. Save this scaler as a variable named scaler.

## Step 2: Model Creation
* Create a classification model using library stated above.
* The model should have an appropriate architecture, with the number of input features matching the number of columns in your X variable (minus the label column).
* Save your model as a variable named `model`.

## Step 3: Training the Model
* Train your model using the training data (X and Y variables from Step 1).

## Step 4: Testing the Model
* Load the test data from `test.csv`. The structure of this file is the same as that of `train.csv`.
* Use the scaler to scale the feature data of `test.csv`.
* Run the model on the scaled test data to get predictions.
* Find the confusion matrix variables
* call the function `print_statistics(tp=TP, fp=FP, tn=TN, fn=FN)` imported via `from common import print_statistics`

## Step 5: Creating predictions
* load data from `latest.csv`.  The first column is a string stock ticker.  The remaining columns match the floating point feature vector of `train.csv`
* the data is the most recent data and has no label as we do not yet know the outcome.
* use the model to score latest data, print the 5 tickers with the top scores along with the scores represented as a percent.

## Additional Information
* Comment your code appropriately to explain complex or unclear sections.
* Code should be made with a "less is more" bias.