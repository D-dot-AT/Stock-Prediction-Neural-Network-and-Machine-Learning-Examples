
Your task is to create a Python script to develop a neural network using for stock data classification.
The goal is to build a model that maximizes precision.

Library to use: PyTorch Lightning

You will be using two datasets: `train.csv` and `test.csv`, both located in the `../example_data/` directory. 
The data in these files are structured similarly, where all but the final column are 
floats representing the features, and the final column is a 1/0 integer representing the labels.

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
* Save the trained model and scaler as files for future use.

## Step 4: Testing the Model
* Load the test data from `test.csv`. The structure of this file is the same as that of `train.csv`.
* Use the scaler to scale the feature data of `test.csv`.
* Run the model on the scaled test data to get predictions.
* Calculate the following metrics based on the model predictions:
  * Precision
  * Accuracy
  * True Positives (TP)
  * False Positives (FP)
  * True Negatives (TN)
  * False Negatives (FN)

## Step 5: Statistical Analysis
* Using the counts obtained from Step 4, perform Fisher's exact test to determine the p-value. The test should be structured as follows:
* Model Distribution: Positives: (TP), Total: (TP + FP)
* Overall Distribution: Positives: (FN + TP), Total: (TP + FP + TN + FN)

## Step 6: Output
Print the following information:
* Precision
* Accuracy
* Counts for TP, FP, TN, FN
* The p-value from Fisher's exact test 

## Additional Information
* Comment your code appropriately to explain complex or unclear sections.
* Code should be made with a "less is more" bias