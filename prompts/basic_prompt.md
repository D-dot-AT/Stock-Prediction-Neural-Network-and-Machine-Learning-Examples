* language: Python
* Method: Neural Network
* Library: Tensorflow

Develop a model for classifying stock data with a focus on maximizing precision. 
Follow the steps below using the datasets `train.csv` and `test.csv` in the 
`../example_data/` directory. Note that all but the final column in these files 
are floating-point feature values, and the final column contains boolean labels (1/0).

### Step 1: Data Preparation
1. Load the training data from `train.csv` (no header).
2. Separate data into X (features) and Y (labels).
3. Scale the features using StandardScaler from scikit-learn, storing it in a variable named `scaler`.

### Step 2: Model Creation
1. Develop a classification model using TensorFlow.
2. Ensure the model's input features match the number of columns in X, excluding the label column.
3. Assign the model to a variable named `model`.

### Step 3: Model Training
1. Train the `model` using the X and Y variables from Step 1.

### Step 4: Model Testing
1. Load and scale the test data from `test.csv` using `scaler`.
2. Get predictions by running the model on the scaled test data.
3. Obtain the confusion matrix variables and use `from common import print_statistics` to execute `print_statistics(tp=TP, fp=FP, tn=TN, fn=FN)`.

### Step 5: Creating Predictions
1. Load data from `latest.csv`; it contains a string stock ticker in the first column and floating-point feature vectors in the remaining columns.
2. Predict scores using the `model` and print the top 5 stock tickers along with their percentage scores.

### Note
- Comment complex or unclear code sections adequately.
- Adopt a "less is more" approach when coding.
