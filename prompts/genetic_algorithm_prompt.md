* language: Python
* Method: Genetic algorithm.
* 
A chromosome is a set of genes that are all boolean conditions combined via AND.
A gene has three elements: the index of the feature vector, a lower bound and an upper bound.
A gene returns true if the value of the feature vector at the index is between the lower and upper bound.

Mutation on a chromosome can happen in the following ways:
* a gene can be mutated
* a new gene can be added
* a gene can be deleted.
The mutation rate for additions and deletions is 1% - meaning this type of mutation applies to only 1% of children.
The mutation rate for gene mutations is 5%.

Mutation on a gene can happen in the following way:
* the index is shifted up or down by 1.  if this results in being index out of bounds, then the mutation is not applied.
* the lower bound is multiplied by a random value between .95 and .99
* the upper bound is multiplied by a random value betwee 1.01 and 1.05

The test set is a set of feature vectors of floats each with boolean labels.
to evaluate a generation, randomly select a subset of 10% of the test set and evaluate on that.

Sexual recombination is handled in the following way:
Child is made of a combination of the genes of parents A and B.
Child starts out with no genes.
For each gene in parent A, add that gene to child with a random probability of 50%
Do the same for parent B.
Apply mutation to the child.

Develop a model, using the above criteria, for classifying stock data with a focus on maximizing precision. 
Follow the steps below using the datasets `train.csv` and `test.csv` in the 
`../example_data/` directory. Note that all but the final column in these files 
are floating-point feature values, and the final column contains boolean labels (1/0).

### Step 1: Data Preparation
1. Load the training data from `train.csv` (no header).
2. Separate data into X (features) and Y (labels).
3. If appropriate, scale the features using StandardScaler from scikit-learn, storing it in a variable named `scaler`.

### Step 2: Model Creation and traning
1. Ensure the model's input features match the number of columns in X, excluding the label column.
2. Train the `model` using the X and Y variables from Step 1.

### Step 3: Model Testing
1. Load and scale the test data from `test.csv` using `scaler`.
2. Get predictions by running the model on the scaled test data.
3. Obtain the confusion matrix variables and use `from common import print_statistics` to 
execute `print_statistics(tp=TP, fp=FP, tn=TN, fn=FN)`.

### Step 4: Creating Predictions
1. Load data from `latest.csv`; it contains a string stock ticker in the first column and floating-point 
feature vectors in the remaining columns.
2. Predict scores using the model and print the top 5 stock tickers along with their percentage scores.

### Note
- Comment complex or unclear code sections adequately.
- Adopt a "less is more" approach when coding.
- Create this as one continuous block of code
