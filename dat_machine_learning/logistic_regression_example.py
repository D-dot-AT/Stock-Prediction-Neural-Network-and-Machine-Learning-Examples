import pandas as pd
from scipy.stats import fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, accuracy_score
from sklearn.preprocessing import StandardScaler

# Step 1: Data Preparation
# Load the training data
train_data = pd.read_csv('../example_data/train.csv', header=None)
X_train = train_data.iloc[:, :-1]
Y_train = train_data.iloc[:, -1]

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Step 2: Model Creation
# Create a logistic regression model
model = LogisticRegression(max_iter=500, random_state=0)

# Step 3: Training the Model
# Train the model using the training data
model.fit(X_train_scaled, Y_train)

# Step 4: Testing the Model
# Load the test data
test_data = pd.read_csv('../example_data/test.csv', header=None)
X_test = test_data.iloc[:, :-1]
Y_test = test_data.iloc[:, -1]

# Use the scaler to scale the feature data of test.csv
X_test_scaled = scaler.transform(X_test)

# Run the model on the scaled test data to get predictions
Y_pred = model.predict(X_test_scaled)

# Calculate metrics
TP = sum((Y_test == 1) & (Y_pred == 1))
FP = sum((Y_test == 0) & (Y_pred == 1))
TN = sum((Y_test == 0) & (Y_pred == 0))
FN = sum((Y_test == 1) & (Y_pred == 0))

precision = precision_score(Y_test, Y_pred)
accuracy = accuracy_score(Y_test, Y_pred)

# Step 5: Statistical Analysis
# Using the counts obtained from Step 4, perform Fisher's exact test to determine the p-value.
contingency_table = [[TP, FP], [FN, TN]]
_, p_value = fisher_exact(contingency_table)

# Step 6: Output
# Print the following information:
print(f'Precision: {precision}')
print(f'Accuracy: {accuracy}')
print(f'P-value of precision: {p_value}')
