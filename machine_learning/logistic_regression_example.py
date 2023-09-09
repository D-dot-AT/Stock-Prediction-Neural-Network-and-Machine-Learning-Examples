import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Step 1: Data Preparation
# Load the training data
from common import print_statistics

train_data = pd.read_csv('../example_data/train.csv', header=None)
X_train = train_data.iloc[:, :-1]
Y_train = train_data.iloc[:, -1]

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Step 2: Model Creation
# Create a logistic regression model with hyperparameter tuning
model = LogisticRegression(random_state=0, class_weight='balanced', penalty='l1', solver='liblinear')

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
Y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
threshold = 0.6  # Adjust this value based on your analysis to maximize precision
Y_pred = (Y_pred_prob >= threshold).astype(int)

# Calculate metrics
TP = sum((Y_test == 1) & (Y_pred == 1))
FP = sum((Y_test == 0) & (Y_pred == 1))
TN = sum((Y_test == 0) & (Y_pred == 0))
FN = sum((Y_test == 1) & (Y_pred == 0))

print_statistics(tp=TP, fp=FP, tn=TN, fn=FN)


