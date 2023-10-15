import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from common import print_statistics

# Step 1: Data Preparation
# Load the training data
train_data = pd.read_csv('../../example_data/train.csv', header=None)
# Separate data into X (features) and Y (labels)
X_train = train_data.iloc[:, :-1]
Y_train = train_data.iloc[:, -1]
# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Step 2: Model Creation and Training
# Create a model with input features matching the number of columns in X
model = GradientBoostingClassifier(random_state=0)
# Train the model using X and Y variables
model.fit(X_train, Y_train)

# Step 3: Model Testing
# Load and scale the test data
test_data = pd.read_csv('../../example_data/test.csv', header=None)
X_test = test_data.iloc[:, :-1]
Y_test = test_data.iloc[:, -1]
X_test = scaler.transform(X_test)
# Get predictions by running the model on the scaled test data
predictions = model.predict(X_test)
# Obtain the confusion matrix variables
TN, FP, FN, TP = confusion_matrix(Y_test, predictions).ravel()
# Print the statistics
print_statistics(tp=TP, fp=FP, tn=TN, fn=FN)

# Step 4: Creating Predictions
# Load data from latest.csv
latest_data = pd.read_csv('../../example_data/latest.csv')
# Extract stock tickers
stock_tickers = latest_data.iloc[:, 0]
# Extract feature vectors
X_latest = latest_data.iloc[:, 1:]
# Scale the features
X_latest = scaler.transform(X_latest)
# Predict scores using the model
scores = model.predict_proba(X_latest)[:, 1]
# Get top 5 stock tickers along with their percentage scores
top_5_idx = scores.argsort()[-5:][::-1]
for idx in top_5_idx:
    print(f"{stock_tickers[idx]}: {scores[idx]:.2f}%")
