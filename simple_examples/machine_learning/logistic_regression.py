import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from common import print_statistics

# Step 1: Data Preparation
# 1. Load the training data
train_data = pd.read_csv('../../example_data/train.csv', header=None)

# 2. Separate data into X (features) and Y (labels)
X_train = train_data.iloc[:, :-1]
Y_train = train_data.iloc[:, -1]

# 3. Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Step 2: Model Creation and Training
# 1. Ensure the model's input features match the number of columns in X
model = LogisticRegression(max_iter=1000)

# 2. Train the model using the X and Y variables
model.fit(X_train, Y_train)

# Step 3: Model Testing
# 1. Load and scale the test data
test_data = pd.read_csv('../../example_data/test.csv', header=None)
X_test = scaler.transform(test_data.iloc[:, :-1])
Y_test = test_data.iloc[:, -1]

# 2. Get predictions
Y_pred = model.predict(X_test)

# 3. Obtain the confusion matrix variables
TN, FP, FN, TP = confusion_matrix(Y_test, Y_pred).ravel()

# Execute print_statistics
print_statistics(tp=TP, fp=FP, tn=TN, fn=FN)

# Step 4: Creating Predictions
# 1. Load data from latest.csv
latest_data = pd.read_csv('../../example_data/latest.csv')

# 2. Predict scores using the model and print the top 5 stock tickers along with their percentage scores
stock_tickers = latest_data.iloc[:, 0]
feature_vectors = latest_data.iloc[:, 1:]
feature_vectors_scaled = scaler.transform(feature_vectors)
scores = model.predict_proba(feature_vectors_scaled)[:, 1]

# Getting the top 5 stock tickers along with their percentage scores
top_5_indices = scores.argsort()[-5:][::-1]
for idx in top_5_indices:
    print(f'{stock_tickers[idx]}: {scores[idx] * 100:.2f}%')
