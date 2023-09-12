import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from common import print_statistics

# Step 1: Data Preparation
data_train = pd.read_csv('../example_data/train.csv', header=None)
X_train = data_train.iloc[:, :-1]
Y_train = data_train.iloc[:, -1]

# If appropriate, scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Step 2: Model Creation and training
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, Y_train)

# Step 3: Model Testing
data_test = pd.read_csv('../example_data/test.csv', header=None)
X_test = data_test.iloc[:, :-1]
Y_test = data_test.iloc[:, -1]
X_test_scaled = scaler.transform(X_test)

# Get predictions by running the model on the scaled test data
Y_pred = model.predict(X_test_scaled)

# Obtain the confusion matrix variables and use print_statistics to execute print_statistics
tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
print_statistics(tp=tp, fp=fp, tn=tn, fn=fn)

# Step 4: Creating Predictions
data_latest = pd.read_csv('../example_data/latest.csv')
stock_tickers = data_latest.iloc[:, 0]
features_latest = data_latest.iloc[:, 1:]

# Predict scores using the model and print the top 5 stock tickers along with their percentage scores
scores = model.predict_proba(features_latest)
percentage_scores = scores[:, 1]  # Get the probability of belonging to class 1
top_5_indices = percentage_scores.argsort()[-5:][::-1]  # Get indices of top 5 scores

# Print top 5 stock tickers with their percentage scores
for i in top_5_indices:
    print(f"{stock_tickers[i]}: {percentage_scores[i]:.2%}")
