import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from common import print_statistics

# Step 1: Data Preparation
# 1. Load the training data from train.csv (no header)
data_train = pd.read_csv('../../example_data/train.csv', header=None)
# 2. Separate data into X (features) and Y (labels)
X_train = data_train.iloc[:, :-1]
Y_train = data_train.iloc[:, -1]
# 3. If appropriate, scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Step 2: Model Creation and training
# 1. Ensure the model's input features match the number of columns in X, excluding the label column
# 2. Train the model using the X and Y variables from Step 1
model = KMeans(n_clusters=2, random_state=42) # We use 2 clusters since there are two classes (1 and 0)
model.fit(X_train_scaled, Y_train)

# Step 3: Model Testing
# 1. Load and scale the test data from test.csv using scaler
data_test = pd.read_csv('../../example_data/test.csv', header=None)
X_test = data_test.iloc[:, :-1]
Y_test = data_test.iloc[:, -1]
X_test_scaled = scaler.transform(X_test)

# 2. Get predictions by running the model on the scaled test data
Y_pred = model.predict(X_test_scaled)

# 3. Obtain the confusion matrix variables and use print_statistics to execute print_statistics
tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
print_statistics(tp=tp, fp=fp, tn=tn, fn=fn)

# Step 4: Creating Predictions
# 1. Load data from latest.csv
data_latest = pd.read_csv('../../example_data/latest.csv')
stock_tickers = data_latest.iloc[:, 0]
features_latest = data_latest.iloc[:, 1:]

# 2. Predict scores using the model and print the top 5 stock tickers along with their percentage scores
scores = model.transform(features_latest) # Get the distance of each point from cluster centers
# Getting the percentage score as the inverse of distance to the cluster center, normalized
# by the sum of distances to all centers
percentage_scores = 1 - (scores[:, 0] / scores.sum(axis=1))
top_5_indices = percentage_scores.argsort()[-5:][::-1] # Get indices of top 5 scores

# Print top 5 stock tickers with their percentage scores
for i in top_5_indices:
    print(f"{stock_tickers[i]}: {percentage_scores[i]:.2%}")

