import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from common import print_statistics

# Step 1: Data Preparation
# 1.1 Load the training data
train_data = pd.read_csv('../../example_data/train.csv', header=None)

# 1.2 Separate data into X (features) and Y (labels)
X_train = train_data.iloc[:, :-1]
Y_train = train_data.iloc[:, -1]

# 1.3 Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Step 2: Model Creation and Training
# 2.1 Ensure the model's input features match the number of columns in X, excluding the label column
model = SVC(kernel='linear', class_weight='balanced', probability=True)

# 2.2 Train the model
model.fit(X_train, Y_train)

# Step 3: Model Testing
# 3.1 Load and scale the test data
test_data = pd.read_csv('../../example_data/test.csv', header=None)
X_test = scaler.transform(test_data.iloc[:, :-1])
Y_test = test_data.iloc[:, -1]

# 3.2 Get predictions
predictions = model.predict(X_test)

# 3.3 Obtain the confusion matrix variables and print statistics
TN, FP, FN, TP = confusion_matrix(Y_test, predictions).ravel()
print_statistics(tp=TP, fp=FP, tn=TN, fn=FN)

# Step 4: Creating Predictions
# 4.1 Load data from latest.csv
latest_data = pd.read_csv('../../example_data/latest.csv')
stock_tickers = latest_data.iloc[:, 0]
feature_vectors = latest_data.iloc[:, 1:]

# 4.2 Predict scores and print the top 5 stock tickers along with their percentage scores
score_predictions = model.predict_proba(scaler.transform(feature_vectors))[:, 1]
top_5_indices = score_predictions.argsort()[-5:][::-1]
for i in top_5_indices:
    print(f"{stock_tickers[i]}: {score_predictions[i]*100:.2f}%")
