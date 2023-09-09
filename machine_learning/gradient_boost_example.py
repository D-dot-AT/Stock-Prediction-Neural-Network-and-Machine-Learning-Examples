import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from common import print_statistics

# Step 1: Data Preparation
# Load the training data
train_data = pd.read_csv('../example_data/train.csv', header=None)
X_train = train_data.iloc[:, :-1]
Y_train = train_data.iloc[:, -1]

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Step 2: Model Creation
model = GradientBoostingClassifier(random_state=0)

# Step 3: Training the Model
model.fit(X_train_scaled, Y_train)


# Step 4: Testing the Model
# Load the test data
test_data = pd.read_csv('../example_data/test.csv', header=None)
X_test = test_data.iloc[:, :-1]
Y_test = test_data.iloc[:, -1]

# Scaling the test data
X_test_scaled = scaler.transform(X_test)

# Getting predictions
Y_pred = model.predict(X_test_scaled)

# Calculate metrics
TP = sum((Y_test == 1) & (Y_pred == 1))
FP = sum((Y_test == 0) & (Y_pred == 1))
TN = sum((Y_test == 0) & (Y_pred == 0))
FN = sum((Y_test == 1) & (Y_pred == 0))

print_statistics(tp=TP, fp=FP, tn=TN, fn=FN)