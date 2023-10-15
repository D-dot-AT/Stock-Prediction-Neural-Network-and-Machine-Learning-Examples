import pandas as pd
from keras import layers
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from common import print_statistics

# Step 1: Data Preparation
# Load training data
data_train = pd.read_csv('../../example_data/train.csv', header=None)
# Separate into X and Y
X_train = data_train.iloc[:, :-1]
Y_train = data_train.iloc[:, -1]
# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Step 2: Model Creation
# Develop a classification model using TensorFlow
input_dim = X_train_scaled.shape[1]
model = keras.Sequential([
    layers.Dense(64, input_dim=input_dim,  activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])

# Compile the model with precision as a metric
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[keras.metrics.Precision()])

# Step 3: Model Training
# Train the model using the X and Y variables from Step 1
model.fit(X_train_scaled, Y_train, epochs=10, batch_size=32)

# Step 4: Model Testing
# Load and scale the test data
data_test = pd.read_csv('../../example_data/test.csv', header=None)
X_test = data_test.iloc[:, :-1]
Y_test = data_test.iloc[:, -1]
X_test_scaled = scaler.transform(X_test)
# Get predictions
predictions = (model.predict(X_test_scaled) > 0.5).astype('int32')
# Obtain the confusion matrix variables
tn, fp, fn, tp = confusion_matrix(Y_test, predictions).ravel()
# Use print_statistics to print the statistics

print_statistics(tp=tp, fp=fp, tn=tn, fn=fn)

# Step 5: Creating Predictions
# Load data
data_latest = pd.read_csv('../../example_data/latest.csv')
stock_tickers = data_latest.iloc[:, 0]
feature_vectors = data_latest.iloc[:, 1:]
# Scale the feature vectors
feature_vectors_scaled = scaler.transform(feature_vectors)
# Predict scores
scores = model.predict(feature_vectors_scaled)
# Get the top 5 stock tickers and their percentage scores
top_5_idx = scores[:, 0].argsort()[-5:][::-1]
print("\nStocks most likely to gain 5% in the next 10 trading days")
for idx in top_5_idx:
    print(f'{stock_tickers[idx]}: {scores[idx][0]*100:.2f}%')
