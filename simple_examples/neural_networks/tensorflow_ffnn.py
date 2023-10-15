import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from common import print_statistics

# Step 1: Data Preparation
data_train = pd.read_csv('../../example_data/train.csv', header=None)
X_train = data_train.iloc[:, :-1]
Y_train = data_train.iloc[:, -1]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Step 2: Model Creation
input_dim = X_train_scaled.shape[1]
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_dim=input_dim, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

# Compile the model
precision_metric = tf.keras.metrics.Precision()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[precision_metric])

# Step 3: Model Training
model.fit(X_train_scaled, Y_train, epochs=10, batch_size=32)

# Step 4: Model Testing
data_test = pd.read_csv('../../example_data/test.csv', header=None)
X_test = data_test.iloc[:, :-1]
Y_test = data_test.iloc[:, -1]
X_test_scaled = scaler.transform(X_test)
predictions = (model.predict(X_test_scaled) > 0.5).astype('int32')
tn, fp, fn, tp = confusion_matrix(Y_test, predictions).ravel()
print_statistics(tp=tp, fp=fp, tn=tn, fn=fn)

# Step 5: Creating Predictions
data_latest = pd.read_csv('../../example_data/latest.csv')
stock_tickers = data_latest.iloc[:, 0]
feature_vectors = data_latest.iloc[:, 1:]
feature_vectors_scaled = scaler.transform(feature_vectors)
scores = model.predict(feature_vectors_scaled)
top_5_idx = scores[:, 0].argsort()[-5:][::-1]
for idx in top_5_idx:
    print(f'{stock_tickers[idx]}: {scores[idx][0]*100:.2f}%')
