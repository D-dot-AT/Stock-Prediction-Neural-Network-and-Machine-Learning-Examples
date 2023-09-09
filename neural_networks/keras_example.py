import numpy as np
import pandas as pd
from keras import layers
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from common import print_statistics

# Step 1: Data Preparation
# Load the training data
train_data = pd.read_csv('../example_data/train.csv', header=None)
X_train = train_data.iloc[:, :-1]
Y_train = train_data.iloc[:, -1]

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Step 2: Model Creation
# Get the number of features
input_features = X_train_scaled.shape[1]

# Create a classification model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(input_features,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 3: Training the Model
# Train the model
model.fit(X_train_scaled, Y_train, epochs=10, batch_size=32)

# Save the model and scaler
# model.save('model.h5')
# scaler.save('scaler.pkl')

# Step 4: Testing the Model
# Load the test data
test_data = pd.read_csv('../example_data/test.csv', header=None)
X_test = test_data.iloc[:, :-1]
Y_test = test_data.iloc[:, -1]

# Scale the feature data
X_test_scaled = scaler.transform(X_test)

# Get the predictions
predictions = (model.predict(X_test_scaled) > 0.6).astype("int32")

# Calculate metrics
Y_test_array = Y_test.to_numpy().ravel()
predictions_array = predictions.ravel()

TP = np.sum((predictions_array == 1) & (Y_test_array == 1))
FP = np.sum((predictions_array == 1) & (Y_test_array == 0))
TN = np.sum((predictions_array == 0) & (Y_test_array == 0))
FN = np.sum((predictions_array == 0) & (Y_test_array == 1))

print_statistics(tp=TP, fp=FP, tn=TN, fn=FN)
