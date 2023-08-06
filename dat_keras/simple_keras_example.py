import os
import pickle

import pandas as pd
from sklearn.metrics import precision_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense
from keras.models import Sequential

# Load and preprocess training data
train = pd.read_csv('../example_data/train.csv', header=None)
X_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Define model
model = Sequential()
model.add(Dense(16, input_dim=X_train.shape[1], activation='relu'))  # Input layer
model.add(Dense(16, activation='relu'))  # Hidden layer
model.add(Dense(1, activation='sigmoid'))  # Output layer

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Load and preprocess test data
test = pd.read_csv('../example_data/test.csv', header=None)
X_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]

# Standardize test data using the same scaler used for training data
X_test = scaler.transform(X_test)

# Predict the labels for test data
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Calculate accuracy and precision
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print(f'Accuracy: {accuracy*100:.2f}%')
print(f'Precision: {precision*100:.2f}%')

# Create output directory if it does not exist
os.makedirs('output', exist_ok=True)

# Save the model and scaler
pickle.dump(model, open('output/model.pkl', 'wb'))
pickle.dump(scaler, open('output/scaler.pkl', 'wb'))
