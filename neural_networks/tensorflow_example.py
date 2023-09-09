import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from common import print_statistics

# Step 1: Data Preparation
# Load the training data
train_data = pd.read_csv('../example_data/train.csv', header=None)
X_train = train_data.iloc[:, :-1]
Y_train = train_data.iloc[:, -1]

# Scale the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Step 2: Model Creation
# Create a classification model
input_features = X_train.shape[1]
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_features,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 3: Training the Model
# Train the model
model.fit(X_train, Y_train, epochs=10, batch_size=32)

# Step 4: Testing the Model
# Load the test data
test_data = pd.read_csv('../example_data/test.csv', header=None)
X_test = test_data.iloc[:, :-1]
Y_test = test_data.iloc[:, -1]

# Scale the feature data
X_test = scaler.transform(X_test)

# Get predictions
Y_pred = (model.predict(X_test) > 0.6).astype(int).flatten()

# Calculate metrics
precision = precision_score(Y_test, Y_pred)
accuracy = accuracy_score(Y_test, Y_pred)
TN, FP, FN, TP = confusion_matrix(Y_test, Y_pred).ravel()

print_statistics(tp=TP, fp=FP, tn=TN, fn=FN)