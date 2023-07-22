import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score
import os
import pickle

# Load and preprocess training data
train = pd.read_csv('../example_data/train.csv', header=None)
X_train = train.iloc[:,:-1].values
y_train = train.iloc[:,-1].values

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Specify the model
feature_columns = [tf.feature_column.numeric_column("x", shape=[X_train.shape[1]])]
model = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                   hidden_units=[10, 20, 10],
                                   n_classes=2)

# Train the model
train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(x={"x": X_train},
                                                              y=y_train,
                                                              num_epochs=50,
                                                              shuffle=True)
model.train(input_fn=train_input_fn, steps=5000)

# Save the scaler and model
os.makedirs('output', exist_ok=True)
pickle.dump(scaler, open('output/scaler.pkl', 'wb'))

# Define serving input function
serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
    tf.feature_column.make_parse_example_spec(feature_columns)
)
model.export_saved_model('output', serving_input_fn)

# Load and preprocess test data
test = pd.read_csv('../example_data/test.csv', header=None)
X_test = test.iloc[:,:-1].values
y_test = test.iloc[:,-1].values
X_test = scaler.transform(X_test)

# Evaluate the model on test data
input_fn_test = tf.compat.v1.estimator.inputs.numpy_input_fn(x={"x": X_test},
                                                             y=y_test,
                                                             num_epochs=1,
                                                             shuffle=False)
predictions = list(model.predict(input_fn=input_fn_test))
y_pred = np.array([int(p['classes'][0]) for p in predictions])

# Calculate and print precision and accuracy
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f'Precision: {precision}')
print(f'Accuracy: {accuracy}')
