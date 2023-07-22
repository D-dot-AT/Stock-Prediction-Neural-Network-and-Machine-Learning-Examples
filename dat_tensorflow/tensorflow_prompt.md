Create a Python neural net using tensorflow.  Do not use Keras.
The model will be trained using a file `train.csv` located at `../example_data/train.csv`.
`train.csv` has no header and all but the final column are floats.
The final column of `train.csv` is a 1/0 int that is a boolean label for the given row.
Create a classification model using train.
The `StandardScaler` will be a variable named `scaler`.
The model will be a variable named `model`.
The structure of `test.csv` is the same as that of `train.csv`.
Run the newly created model on `test.csv` to determine the precision and accuracy of the model.
print the precision and accuracy.
Save `model` as a pickle file to `output/scaler.pkl`, creating any necessary directories.
Save `scaler` as a pickle file to `output/scaler.pkl`.
