import keras
import numpy as np
from sklearn.metrics import confusion_matrix
import argparse

parser = argparse.ArgumentParser(description='Create a confusion matrix from a model \
                                              and a testing set.')
parser.add_argument('model', type=str,
                    help='The model to create a confusion matrix on.')
parser.add_argument('--test-features', type=str,
                    help='Path to a .npy file containing the feature array \
                          of the testing set.')
parser.add_argument('--test-labels', type=str,
                    help='Path to a .npy file containing the label array \
                          of the testing set.')
args = parser.parse_args()

X_test = np.load(args.test_features)
y_test = np.load(args.test_labels)

model = keras.models.load_model(args.model)

y_pred = model.predict(X_test, verbose=1)

y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

matrix = confusion_matrix(y_test, y_pred)

print(matrix)
