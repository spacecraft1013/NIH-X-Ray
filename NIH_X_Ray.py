from __future__ import absolute_import
import numpy as np
import keras
import datetime
import os
import create_data_multithreaded_v2 as preprocessing
import sklearn as sk
import tensorflow as tf
from model import model_generator
import kerastuner as kt

model_save_name = "model_kt"

if os.path.exists(r"data/arrays/X_train_128.npy") == False:
    preprocessing.main()

print("Importing Arrays")
X_train = np.load(open(r"data/arrays/X_train_128.npy", "rb"))
y_train = np.load(open(r"data/arrays/y_train_128.npy", "rb"))
X_test = np.load(open(r"data/arrays/X_test_128.npy", "rb"))
y_test = np.load(open(r"data/arrays/y_test_128.npy", "rb"))

tuner = kt.BayesianOptimization(model_generator, objective='val_accuracy', max_trials=500, project_name="NIH X-Ray Model")

tuner.search(X_train, y_train, epochs=5, validation_split=0.1)

best_model = tuner.get_best_models(1)[0]
print(best_model.summary())

test_loss, test_acc = best_model.evaluate(X_test, y_test)
print('\nTest Accuracy: {}\nTest Loss: {}'.format(test_acc, test_loss))

best_model.save('data/models/{}.h5'.format(model_save_name))