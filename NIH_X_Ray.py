from __future__ import absolute_import
import numpy as np
import keras
import pickle
import datetime
import os
import create_data
import sklearn as sk
import tensorflow as tf
import model
import kerastuner as kt

model_save_name = "model_kt"

if os.path.exists(r"data/arrays/X_train_128.pickle") == False:
    create_data.main()

print("Importing Arrays")
X_train = pickle.load(open(r"data/arrays/X_train_128.pickle", "rb"))
y_train = pickle.load(open(r"data/arrays/y_train_128.pickle", "rb"))
X_test = pickle.load(open(r"data/arrays/X_test_128.pickle", "rb"))
y_test = pickle.load(open(r"data/arrays/y_test_128.pickle", "rb"))

model = model.model(128)
print(model.summary())

tuner = kt.Hyperband(model, objective='val_accuracy', max_epochs=250, hyperband_iterations=5)

tuner.search(X_train, y_train, epochs=250, validation_data=(X_test, y_test))

best_model = tuner.get_best_models(1)[0]
print(tuner.get_best_hyperparameters(1)[0])

test_loss, test_acc = best_model.evaluate(X_test, y_test)
print('\nTest Accuracy: {}\nTest Loss: {}'.format(test_acc, test_loss))

best_model.save('data/models/{}.h5'.format(model_save_name))