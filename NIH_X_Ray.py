from __future__ import absolute_import
import numpy as np
import keras
import datetime
import os
import multithreaded_preprocessing as preprocessing
import sklearn as sk
import tensorflow as tf
from model import model_generator
import kerastuner as kt

model_save_name = "densenet201"

if os.path.exists("data/arrays/X_train_128.npy") == False:
    preprocessing.main()

print("Importing Arrays")
X_train = np.load(open("data/arrays/X_train_128.npy", "rb"))
y_train = np.load(open("data/arrays/y_train_128.npy", "rb"))
X_test = np.load(open("data/arrays/X_test_128.npy", "rb"))
y_test = np.load(open("data/arrays/y_test_128.npy", "rb"))

densenet = keras.applications.DenseNet201(input_shape=(128, 128, 1), weights=None, classes=15)

# tuner = kt.BayesianOptimization(model_generator, objective='val_accuracy', max_trials=500, project_name="NIH X-Ray Model")

# tuner.search(X_train, y_train, epochs=5, validation_split=0.1)

# best_model = tuner.get_best_models(1)[0]
# print(best_model.summary())

densenet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

log_dir = "data/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoint_path = "data/checkpoints/" + model_save_name + "-{epoch:03d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
densenet.save_weights(checkpoint_path.format(epoch=0))
cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

densenet.fit(X_train, y_train, batch_size=32, epochs=250, validation_data=(X_test, y_test), callbacks=[tensorboard_callback, cp_callback])

test_loss, test_acc = densenet.evaluate(X_test, y_test)
print('\nTest Accuracy: {}\nTest Loss: {}'.format(test_acc, test_loss))

densenet.save('data/models/{}.h5'.format(model_save_name))