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
#import kerastuner as kt
import autokeras as ak

model_save_name = "model_ak"

if os.path.exists(r"data/arrays/X_train_128.pickle") == False:
    create_data.main()

print("Importing Arrays")
X_train = pickle.load(open(r"data/arrays/X_train_128.pickle", "rb"))
y_train = pickle.load(open(r"data/arrays/y_train_128.pickle", "rb"))
X_test = pickle.load(open(r"data/arrays/X_test_128.pickle", "rb"))
y_test = pickle.load(open(r"data/arrays/y_test_128.pickle", "rb"))

# model = model.model(128)
# print(model.summary())

#tuner = kt.Hyperband(model, objective='val_accuracy', max_epochs=250, hyperband_iterations=5)

model = ak.ImageClassifier(max_trials=10)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=250)

# model = keras.applications.inception_v3.InceptionV3(input_shape=(128, 128, 1), include_top=True, weights=None, classes=15)
# model = keras.applications.mobilenet_v2.MobileNetV2(input_shape=(128, 128, 1), include_top=True, weights=None, classes=15)
# model = keras.applications.vgg19.VGG19(input_shape=(128, 128, 1), include_top=True, weights=None, classes=15)

# print("Loading from previous checkpoint")
# model.load_weights("data/checkpoints/.ckpt")

# print("Computing sample weights...")
# class_weights = sk.utils.class_weight.compute_sample_weight('balanced', y_train)

# checkpoint_path = "data/checkpoints/" + model_save_name + "-{epoch:03d}.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# model.save_weights(checkpoint_path.format(epoch=0))

# cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

# model.fit(X_train, y_train, epochs=250, batch_size=64, validation_data=(X_test, y_test), class_weight=class_weights, callbacks=[cp_callback])
# tuner.search(X_train, y_train, epochs=250, validation_data=(X_test, y_test))

# best_model = tuner.get_best_models(1)[0]
# print(tuner.get_best_hyperparameters(1)[0])

test_loss, test_acc = model.evaluate(X_test, y_test)
print('\nTest Accuracy: {}\nTest Loss: {}'.format(test_acc, test_loss))

#model.save('data/models/{}.h5'.format(model_save_name))
