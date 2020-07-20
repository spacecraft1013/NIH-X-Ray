from __future__ import absolute_import
import numpy as np
import keras
import datetime
import os
import create_data
import sklearn as sk
import tensorflow as tf
import model

model_save_name = "model_adam"

if os.path.exists(r"data/arrays/X_train_128.np") == False:
    create_data.main()

print("Importing Arrays")
X_train = np.load("data/arrays/X_train_128.npy")
y_train = np.load("data/arrays/y_train_128.npy")
X_test = np.load("data/arrays/X_test_128.npy")
y_test = np.load("data/arrays/y_test_128.npy")

model = model.model(128)
print(model.summary())

# model = keras.applications.inception_v3.InceptionV3(input_shape=(128, 128, 1), include_top=True, weights=None, classes=15)
# model = keras.applications.mobilenet_v2.MobileNetV2(input_shape=(128, 128, 1), include_top=True, weights=None, classes=15)
# model = keras.applications.vgg19.VGG19(input_shape=(128, 128, 1), include_top=True, weights=None, classes=15)

print("Loading from previous checkpoint")
model.load_weights("data/checkpoints/.ckpt")

print("Computing sample weights...")
class_weights = sk.utils.class_weight.compute_sample_weight('balanced', y_train)

checkpoint_path = "data/checkpoints/" + model_save_name + "-{epoch:03d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model.save_weights(checkpoint_path.format(epoch=0))

cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

model.fit(X_train, y_train, epochs=250, batch_size=64, validation_data=(X_test, y_test), class_weight=class_weights, callbacks=[cp_callback])

test_loss, test_acc = model.evaluate(X_test, y_test)
print('\nTest Accuracy: {}\nTest Loss: {}'.format(test_acc, test_loss))

model.save('data/models/{}.h5'.format(model_save_name))